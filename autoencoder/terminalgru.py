from keras.layers.recurrent import GRU
from keras import backend as K
import theano as T
from theano.tensor.extra_ops import squeeze
import numpy as np


def sampled_rnn(step_function, inputs, initial_states,
                go_backwards=False, mask=None, constants=None):
    '''Iterates over the time dimension of a tensor.
    # Arguments
        inputs: tensor of temporal data of shape (samples, time, ...)
            (at least 3D).
        step_function:
            Parameters:
                input: tensor with shape (samples, ...) (no time dimension),
                    representing input for the batch of samples at a certain
                    time step.
                states: list of tensors.
            Returns:
                output: tensor with shape (samples, ...) (no time dimension),
                new_states: list of tensors, same length and shapes
                    as 'states'.
        initial_states: tensor with shape (samples, ...) (no time dimension),
            containing the initial values for the states used in
            the step function.
        go_backwards: boolean. If True, do the iteration overx
            the time dimension in reverse order.
        mask: binary tensor with shape (samples, time),
            with a zero for every element that is masked.
        constants: a list of constant values passed at each step.
    # Returns
        A tuple (last_output, outputs, new_states).
            last_output: the latest output of the rnn, of shape (samples, ...)
            outputs: tensor with shape (samples, time, ...) where each
                entry outputs[s, t] is the output of the step function
                at time t for sample s.
            new_states: list of tensors, latest states returned by
                the step function, of shape (samples, ...).
    '''
    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)

    # if constants is None:
    #     constants = []

    if mask is not None:
        raise NotImplementedError("Mask is not doing anything right now :(")

    def _step(h, *states):
        output, new_states = step_function(h, states)
        return [output] + new_states

    results, updates = T.scan(_step,
                           sequences=inputs,
                           outputs_info=[None] + initial_states,
                           non_sequences=constants,
                           go_backwards=go_backwards)

    # deal with Theano API inconsistency
    if type(results) is list:
        outputs = results[0]
        states = results[1:]
    else:
        outputs = results
        states = []

    outputs = squeeze(outputs)
    last_output = outputs[-1][-1] # -1 for sampled output,

    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)[-1] # -1 for sampled output,
    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)

    states = [squeeze(state[-1]) for state in states]
    return last_output, outputs, states, updates


class TerminalGRU(GRU):
    '''GRU where the one-hot output of each neuron is fed into the next.
    In training it uses the actual training data, in testing it uses the multinomial
    sampled output of the previous neuron.
    '''
    def __init__(self, output_dim, temperature=1,
                 rnd_seed=None, **kwargs):
        super(TerminalGRU, self).__init__(output_dim, **kwargs)
        self.temperature = temperature
        self.rnd_seed = rnd_seed

    def build(self):
        self.Y = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_Y'.format(self.name))
        super(TerminalGRU, self).build()

        self.trainable_weights += [self.Y]

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [(initial_state, initial_state) for _ in range(len(self.states))]
        return initial_states

    def get_constants(self, x, train=False):
        # if train and (0 < self.dropout_U < 1):
        #     ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
        #     ones = K.concatenate([ones] * self.output_dim, 1)
        #     B_U = [K.dropout(ones, self.dropout_U) for _ in range(4)]
        #     return [B_U]
        return []

    def get_first_input(self, train=False):
        def recursive_input_getter(self, train=train):
            if hasattr(self, 'previous'):
                prev_input = self.previous.get_first_input(train=train)
            elif hasattr(self, 'input'):
                prev_input = self.input
            return prev_input
        first_input = recursive_input_getter(self, train=train)
        return first_input

    def get_output(self, train=False):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        mask = self.get_input_mask(train)

        assert K.ndim(X) == 3

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)
        constants = self.get_constants(X, train)
        preprocessed_input = self.preprocess_input(X, train)

        if train is True:
            initial_X = self.get_first_input(train=train)
            axes = [1, 0] + list(range(2, initial_X.ndim))
            initial_X = initial_X.dimshuffle(axes)
            zeros = K.zeros_like(initial_X[:1])
            initial_X = K.concatenate([zeros, initial_X[:-1]], axis=0)
            shifted_raw_inputs = initial_X.dimshuffle(axes)
            ## Silly concatenate to have same dimension as preprocessed inputs 3xoutput_dim
            shifted_raw_inputs = K.concatenate([shifted_raw_inputs,
                                                shifted_raw_inputs,
                                                shifted_raw_inputs], axis=2)
            all_inputs = K.stacklists([preprocessed_input, shifted_raw_inputs])
            ndim = all_inputs.ndim
            axes = [1, 2, 0] + list(range(3, ndim))
            all_inputs = all_inputs.dimshuffle(axes)
            self.train = True
        else:
            all_inputs = preprocessed_input
            self.train = False

        last_output, outputs, states, updates = sampled_rnn(self.step,
                                                              all_inputs,
                                                              initial_states,
                                                              go_backwards=self.go_backwards,
                                                              mask=mask,
                                                              constants=constants)

        del self.train
        self.updates = updates

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def step(self, h, states):
        prev_output = states[0][0]

        if len(states) == 2 and self.train:
            B_U = states[-1]
        elif len(states) == 1 or not self.train:
            B_U = np.array([1., 1., 1., 1.], dtype='float32')
        elif len(states) > 2:
            raise Exception('States has three elements')
        else:
            raise Exception('Should either be training with dropout,' +
                            ' training without it or predicting')

        #  If training and  h has an extra dimension, that is the input form the first_layer
        #  and is used as the sampled output from the previous node
        if h.ndim > 2 and self.train:
            axes = [1, 0] + list(range(2, h.ndim))
            h = h.dimshuffle(axes)
            prev_sampled_output = h[1][:, :self.output_dim]
            h = h[0]
        #  If not training h shouldn't have an extra dimension and we need to use the actual
        #  sampled output from the previous layer
        elif h.ndim <= 2 and not self.train:
            prev_sampled_output = states[0][1]
        else:
            raise Exception('Should either be training with first layer input or predicting'+
                            ' with previous output')

        x_z = h[:, :self.output_dim]
        x_r = h[:, self.output_dim: 2 * self.output_dim]
        x_h = h[:, 2 * self.output_dim:]

        z = self.inner_activation(x_z + K.dot(prev_output * B_U[0], self.U_z))
        r = self.inner_activation(x_r + K.dot(prev_output * B_U[1], self.U_r))

        hh = self.activation(x_h +
                             K.dot(r * prev_output * B_U[2], self.U_h) +
                             K.dot(r * prev_sampled_output * B_U[3], self.Y))

        output = z * prev_output + (1. - z) * hh

        if self.train is True:
            final_output = output
        else:
            sampled_output = output / K.sum(output,
                                            axis=-1, keepdims=True)

            sampled_output = K.log(sampled_output) / self.temperature
            exp_sampled = K.exp(sampled_output)
            norm_exp_sampled_output = exp_sampled / K.sum(exp_sampled,
                                                          axis=-1, keepdims=True)

            # Right now this is copying the same random number over and over
            # across both molecules and characters
            # ideally would have a different number
            if self.rnd_seed is not None:
                np.random.seed(self.rnd_seed)
                rand_matrix = np.random.uniform(size=(self.output_dim, ))
            # Right now this is copying the same random number over and over
            # across both molecules and characters
            # ideally would have a different number
            else:
                rand_matrix = K.random_uniform(shape=(self.output_dim, ))[:]

            cumul = K.cumsum(norm_exp_sampled_output, axis=-1)
            cumul_minus = cumul - norm_exp_sampled_output
            sampled_output = K.gt(cumul, rand_matrix) * K.lt(cumul_minus, rand_matrix)

            maxes = K.argmax(sampled_output, axis=-1)
            final_output = K.to_one_hot(maxes, self.output_dim)

        output_2d_tensor = K.stacklists([output, final_output])

        return output_2d_tensor, [output_2d_tensor]
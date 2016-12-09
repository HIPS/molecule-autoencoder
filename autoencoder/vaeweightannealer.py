from keras import backend as K
from keras.callbacks import Callback


class VAEWeightAnnealer(Callback):
    '''Weight of variational autoencoder scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            weight for the VAE (float).
    '''
    def __init__(self, schedule, VAE_layer_idx):
        super(VAEWeightAnnealer, self).__init__()
        self.schedule = schedule
        self.VAE_layer_idx = VAE_layer_idx

    def on_epoch_begin(self, epoch, logs={}):
        layer = self.model.layers[self.VAE_layer_idx]
        assert hasattr(layer, 'regularizer_scale'), \
            'Optimizer must have a "regularizer_scale" attribute.'
        weight = self.schedule(epoch)
        print("Current vae annealer weight is {}".format(weight))
        assert type(weight) == float, 'The output of the "schedule" function should be float.'
        K.set_value(layer.regularizer_scale, weight)

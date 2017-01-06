import argparse
import json
import logging
import os
from random import shuffle

import h5py
from keras.models import model_from_json
import numpy as np
from train_autoencoder import smile_convert

def adapt_model_dict(
    model_dict,
    regularizer_scale=1,
    rnd_seed=None,
    temperature=1,
    output_sample=False
):
    """
    Add in some custom options to the model json output from keras
    """
    updated = model_dict.copy()
    if "variationaldense" in updated:
        if "regularizer_scale" not in updated:
            logging.info('Adding a regularizer_scale = {} to the VAE layer'.format(regularizer_scale))
            updated["regularizer_scale"] = regularizer_scale
        if "output_sample" not in updated:
            logging.info('Adding output_sample = {} to the VAE layer'.format(output_sample))
            updated["output_sample"] = output_sample

    if "terminalgru" in updated:
        if "rnd_seed" not in updated:
            logging.info('Adding a rnd_seed parameter of {}'.format(rnd_seed))
            updated["rnd_seed"] = rnd_seed
        if "temperature" not in updated:
            logging.info('Adding a temperature parameter of {}'.format(temperature))
            updated["temperature"] = temperature
    return updated

def set_weights_from_file(weights_file, model):
    with h5py.File(weights_file, mode='r') as fp:
        for k in range(fp.attrs['nb_layers']):
            g = fp['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            w_shape = [i.shape for i in weights]
            logging.debug('Weights for this layer have shapes {}'.format(w_shape))
            try:
                model.layers[k].set_weights(weights)
            except AssertionError:
                logging.exception('Failed loading weights on layer {}. '
                                   'Weights initiated with random'.format(k))
                continue

def load_test_data(test_path, n_chars, max_len, char_list, limit=None):
    with open(test_path, 'r') as f:
        smiles = f.readlines()
    smiles = [s.strip() for s in smiles]
    if limit is not None:
        smiles = smiles[:limit]
    print('Training set size is', len(smiles))
    smiles = [smile_convert(i) for i in smiles if smile_convert(i)]
    print('Training set size is {}, after filtering to max length of {}'.format(len(smiles), max_len))
    shuffle(smiles)

    print(('total chars:', n_chars))

    cleaned_data = np.zeros((len(smiles), max_len, n_chars), dtype=np.float32)

    char_lookup = dict((c, i) for i, c in enumerate(char_list))

    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            cleaned_data[i, t, char_lookup[char]] = 1

    return cleaned_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sample a trained autoencoder.')
    parser.add_argument('model_file', type=str,
                        help='a file path of a model json file')
    parser.add_argument('weights_file', type=str,
                        help='a file path of a weights file')
    parser.add_argument('test_file', type=str,
                        help='a file path of a smiles list file to sample from')
    parser.add_argument('char_file', type=str,
                        help='a file path of a char index json')
    parser.add_argument('--limit', '-l', type=int, default=5000,
                        help='limit test data to this count')

    args = parser.parse_args()

    model_dict = json.load(open(args.model_file, 'r'))
    model_dict = adapt_model_dict(model_dict)

    model = model_from_json(json.dumps(model_dict))
    set_weights_from_file(args.weights_file, model)

    max_len = model_dict["layers"][0]["batch_input_shape"][1]
    n_chars = model_dict["layers"][0]["batch_input_shape"][2]

    char_list = json.load(open(args.char_file))
    test_set = load_test_data(args.test_file, n_chars, max_len, char_list, limit=args.limit)
    loss, accuracy = model.test_on_batch(test_set, test_set, sample_weight=None, accuracy=True)
    print("Loss: {}, Accuracy: {}".format(loss, accuracy))

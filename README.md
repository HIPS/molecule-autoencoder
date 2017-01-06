# Molecular Autoencoder

<img src="https://www.cs.toronto.edu/~duvenaud/pictures/autochem-icon.png" width="200">

This is the code used for the paper:

[Automatic chemical design using a data-driven continuous representation of molecules](https://arxiv.org/abs/1610.02415)


Abstract: We develop a molecular autoencoder, which converts discrete representations of molecules to and from a vector representation.
This allows efficient gradient-based optimization through open-ended spaces of chemical compounds.
Continuous representations also allow us to automatically generate novel chemical structures by performing simple operations in the latent space, such as interpolating between molecules.

By 
 * [Rafa Gómez-Bombarelli](http://aspuru.chem.harvard.edu/rafa-gomez-bombarelli/),
 * [David Duvenaud](https://www.cs.toronto.edu/~duvenaud/),
 * [José Miguel Hernández-Lobato](https://jmhl.org/),
 * [Jorge Aguilera-Iparraguirre](http://aspuru.chem.harvard.edu/jorge-aguilera/),
 * [Timothy Hirzel](https://www.linkedin.com/in/t1m0thy),
 * [Ryan P. Adams](http://people.seas.harvard.edu/~rpa/'),
 * [Alán Aspuru-Guzik](http://aspuru.chem.harvard.edu/about-alan/)

[bibtex file](https://www.cs.toronto.edu/~duvenaud/papers/molauto.bib) | [slides](https://www.cs.toronto.edu/~duvenaud/talks/mol-auto-talk.pdf)

### Notes
This code requires a fork of Keras that forked from the dev version around approximately version 0.3.2 and Theano > 0.8.2.  (Recently, to test on OS X 10.12.2, we are running Theano 0.9.0 dev4)  We want to point you to the work of Max Hodak who re-implemented this tool based on the paper.  For beginning your own project, you may have greater success starting there.  https://github.com/maxhodak/keras-molecules


#### To test the weights generated in the paper (limited to 5000 test SMILES)
        python sample_autoencoder.py \
            ../data/best_vae_model.json \
            ../data/best_vae_annealed_weights.h5 \
            ../data/250k_rndm_zinc_drugs_clean.smi \
            ../data/zinc_char_list.json \
            -l5000


Which should result is something close to this (values will range from random selection of 5000 samples from test file)

        Using Theano backend.
        ('Training set size is', 5000)
        Training set size is 5000, after filtering to max length of 120
        ('total chars:', 35)
        Loss: 0.834809958935, Accuracy: 0.948206666667


#### To train a new model (limit of 5000 training SMILES)
        python train_autoencoder.py \
            ../data/250k_rndm_zinc_drugs_clean.smi \
            ../data/zinc_char_list.json \
            -l5000
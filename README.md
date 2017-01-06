# molecule-autoencoder
A project to enable optimization of molecules by transforming them to and from a continuous representation.


This is the code used for the this paper:  https://arxiv.org/abs/1610.02415


This code requires a fork of Keras that forked from the dev version around approximately version 0.3.2 and Theano > 0.8.2.  (Recently, to test on OS X 10.12.2, we are running Theano 0.9.0 dev4)  We want to point you to the work of Max Hodak who re-implemented this tool based on the paper.  For beginning your own project, you may have greater success starting there.  https://github.com/maxhodak/keras-molecules


# To test the weights generated in the paper (limited to 5000 test SMILES)
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


# To train a new model (for quick test limit of 5000 training SMILES)
        python train_autoencoder.py \
            ../data/250k_rndm_zinc_drugs_clean.smi \
            ../data/zinc_char_list.json \
            -l5000
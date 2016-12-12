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

## Note
This code requires a fork of Keras that forked from the dev version around approximately version 0.3.2 and Theano 0.8.2.  We want to point you to the work of Max Hodak who re-implemented this tool based on the paper.  For beginning your own project, you may have greater success starting there.  https://github.com/maxhodak/keras-molecules

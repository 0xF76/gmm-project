.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/gmm-project.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/gmm-project
    .. image:: https://readthedocs.org/projects/gmm-project/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://gmm-project.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/gmm-project/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/gmm-project
    .. image:: https://img.shields.io/pypi/v/gmm-project.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/gmm-project/
    .. image:: https://img.shields.io/conda/vn/conda-forge/gmm-project.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/gmm-project
    .. image:: https://pepy.tech/badge/gmm-project/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/gmm-project
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/gmm-project

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===========
gmm-project
===========


    A from-scratch implementation of the **Gaussian Mixture Model (GMM)** based on the **Expectation-Maximization (EM)** algorithm.

This project was developed as part of the *Programming in Python Language* course at AGH University of Krakow. It includes:

- a public API simillar to ``sklearn.mixture.GaussianMixture``
- soft (``predict_proba``) and hard (``predict``) clustering.
- log-likelihood computation (``score``).

Project Structure
=================

The project uses the ``src/`` layout recommended by PyScaffold:

::

    gmm-project/
    ├── src/gmm/
    │   ├── __init__.py
    │   └── model.py
    ├── tests/
    ├── docs/
    ├── examples/
    ├── CHANGELOG.rst
    ├── README.rst
    └── pyproject.toml

Usage Example
=============

Basic usage of the custom GMM model::

    import numpy as np
    from gmm import GaussianMixture

    # Generate random data
    X = np.random.randn(300, 2)

    # Initialize model
    gmm = GaussianMixture(n_components=3, rng_seed=0)

    # Fit model
    gmm.fit(X)

    # Predict labels
    labels = gmm.predict(X)

    # Soft cluster probabilities
    probs = gmm.predict_proba(X)


Examples
=============
The ``examples/`` directory contains runnable demo scripts that illustrate the behavior of the implemented Gaussian Mixture Model.

Available examples include:

- **GMM vs. scikit-learn comparison**

  A script that fits both the custom and scikit implementation on the same dataset and compares:
    
  - estimated means
  - covariance matrices
  - mixture weights

- **Covariance ellipse visualization**
  
  A visualization demo that plots the learned Gaussian components as
  covariance ellipses overlaid on the input data.

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.

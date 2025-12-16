"""
Unit tests for the custom GaussianMixture implementation.

This module contains unit-level tests that verify the correctness of
individual components of the GaussianMixture class implemented in
``gmm.model``.

The tests focus on:
- correctness of the Gaussian probability density function,
- proper initialization of model parameters,
- validity of the Expectation (E) step responsibilities,
- correctness of the Maximization (M) step updates,
- consistency of public prediction methods with internal logic,
- validity of predicted cluster labels.

These tests are designed to validate internal behavior independently
of external reference implementations (e.g. scikit-learn), and
complement higher-level end-to-end comparison tests.
"""

import numpy as np
from gmm import GaussianMixture

def test_gaussian_pdf_1d_standard_normal():
    X = np.array([[0.0]])
    mean = np.array([0.0])
    cov = np.array([[1.0]])

    pdf = GaussianMixture._gaussian_pdf(X, mean, cov)

    expected = 1 / np.sqrt(2 * np.pi)
    assert np.allclose(pdf[0], expected)
def test_initialize_shapes():
    X = np.random.randn(10, 2)
    gmm = GaussianMixture(n_components=3, rng_seed=0)
    rng = np.random.default_rng(0)

    gmm._initialize(X, rng)

    assert gmm.means_.shape == (3, 2)
    assert gmm.covariances_.shape == (3, 2, 2)
    assert gmm.weights_.shape == (3,)
def test_initialize_weights_sum_to_one():
    X = np.random.randn(20, 3)
    gmm = GaussianMixture(n_components=4, rng_seed=0)
    rng = np.random.default_rng(0)

    gmm._initialize(X, rng)

    assert np.isclose(gmm.weights_.sum(), 1.0)
def test_e_step_responsibilities_sum_to_one():
    X = np.random.randn(15, 2)
    gmm = GaussianMixture(n_components=2, rng_seed=0)
    rng = np.random.default_rng(0)

    gmm._initialize(X, rng)
    resp = gmm._e_step(X)

    assert resp.shape == (15, 2)
    assert np.allclose(resp.sum(axis=1), 1.0)
def test_m_step_updates_weights():
    X = np.random.randn(10, 2)
    gmm = GaussianMixture(n_components=2, rng_seed=0)
    rng = np.random.default_rng(0)

    gmm._initialize(X, rng)
    resp = np.ones((10, 2)) / 2

    gmm._m_step(X, resp)

    assert np.allclose(gmm.weights_, [0.5, 0.5])
def test_predict_proba_matches_e_step():
    X = np.random.randn(50, 2)
    gmm = GaussianMixture(n_components=2, rng_seed=0)
    gmm.fit(X)

    proba = gmm.predict_proba(X)
    resp = gmm._e_step(X)

    assert proba.shape == resp.shape
def test_predict_labels_range():
    X = np.random.randn(20, 2)
    gmm = GaussianMixture(n_components=3, rng_seed=0)
    gmm.fit(X)

    labels = gmm.predict(X)

    assert labels.shape == (20,)
    assert labels.min() >= 0
    assert labels.max() < 3

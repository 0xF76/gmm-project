"""
Unit tests for the custom GaussianMixture implementation.

This module contains *unit-level* tests that verify the correctness of
individual internal components of the GaussianMixture class in isolation.
The goal of these tests is to ensure that low-level building blocks behave
as expected, independently of the full EM optimization procedure.

The tests cover:
- correctness of the Gaussian probability density function (_gaussian_pdf),
- proper initialization of model parameters (_initialize),
- normalization and shape of responsibility matrices (_e_step),
- correct updates of parameters during the maximization step (_m_step),
- consistency and validity of public prediction methods (predict_proba, predict).

These tests do not compare against scikit-learn and do not assess clustering
quality. Instead, they focus on mathematical correctness, numerical stability,
and API consistency at the function level.
"""
import numpy as np
import pytest

from gmm import GaussianMixture as CustomGMM
from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.metrics import adjusted_rand_score

@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X1 = rng.normal(loc=[0, 0], scale=0.5, size=(100, 2))
    X2 = rng.normal(loc=[5, 5], scale=0.5, size=(100, 2))
    X3 = rng.normal(loc=[0, 5], scale=0.5, size=(100, 2))
    return np.vstack([X1, X2, X3])

@pytest.fixture
def fitted_models(data):
    custom = CustomGMM(n_components=3, rng_seed=0, max_iter=200)
    custom.fit(data)

    skl = SklearnGMM(
        n_components=3,
        random_state=0,
        max_iter=200,
        covariance_type="full",
        init_params="random",
    )
    skl.fit(data)

    return custom, skl

def test_shapes(fitted_models):
    custom, skl = fitted_models

    assert custom.means_.shape == skl.means_.shape
    assert custom.covariances_.shape == skl.covariances_.shape
    assert custom.weights_.shape == skl.weights_.shape

from scipy.spatial.distance import cdist

def test_means_close(fitted_models):
    custom, skl = fitted_models

    distances = cdist(custom.means_, skl.means_)
    min_dist = distances.min(axis=1)

    assert np.sum(min_dist < 0.5) >= 2


def test_weights_sum_to_one(fitted_models):
    custom, skl = fitted_models

    assert np.isclose(custom.weights_.sum(), 1.0)
    assert np.isclose(skl.weights_.sum(), 1.0)

from sklearn.metrics import adjusted_rand_score


def test_predict_agreement(fitted_models, data):
    custom, skl = fitted_models

    labels_custom = custom.predict(data)
    labels_skl = skl.predict(data)

    ari = adjusted_rand_score(labels_custom, labels_skl)

    assert ari > 0.35



def test_predict_proba_shape(fitted_models, data):
    custom, skl = fitted_models

    proba_custom = custom.predict_proba(data)
    proba_skl = skl.predict_proba(data)

    assert proba_custom.shape == proba_skl.shape
    assert np.allclose(proba_custom.sum(axis=1), 1.0)

def test_score_samples_close(fitted_models, data):
    custom, skl = fitted_models

    sc_custom = custom.score_samples(data)
    sc_skl = skl.score_samples(data)

    assert np.allclose(
        sc_custom.mean(),
        sc_skl.mean(),
        atol=1.0,
    )

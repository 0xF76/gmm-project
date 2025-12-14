"""
Demo: Comparison of custom GMM with scikit-learn implementation.

This script fits both models on the same dataset and compares
their learned parameters and clustering behavior.
"""

from gmm import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as SklearnGMM

def generate_random_gmm(K=3, N=600, seed=0):
    rng = np.random.default_rng(seed)

    true_means = rng.uniform(low=-10, high=10, size=(K, 2))

    true_covs = []
    for _ in range(K):
        A = rng.normal(size=(2, 2))
        cov = A @ A.T + 0.5 * np.eye(2)
        true_covs.append(cov)
    true_covs = np.array(true_covs)

    w = rng.uniform(size=K)
    true_weights = w / w.sum()

    z = rng.choice(K, size=N, p=true_weights)

    X_list = []
    for k in range(K):
        nk = np.sum(z == k)
        if nk > 0:
            X_list.append(rng.multivariate_normal(true_means[k], true_covs[k], size=nk))

    X = np.vstack(X_list)
    return X, true_means, true_covs, true_weights


def sort_by_x(means, weights, covs):
    order = np.argsort(means[:, 0])
    return weights[order], means[order], covs[order]


def main():
    
    X, true_means, true_covs, true_weights = generate_random_gmm(K=3, N=1000, seed=42)

    gmm = GaussianMixture(n_components=3, max_iter=1000, tol=1e-4, rng_seed=0)
    gmm.fit(X)

    skgmm = SklearnGMM(n_components=3, covariance_type='full', max_iter=1000, tol=1e-4, random_state=0)
    skgmm.fit(X)

    true_weights_s, true_means_s, true_covs_s = sort_by_x(true_means, true_weights, true_covs)
    est_weights_s, est_means_s, est_covs_s = sort_by_x(gmm.means_, gmm.weights_, gmm.covariances_)
    sk_weights_s, sk_means_s, sk_covs_s = sort_by_x(skgmm.means_, skgmm.weights_, skgmm.covariances_)

    print("True Weights:\n", true_weights_s)
    print("Estimated Weights:\n", est_weights_s)
    print("\nSklearn Weights:\n", sk_weights_s)

    print("\nTrue Means:\n", true_means_s)
    print("Estimated Means:\n", est_means_s)
    print("Sklearn Means:\n", sk_means_s)

    print("\nTrue Covariances:\n", true_covs_s)
    print("Estimated Covariances:\n", est_covs_s)
    print("Sklearn Covariances:\n", sk_covs_s)

    print("\nDifference in Weights:", np.linalg.norm(sk_weights_s - est_weights_s))
    print("Difference in Means:", np.linalg.norm(sk_means_s - est_means_s))
    cov_diff = np.linalg.norm(sk_covs_s - est_covs_s)
    print("Difference in Covariances:", cov_diff)


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], s=5, alpha=0.5)
    plt.title("Raw Data")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")

    resp = gmm.predict_proba(X)
    cluster_assignments = np.argmax(resp, axis=1)
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments, s=5, cmap='viridis', alpha=0.5)
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=80, marker='x')

    plt.title("GMM Cluster Assignments")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()


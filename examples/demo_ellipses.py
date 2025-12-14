import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from gmm import GaussianMixture

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

def add_sigma_ellipses(ax, mean, cov, color, sigmas=(1, 2, 3), base_alpha=0.70):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    for s in sigmas:
        width = 2.0 * s * np.sqrt(vals[0])
        height = 2.0 * s * np.sqrt(vals[1])

        alpha = base_alpha / (s ** 1.3)

        e = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            linewidth=2,
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(e)


def main():
    X, _, _, _ = generate_random_gmm(K=3, N=1000, seed=42)

    gmm = GaussianMixture(n_components=3, max_iter=1000, tol=1e-4, rng_seed=0)
    gmm.fit(X)

    labels = gmm.predict(X)
    base_colors = list(plt.cm.tab10.colors)
    colors = [base_colors[k % len(base_colors)] for k in range(3)]

    plt.figure(figsize=(16, 9))
    point_colors = [colors[int(k)] for k in labels]
    plt.scatter(X[:, 0], X[:, 1], c=point_colors, s=14, alpha=0.75, zorder=3)
    for k in range(3):
        add_sigma_ellipses(plt.gca(), gmm.means_[k], gmm.covariances_[k], colors[k], sigmas=(1, 2, 3))
        plt.scatter(gmm.means_[k, 0], gmm.means_[k, 1], c='black', marker="x", s=140, linewidths=3, zorder=4)
    plt.title("GMM result with covariance ellipses")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

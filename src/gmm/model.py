import numpy as np

class GaussianMixture:
    """
    Gaussian Mixture Model implementation using the Expectation-Maximization algorithm.

    Parameters
    ----------
    n_components : int
        Number of mixture components (clusters).
    max_iter : int, default=100
        Maximum number of iterations for the EM algorithm.
    tol : float, default=1e-4
        Convergence threshold. EM iterations will stop when the log-likelihood
        change is below this value.
    rng_seed : int, default=0
        Optional random seed.
    """

    def __init__(self, n_components: int, max_iter: int=100, tol: float=1e-4, rng_seed: int=0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.rng_seed = rng_seed

        #learned parameters
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    @staticmethod
    def _gaussian_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Multivariate Gaussian Probability Density Function evaluated at X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points at which to evaluate the PDF.
        mean : ndarray of shape (n_features,)
            Mean of the Gaussian distribution.
        cov : ndarray of shape (n_features, n_features)
            Covariance matrix of the Gaussian distribution.

        Returns
        -------
        pdf: ndarray of shape (n_samples,)
            Probability density function values for each data point in ``X``.
        """
        d = X.shape[1]
        diff = X - mean

        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)

        #The exponent term for Gaussian: (X - mean)^T * cov_inv * (X - mean)
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        #The normalization constant
        norm_const = np.sqrt((2 * np.pi) ** d * cov_det)

        return np.exp(exponent) / norm_const
    
    def _initialize(self, X: np.ndarray, rng: np.random.Generator) -> None:
        """
        Initialize mixture parameters.

        This method chooses random data points as initial means,
        initializes all covariances to the global covariance of the data,
        and sets equal weights for all components.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input data points.
        rng: np.random.Generator
            Random number generator used for choosing initial centers.
        """
        N, d = X.shape
        
        #Pick random data points as initial means
        initial_cluster_centers = rng.choice(N, size=self.n_components, replace=False)
        self.means_ = X[initial_cluster_centers]

        #Initialize covariances to global covariance
        cov_global = np.cov(X, rowvar=False)
        self.covariances_ = np.array([cov_global.copy() for _ in range(self.n_components)])

        #Equal weights
        self.weights_ = np.ones(self.n_components) / self.n_components

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Perform the Expectation step of the EM algorithm.
        Compute the responsibilities for each data point.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.

        Returns
        -------
        resp : ndarray of shape (n_samples, n_components)
            Responsibilities for each data point and component.
            In other words, probability that each data point belongs to each component.
        """
        N, d = X.shape

        resp = np.zeros((N, self.n_components))
        for k in range(self.n_components):
            resp[:, k] = self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])

        resp_sum = resp.sum(axis=1, keepdims=True) + 1e-12  # avoid division by zero
        resp /= resp_sum
        return resp
    
    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        """
        Perform the Maximization step of the EM algorithm.
        Update the mixture parameters based on the current responsibilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        resp : ndarray of shape (n_samples, n_components)
            Responsibilities computed in the E-step.
        """
        N, d = X.shape

        Nk = resp.sum(axis=0)
        self.weights_ = Nk / N
        self.means_ = (resp.T @ X) / Nk[:, None]

        self.covariances_ = np.zeros((self.n_components, d, d))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = resp[:, k][:, None] * diff
            self.covariances_[k] = (weighted_diff.T @ diff) / Nk[k]

    def _log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute the log-likelihood of the data given the current mixture parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        """
        N, d = X.shape
        total = np.zeros(N)
        for k in range(self.n_components):
            total += self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])
        return np.sum(np.log(total + 1e-12))
    
    #--------------------------------------
    # Public API
    #--------------------------------------
    def fit(self, X: np.ndarray) -> "GaussianMixture":
        """
        Fit the Gaussian Mixture Model to the data using the EM algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        
        Returns
        -------
        self : GaussianMixture
            The fitted GaussianMixture instance.
        """

        rng = np.random.default_rng(self.rng_seed)
        
        self._initialize(X, rng)
        prev_ll = None

        for _ in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            ll = self._log_likelihood(X)

            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break

            prev_ll = ll
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute responsibilities for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.

        Returns
        -------
        resp : ndarray of shape (n_samples, n_components)
            Responsibilities for each data point and component.
            In other words, probability that each data point belongs to each component.
        """
        return self._e_step(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample tot he most likely component (hard clustering).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Assigned component labels for each data point.
        """
        resp = self.predict_proba(X)
        return np.argmax(resp, axis=1)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the log-probability density of each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.

        Returns
        -------
        log_p: ndarray of shape (n_samples,)
            Log-probability density of each data point.
        """
        N, d = X.shape
        total = np.zeros(N)
        for k in range(self.n_components):
            total += self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k])
        return np.log(total + 1e-12) # avoid log(0)





=========
Changelog
=========

Version 0.1
===========

- Implemented the core Gaussian Mixture Model using the Expectation-Maximization algorithm.
- Internal functions:
    - ``_gausian_pdf``: Computes the Gaussian probability density function.
    - ``_e_step``: Performs the Expectation step of the EM algorithm.
    - ``_m_step``: Performs the Maximization step of the EM algorithm.
    - ``_log_likelihood``: Calculates the log-likelihood of the data given the current parameters.
- Public methods:
    - ``fit``: Fits the GMM to the provided data.
- Basic EM loop with convergence threshold.

Version 0.2
===========
- Added public API methods:
    - ``predict``: Predicts the cluster labels for the input data.
    - ``score``: Computes the log-likelihood of the data under the model.
- Changelog documenting version history.
- Add documentation for all public methods.


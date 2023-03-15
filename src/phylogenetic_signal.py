import numpy as np
import ete3
from scipy.optimize import minimize


class PagelsLambda(object):
    def __init__(self, tree, **kwargs):
        """Initialize PagelsLambda object. Takes ete tree or path to tree file."""
        if isinstance(tree, str):
            tree = ete3.Tree(tree, **kwargs)
        self.tree = tree
        self.leaf_order = [leaf.name for leaf in tree.get_leaves()]
        self.N = len(tree.get_leaves())

        # Raw covariance matrix has C[i,j] = d(MRCA(i,j), root)
        self.C = np.zeros((self.N, self.N))
        for i, leaf_i in enumerate(self.tree):
            for j, leaf_j in enumerate(self.tree):
                # If i == j, then the distance is leaf to root
                mrca = self.tree.get_common_ancestor(leaf_i, leaf_j)
                self.C[i, j] = mrca.get_distance(self.tree)

    def fit(self, x: np.ndarray, y=None, method="optimize") -> None:
        """
        Fit Pagels lambda to a set of traits.

        Args:
            x: (N, 1) array of trait values
            y: ignored (for sklearn compatibility)
            method: "grid" or "optimize":
                "grid" uses a grid search to find the best lambda
                "optimize" uses scipy.optimize.minimize to find the best lambda

        Returns:
            None (sets self.lam)
        """

        if x.ndim == 1:
            x = x.reshape(x.shape[0], 1)

        # Remove missing values: need to remove from covariance matrix too
        missing = np.isnan(x)[:, 0]
        x = x[~missing, :]
        C = self.C[~missing, :][:, ~missing]

        # Compute null: C[i,j] = d(i,j)
        self.null_lnL = self.mle(x, self.rescale_cov(0, cov=C))[2]

        # Maximum likelihood estimation
        if method == "grid":
            max_ll = -np.inf
            lam_mle = None
            for lam in np.linspace(0, 1, 101):
                C_lam = self.rescale_cov(lam, cov=C)
                z0, sigma2, ll = self.mle(x, C_lam)
                if ll > max_ll:
                    max_ll = ll
                    lam_mle = lam
            self.lam = lam_mle
            self.lnL = max_ll

        elif method == "optimize":

            def neg_ll(lam):
                C_lam = self.rescale_cov(lam, cov=C)
                z0, sigma2, ll = self.mle(x, C_lam)
                return -ll

            # Trying some more robust try/except structure here
            try:
                res = minimize(neg_ll, x0=0.5, bounds=[(0, 1)])
                self.lam = res.x[0]
                self.lnL = -res.fun
                return
            except:
                pass
            try:
                res = minimize(
                    neg_ll, x0=0.5, bounds=[(0, 1)], method="Nelder-Mead"
                )
                self.lam = res.x[0]
                self.lnL = -res.fun
                return
            except:
                self.lam = np.nan
                self.lnL = np.nan
                return
        else:
            raise ValueError("Unknown method for fitting Pagels lambda.")

    def rescale_cov(self, lam: float, cov: np.ndarray = None) -> np.ndarray:
        """
        Rescale covariance matrix by lambda.

        Args:
            lam: lambda value
            cov: covariance matrix to rescale. If None, use self.C.

        Returns:
            Rescaled covariance matrix.
        """
        if cov is None:
            cov = self.C

        C_lam = cov.copy() * lam
        np.fill_diagonal(C_lam, cov.diagonal())

        return C_lam

    def mle(self, x: np.ndarray, C_lam: np.ndarray) -> (float, float, float):
        """
        Estimate z0 and sigma2 for Brownian motion, plus log-likelihood.

        Args:
            x: (M, 1) array of trait values for M <= N
            C_lam: (M, M) rescaled covariance matrix

        Returns:
            z0: value at root
            sigma2: rate of evolution
            ll: log-likelihood of the data given the model
        """

        N = len(x)

        C_inv = np.linalg.pinv(C_lam)

        # First, get z0
        one = np.ones(shape=(N, 1))
        # z0_front = np.linalg.pinv(one.T @ C_inv @ one)
        # z0_end = one.T @ C_inv @ x
        # z0 = z0_front * z0_end
        z0 = (np.linalg.pinv(one.T @ C_inv @ one) @ (one.T @ C_inv @ x)).item()

        # Next, get sigma2
        x0 = x - z0 * one  # (N, 1)
        sigma2 = x0.T @ C_inv @ x0 / N  # (1, N) @ (N, N) @ (N, 1) = (1, 1)

        # temp = XSubZ0_vector.T @ pinv(deltaSquare * C)) @ XSubZ0_vector
        ll_num = -0.5 * x0.T @ np.linalg.pinv(sigma2 * C_lam) @ x0
        ll_denom = 0.5 * (
            N * np.log(2 * np.pi) + np.linalg.slogdet(sigma2 * C_lam)[1]
        )
        ll = (ll_num - ll_denom)[0][0]

        return z0, sigma2, ll

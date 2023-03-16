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

    def fit(
        self,
        x: np.ndarray,
        y=None,
        method: str = "optimize",
        unbiased: bool = False,
    ) -> None:
        """
        Fit Pagels lambda to a set of traits.

        Args:
            x: (N, 1) array of trait values
            y: ignored (for sklearn compatibility)
            method: "grid" or "optimize":
                "grid" uses a grid search to find the best lambda
                "optimize" uses scipy.optimize.minimize to find the best lambda
            unbiased: if True, use unbiased estimator for sigma2

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
                z0, sigma2, ll = self.mle(x, C_lam, unbiased=unbiased)
                if ll > max_ll:
                    max_ll = ll
                    lam_mle = lam
            self.lam = lam_mle
            self.lnL = max_ll

        elif method == "optimize":

            def neg_ll(lam):
                C_lam = self.rescale_cov(lam, cov=C)
                z0, sigma2, ll = self.mle(x, C_lam, unbiased=unbiased)
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

    def mle(
        self, x: np.ndarray, C_lam: np.ndarray, unbiased: bool = False
    ) -> (float, float, float):
        """
        Estimate z0 and sigma2 for Brownian motion, plus log-likelihood.

        Args:
            x: (M, 1) array of trait values for M <= N
            C_lam: (M, M) rescaled covariance matrix
            unbiased: if True, use unbiased estimator for sigma2

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
        sigma2 = x0.T @ C_inv @ x0  # (1, N) @ (N, N) @ (N, 1) = (1, 1)
        if unbiased:
            sigma2 = sigma2 / (N - 1)
        else:
            sigma2 = sigma2 / N
        sigma2 = sigma2.item()

        # Finally, get log-likelihood
        ll_num = -0.5 * x0.T @ np.linalg.pinv(sigma2 * C_lam) @ x0
        ll_denom = 0.5 * (
            N * np.log(2 * np.pi) + np.linalg.slogdet(sigma2 * C_lam)[1]
        )
        ll = (ll_num - ll_denom).item()

        return z0, sigma2, ll


class PagelsLambdaMulti(PagelsLambda):
    def __init__(self, tree, **kwargs):
        super().__init__(tree, **kwargs)

    def rescale_covs(
        self, lam: np.ndarray, cov: np.ndarray = None
    ) -> np.ndarray:
        """Apply original rescale_cov for each sample"""
        # return np.array([super().rescale_cov(l, cov) for l in lam])
        # Vectorized version using np.apply_along_axis
        return np.array([self.rescale_cov(l, cov) for l in lam])

    def mle(
        self, X: np.ndarray, C_lam: np.ndarray, unbiased: bool = False
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Estimate z0 and sigma2 for Brownian motion, plus log-likelihood.

        Args:
            X: (n_samples, n_taxa, 1) array of trait values for M <= N
            C_lam: (n_samples, n_taxa, n_taxa) stack of rescaled cov. matrices
            unbiased: if True, use unbiased estimator for sigma2

        Returns:
            z0: (n_samples, ) vector of values at root
            sigma2: (n_samples, ) vector of rates of evolution
            ll: (n_samples, ) vector of log-likelihoods of data given model
        """

        # print("X", X.shape, "should be (69, 100, 1)")

        # N = x.shape[1]
        M, N, _ = X.shape  # (n_samples, n_taxa, 1)

        # C_inv_2d = np.linalg.pinv(C_lam)  # (N, N)
        # C_inv = np.repeat(C_inv_2d[np.newaxis, :, :], M, axis=0)  # (M, N, N)
        C_inv = np.linalg.pinv(C_lam)  # (M, N, N)

        # print("C", C_inv.shape, "should be (69, 100, 100)")

        # # First, get z0
        one = np.ones(shape=(M, N, 1))  # (M, N, 1)
        oneT = one.transpose(0, 2, 1)  # (M, 1, N)
        # print("one", one.shape, "should be (69, 100, 1)")
        # print("oneT", one.T.shape, "should be (69, 1, 100)")
        z0 = np.linalg.pinv(
            (oneT @ C_inv @ one) @ (oneT @ C_inv @ X)
        )  # (M, 1, N) @ (M, N, N) @ (M, N, 1) = (M, 1, 1)

        # # the pinv thing is unstable. Let's do a loop:
        # z0_loop = np.zeros((M, 1, 1))
        # one_2d = np.ones((N, 1))
        # tmp_a = one_2d.T @ C_inv_2d @ one_2d
        # tmp_b = one_2d.T @ C_inv_2d
        # for i in range(M):
        #     # z0[i, 0, 0] = np.linalg.pinv(oneT[i] @ C_inv[i] @ one[i]) @ (oneT[i] @ C_inv[i] @ X[i])
        #     z0_case = np.linalg.pinv(tmp_a) @ (tmp_b @ X[i])
        #     print("z0_case", z0_case.shape, "should be (1, 1)")
        #     z0_loop[i, 0, 0] = z0_case

        # print("z0", z0.shape, "should be (69, 1, 1)")
        # print("z0_loop", z0_loop.shape, "should be (69, 1, 1)")

        # print("Difference:", np.linalg.norm(z0 - z0_loop))

        # Next, get sigma2
        x0 = X - z0 * one  # (M, N, 1)
        # print("x0", x0.shape, "should be (69, 100, 1)")
        x0T = x0.transpose(0, 2, 1)  # (M, 1, N)
        # print("x0T", x0T.shape, "should be (69, 1, 100)")
        sigma2 = (
            x0T @ C_inv @ x0
        )  # (M, 1, N) @ (M, N, N) @ (M, N, 1) = (M, 1, 1)
        if unbiased:
            sigma2 = sigma2 / (N - 1)
        else:
            sigma2 = sigma2 / N
        # print("sigma2", sigma2.shape, "should be (69, 1, 1)")

        # Finally, get log-likelihood
        ll_num = -0.5 * x0T @ np.linalg.pinv(sigma2 * C_lam) @ x0  # (M, 1, 1)
        # print("ll_num", ll_num.shape, "should be (69, 1, 1)")
        ll_denom = 0.5 * (
            N * np.log(2 * np.pi) + np.linalg.slogdet(sigma2 * C_lam)[1]
        ).reshape(
            M, 1, 1
        )  # (M, 1, 1)
        # print("ll_denom", ll_denom.shape, "should be (69, 1, 1)")
        ll = ll_num - ll_denom  # (M, 1, 1)
        # print("ll", ll.shape, "should be (69, 1, 1)")

        return z0.flatten(), sigma2.flatten(), ll.flatten()

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        Fit Pagels lambda to a matrix of traits.

        Args:
            X: (n_samples, n_taxa, 1) array of trait values
            y: ignored (for sklearn compatibility)
            method: "grid" or "optimize":
                "grid" uses a grid search to find the best lambda
                "optimize" uses scipy.optimize.minimize to find the best lambda

        Returns:
            None (sets self.lam)
        """

        # Still works for 1D arrays
        # print("ndim", X.ndim)
        if X.ndim == 1:
            X = X.reshape(1, X.shape[0], 1)
        elif X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
            print("Reshaped X to", X.shape)
        # print(X.shape)

        # Find columns (taxa) with any missing or inf values
        to_remove = np.logical_or(
            np.isnan(X).any(axis=0), np.isinf(X).any(axis=0)
        ).squeeze()  # (N, )
        print(to_remove.shape)
        X = X[:, ~to_remove, :]
        C = self.C[~to_remove, :][:, ~to_remove]

        # # log-likelihood of null model, vectorized (still C[i,j] = d(i,j)):
        # self.null_lnL = np.apply_along_axis(
        #     self.mle(X, self.rescale_cov(0, cov=C))[2], 0, X
        # )  # TODO: verify this matches repeated calls to self.mle
        # TODO: implement this

        # Maximum likelihood estimation - skip grid search
        def neg_ll(lams):
            C_lam = self.rescale_covs(lams, cov=C)
            z0s, sigma2s, lls = self.mle(X, C_lam)
            return -lls.sum()

        res = minimize(
            neg_ll, x0=0.5 * np.ones(X.shape[0]), bounds=[(0, 1)] * X.shape[0]
        )
        self.lam = res.x[0]
        self.lnL = -res.fun

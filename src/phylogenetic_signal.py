import numpy as np
import pandas as pd
import ete3
from scipy.optimize import minimize
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch


def _get_mrca(i, N, leaves, tree):
    leaf_i = leaves[i]
    dists = []
    for j in range(N):
        leaf_j = leaves[j]
        mrca = tree.get_common_ancestor(leaf_i, leaf_j)
        dists.append(mrca.get_distance(tree))
    return np.array(dists)


class PagelsLambda(object):
    def __init__(
        self, tree, memoized=False, multiprocess=True, matrix=None, **kwargs
    ):
        """Initialize PagelsLambda object. Takes ete tree or path to tree file."""

        self.memoized = memoized
        self.multiprocess = multiprocess

        if isinstance(tree, str):
            tree = ete3.Tree(tree, **kwargs)
        self.tree = tree
        leaves = tree.get_leaves()
        self.leaf_order = [leaf.name for leaf in leaves]
        self.N = len(leaves)

        # Raw covariance matrix has C[i,j] = d(MRCA(i,j), root)
        self.C = np.zeros((self.N, self.N))

        # Multiprocessing version
        if matrix is not None:
            self.C = matrix
        elif multiprocess:
            pool = Pool(12)
            func = partial(_get_mrca, N=self.N, leaves=leaves, tree=self.tree)

            for i, dists in enumerate(
                tqdm(pool.imap(func, range(self.N)), total=self.N)
            ):
                self.C[i, :] = dists
        else:
            for i in trange(self.N):
                leaf_i = leaves[i]
                for j in range(i, self.N):
                    leaf_j = leaves[j]
                    mrca = self.tree.get_common_ancestor(leaf_i, leaf_j)
                    self.C[i, j] = mrca.get_distance(self.tree)
                    if i != j:  # Symmetric
                        self.C[j, i] = self.C[i, j]

        self.memos = {}  # Memoize C-inverse for speed :)

    def fit(
        self,
        x: np.ndarray,
        y=None,
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

        if isinstance(x, list):
            x = np.array(x)

        if isinstance(x, pd.Series):
            x = x.values

        if x.ndim == 1:
            x = x.reshape(x.shape[0], 1)

        # Remove missing values: need to remove from covariance matrix too
        missing = np.isnan(x)[:, 0]
        x = x[~missing, :]
        C = self.C[~missing, :][:, ~missing]

        # Compute null: C[i,j] = d(i,j)
        self.null_lnL = self.mle(x, 0, C=C, unbiased=unbiased)[2]

        # Maximum likelihood estimation
        def neg_ll(lam):
            lam = float(lam)
            # if self.memoized:
            #     lam = np.round(lam, 4)  # Round to 4 decimal places
            lam = np.round(lam, 10)
            z0, sigma2, ll = self.mle(x, lam, C=C, unbiased=unbiased)
            return -ll

        res = minimize(
            neg_ll,
            x0=0.5,
            bounds=[(0, 1)],
            tol=1e-6,
        )
        self.lam = res.x[0]
        self.lnL = -res.fun
        return

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
        self,
        x: np.ndarray,
        lam: float,
        C: np.ndarray = None,
        unbiased: bool = False,
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

        if C is None:
            C = self.C

        if self.memoized and lam in self.memos:
            C_inv = self.memos[lam]
            # C_logdet = self.memos[f"{lam}_logdet"]

        else:
            C_lam = self.rescale_cov(lam, cov=C)
            # C_inv = np.linalg.pinv(C_lam)
            C_inv = torch.inverse(torch.tensor(C_lam)).numpy()  # Faster
            # C_logdet = np.linalg.slogdet(C_lam)[1]
            if self.memoized:
                self.memos[lam] = C_inv
                # self.memos[f"{lam}_logdet"] = C_logdet

        C_logdet = -np.linalg.slogdet(C_inv)[1]

        # First, get z0
        z0 = (C_inv @ x).sum() / C_inv.sum().sum()  # No inversion
        x0 = x - z0

        # Next, get sigma2
        sigma2 = x0.T @ C_inv @ x0  # (1, N) @ (N, N) @ (N, 1) = (1, 1)
        if unbiased:
            sigma2 = sigma2 / (N - 1)
        else:
            sigma2 = sigma2 / N
        sigma2 = sigma2.item()

        # Finally, get log-likelihood
        return (
            z0,
            sigma2,
            (1 - C_logdet - N * (1 + np.log(2 * np.pi * sigma2))) / 2,
        )

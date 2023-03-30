import torch
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
        self.C = torch.zeros((self.N, self.N))
        for i, leaf_i in enumerate(self.tree):
            for j, leaf_j in enumerate(self.tree):
                # If i == j, then the distance is leaf to root
                mrca = self.tree.get_common_ancestor(leaf_i, leaf_j)
                self.C[i, j] = mrca.get_distance(self.tree)

    def fit(
        self,
        x: torch.Tensor,
        y=None,
        method: str = "optimize",
        unbiased: bool = False,
    ) -> None:
        """
        Fit Pagels lambda to a set of traits.

        Args:
            x: (N, 1) tensor of trait values
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
        missing = torch.isnan(x)[:, 0]
        x = x[~missing, :]
        C = self.C[~missing, :][:, ~missing]

        # Compute null: C[i,j] = d(i,j)
        self.null_lnL = self.mle(x, self.rescale_cov(0, cov=C))[2]

        # Maximum likelihood estimation
        if method == "grid":
            max_ll = -float("inf")
            lam_mle = None
            for lam in torch.linspace(0, 1, 101):
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
                self.lam = float("nan")
                self.lnL = float("nan")
                return
        else:
            raise ValueError("Unknown method for fitting Pagels lambda.")

    def rescale_cov(self, lam: float, cov: torch.Tensor = None) -> torch.Tensor:
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

        C_lam = cov.clone() * lam
        # torch.fill_diagonal(C_lam, cov.diagonal())
        C_lam += (1 - lam) * torch.eye(self.N)

        return C_lam

    def mle(
        self, x: torch.Tensor, C_lam: torch.Tensor, unbiased: bool = False
    ) -> (float, float, float):
        """
        Estimate z0 and sigma2 for Brownian motion, plus log-likelihood.

        Args:
            x: (M, 1) tensor of trait values for M <= N
            C_lam: (M, M) rescaled covariance matrix
            unbiased: if True, use unbiased estimator for sigma2

        Returns:
            z0: value at root
            sigma2: rate of evolution
            ll: log-likelihood of the data given the model
        """

        N = len(x)

        C_inv = torch.linalg.pinv(C_lam)

        # First, get z0
        one = torch.ones(size=(N, 1))
        z0 = (
            torch.linalg.pinv(one.T @ C_inv @ one) @ (one.T @ C_inv @ x)
        ).item()

        # Next, get sigma2
        x0 = x - z0 * one  # (N, 1)
        sigma2 = x0.T @ C_inv @ x0  # (1, N) @ (N, N) @ (N, 1) = (1, 1)
        if unbiased:
            sigma2 = sigma2 / (N - 1)
        else:
            sigma2 = sigma2 / N
        sigma2 = sigma2.item()

        # Finally, get log-likelihood
        ll_num = -0.5 * x0.T @ torch.linalg.pinv(sigma2 * C_lam) @ x0
        ll_denom = 0.5 * (
            N * torch.log(2 * torch.tensor([3.141592653589793]))
            + torch.linalg.slogdet(sigma2 * C_lam)[1]
        )
        ll = (ll_num - ll_denom).item()

        return z0, sigma2, ll

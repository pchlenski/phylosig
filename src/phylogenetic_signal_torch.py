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
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
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
        def neg_ll(lam):
            C_lam = self.rescale_cov(lam, cov=C)
            z0, sigma2, ll = self.mle(x, C_lam, unbiased=unbiased)
            return -ll

        def closure():
            optimizer.zero_grad()
            loss = neg_ll(lam_tensor)
            loss.backward()
            # print("Gradients:", lam_tensor.grad)
            return loss

        lam_tensor = torch.tensor([0.5], requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [lam_tensor],
            lr=0.01,
            max_iter=100,
            line_search_fn="strong_wolfe",
        )
        prev_loss = float("inf")

        for _ in range(max_iterations):
            loss = optimizer.step(closure)
            # optimizer.zero_grad()
            # loss = neg_ll(lam_tensor)
            # loss.backward()
            if abs(prev_loss - loss) < tolerance:
                break
            prev_loss = loss
            print(lam_tensor.item(), loss)
            if lam_tensor < 0:
                lam_tensor = torch.tensor([0.0])
                break
            if lam_tensor > 1:
                lam_tensor = torch.tensor([1.0])
                break

        self.lam = lam_tensor.item()
        self.lnL = -loss
        return

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
        z0 = torch.linalg.pinv(one.T @ C_inv @ one) @ (
            one.T @ C_inv @ x
        )  # .item()

        # Next, get sigma2
        x0 = x - z0 * one  # (N, 1)
        sigma2 = x0.T @ C_inv @ x0  # (1, N) @ (N, N) @ (N, 1) = (1, 1)
        if unbiased:
            sigma2 = sigma2 / (N - 1)
        else:
            sigma2 = sigma2 / N
        # sigma2 = sigma2.item()

        # Print
        # print(f"z0 = {z0:.4f}, sigma2 = {sigma2:.4f}")

        # Finally, get log-likelihood
        ll_num = -0.5 * x0.T @ torch.linalg.pinv(sigma2 * C_lam) @ x0
        ll_denom = 0.5 * (
            N * torch.log(2 * torch.tensor([3.141592653589793]))
            + torch.linalg.slogdet(sigma2 * C_lam)[1]
        )
        ll = ll_num - ll_denom  # .item()

        # Print
        # print(f"lnL = {ll:.4f}")

        return z0, sigma2, ll

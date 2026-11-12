"""Common exact-MLL interface with an injectable GP calculation."""

from collections.abc import Callable

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

GPMLLFunction = Callable[..., torch.Tensor]


class BaseMLL(ExactMarginalLogLikelihood):
    """gp_mll(mll, output, target, *params, **kwargs) returns the GP MLL.

    The injected function owns the likelihood, priors and division by n.
    Output-warp Jacobians belong to the outer YJ wrapper, not this function.
    The default is GPyTorch's unbound ExactMarginalLogLikelihood.forward.
    """

    def __init__(
        self,
        likelihood,
        model,
        gp_mll: GPMLLFunction = ExactMarginalLogLikelihood.forward,
    ) -> None:
        super().__init__(likelihood, model)
        self._gp_mll_impl = gp_mll

    def _gp_mll(
        self, output: MultivariateNormal, target: torch.Tensor, *params, **kwargs
    ) -> torch.Tensor:
        """Evaluate the selected GP calculation with this MLL as its receiver."""
        return self._gp_mll_impl(self, output, target, *params, **kwargs)

    def forward(
        self, output: MultivariateNormal, target: torch.Tensor, *params, **kwargs
    ) -> torch.Tensor:
        return self._gp_mll(output, target, *params, **kwargs)


class ExactMLL(BaseMLL):
    """Exact GP MLL for native targets, including BoTorch standardization."""

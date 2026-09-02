import math
import warnings

import numpy as np
import torch
from botorch.acquisition import (
    AcquisitionFunction,
    qLogNoisyExpectedImprovement,
    qLogExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor

from .common import GP_Tensor


class HedgedPareto(AcquisitionFunction):
    """
    Calculates acquisition triplet as in HEBO (https://doi.org/10.1613/jair.1.13643)
    """

    def __init__(
        self,
        model: SingleTaskGP,
        best_f: float,
        beta: float,
        train_x: GP_Tensor,
        use_logei: bool = False,
        objective=None,
        outcome_warp=None,
        consider_fast_acq_path=True,
        quadrature_size: int = 64,
        mc_samples: int = 4096,
    ):
        super().__init__(model)
        self.model = model
        self.best_f = self._prepare_best_f_tensor(best_f)
        self.outcome_warp = outcome_warp
        self.use_logei = use_logei
        self.beta_prime = math.sqrt(beta * math.pi / 2)
        self.quadrature_size = quadrature_size
        self.mc_samples = mc_samples
        self.n_obj = 3

        # Determine if the fast quadrature path can be used
        self._use_fast_path = self._check_fast_path() if consider_fast_acq_path else False

        # Initialize quadrature objects if the fast path is available
        if self._use_fast_path:
            self._init_fast_path()

        # Always initialize MC-based acquisition objects (used as fallback or for q>1)
        self._init_mc_objects(train_x, objective, beta)

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------

    def _prepare_best_f_tensor(self, best_f: float) -> Tensor:
        """Convert best_f to a tensor with proper dtype and device."""
        return torch.as_tensor(
            best_f,
            dtype=self.model.train_inputs[0].dtype,
            device=self.model.train_inputs[0].device,
        )


    def _check_fast_path(self) -> bool:
        """
        Check whether the quadrature‑based fast path can be used.
        It is valid only when the Yeo‑Johnson lambda is within [0, 2].
        """
        if self.outcome_warp is None:
            return False
        lam = float(self.outcome_warp.lam.detach())
        if 0.0 <= lam <= 2.0:
            return True
        warnings.warn(
            "Yeo-Johnson lambda is outside [0, 2]; falling back to "
            "the Monte Carlo acquisition implementation because the "
            "inverse transform is not defined on the full Gaussian "
            "posterior support.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False


    def _init_fast_path(self):
        """Load nodes and weights for Gauss‑Legendre and Gauss‑Hermite quadratures."""
        # Legendre quadrature (used for EI)
        leg_nodes, leg_weights = np.polynomial.legendre.leggauss(self.quadrature_size)
        self.register_buffer("legendre_nodes", self.best_f.new_tensor(leg_nodes))
        self.register_buffer("legendre_weights", self.best_f.new_tensor(leg_weights))

        # Hermite quadrature (used for UCB)
        herm_nodes, herm_weights = np.polynomial.hermite.hermgauss(self.quadrature_size)
        self.register_buffer("hermite_nodes", self.best_f.new_tensor(herm_nodes))
        self.register_buffer("hermite_weights", self.best_f.new_tensor(herm_weights))


    def _init_mc_objects(self, train_x: GP_Tensor, objective, beta: float):
        """Create Monte‑Carlo‑based acquisition function objects (always needed)."""
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))

        # qLogEI or qLogNEI
        if not self.use_logei:
            self.qLogNEI = qLogNoisyExpectedImprovement(self.model, X_baseline=train_x, sampler=sampler, objective=objective,)
        else:
            self.qLogNEI = qLogExpectedImprovement(self.model, best_f=self.best_f, sampler=sampler, objective=objective,)
        # qPI
        self.qPI = qProbabilityOfImprovement(self.model, best_f=self.best_f, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples])), objective=objective,)
        # qUCB
        self.q_UCB = qUpperConfidenceBound(self.model, beta=beta, sampler=SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples])), objective=objective,)

    # -------------------------------------------------------------------------
    # Posterior helpers
    # -------------------------------------------------------------------------

    def _posterior_parameters(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Return the mean and standard deviation of the posterior at points X."""
        posterior = self.model.posterior(X)
        mean = posterior.mean[..., 0, 0]
        variance = posterior.variance[..., 0, 0]
        tiny = torch.finfo(variance.dtype).tiny
        return mean, variance.clamp_min(tiny).sqrt()


    def _inverse_warp(self, z: Tensor) -> Tensor:
        """Apply the inverse Yeo‑Johnson transformation."""
        return self.outcome_warp.untransform(z)

    # -------------------------------------------------------------------------
    # Fast quadrature‑based acquisition implementations
    # -------------------------------------------------------------------------

    def _fast_pi(self, mean: Tensor, sigma: Tensor) -> Tensor:
        """Probability of Improvement (PI) via quadrature."""
        best_z = self.outcome_warp(self.best_f)
        standardized = (mean - best_z) / sigma
        return torch.special.ndtr(standardized)


    def _fast_ei(self, mean: Tensor, sigma: Tensor) -> Tensor:
        """Expected Improvement (EI) via Gauss‑Legendre quadrature."""
        best_z = self.outcome_warp(self.best_f)
        lower_prob = torch.special.ndtr((best_z - mean) / sigma)
        tail_prob = 1.0 - lower_prob

        nodes = self.legendre_nodes
        probs = lower_prob.unsqueeze(-1) + 0.5 * tail_prob.unsqueeze(-1) * (nodes + 1.0)
        eps = torch.finfo(probs.dtype).eps
        probs = probs.clamp(min=eps, max=1.0 - eps)

        latent = mean.unsqueeze(-1) + sigma.unsqueeze(-1) * torch.special.ndtri(probs)
        values = self._inverse_warp(latent)
        improvement = (values - self.best_f).clamp_min(0.0)

        return 0.5 * tail_prob * (improvement * self.legendre_weights).sum(dim=-1)


    def _fast_ucb(self, mean: Tensor, sigma: Tensor) -> Tensor:
        """Upper Confidence Bound (UCB) via Gauss‑Hermite quadrature."""
        nodes = self.hermite_nodes
        latent = mean.unsqueeze(-1) + math.sqrt(2.0) * sigma.unsqueeze(-1) * nodes
        values = self._inverse_warp(latent)

        norm_weights = self.hermite_weights / math.sqrt(math.pi)
        outcome_mean = (values * norm_weights).sum(dim=-1)
        mean_abs_dev = ((values - outcome_mean.unsqueeze(-1)).abs() * norm_weights).sum(dim=-1)

        return outcome_mean + self.beta_prime * mean_abs_dev

    # -------------------------------------------------------------------------
    # Main forward pass
    # -------------------------------------------------------------------------

    def forward(self, X: Tensor) -> Tensor:
        """
        Compute the triplet (EI, PI, UCB) for candidate points X.
        Uses the fast quadrature path when possible, otherwise falls back to MC.
        """
        # If the fast path is unavailable or X has more than one point (q>1),
        # use the MC‑based acquisition objects.
        if (not self._use_fast_path) or X.shape[-2] != 1:
            val_ei = torch.exp(self.qLogNEI(X))
            val_pi = self.qPI(X)
            val_ucb = self.q_UCB(X)
            return torch.column_stack((val_ei, val_pi, val_ucb))

        # Fast path: one posterior evaluation supplies all three objectives.
        mean, sigma = self._posterior_parameters(X)

        # EI: use the fast method if use_logei is True, otherwise keep the MC version.
        if self.use_logei:
            val_ei = self._fast_ei(mean, sigma)
        else:
            val_ei = torch.exp(self.qLogNEI(X))

        val_pi = self._fast_pi(mean, sigma)
        val_ucb = self._fast_ucb(mean, sigma)

        return torch.column_stack((val_ei, val_pi, val_ucb))

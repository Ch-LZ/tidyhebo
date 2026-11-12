import math
from abc import abstractmethod

import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qLogExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from scipy.special import roots_hermite
from torch import Tensor

from .base_acquisition import BaseAcquisition
from .config import AcquisitionConfig


class BaseMACE(BaseAcquisition):
    """Shared state and input contract for the (LogEI, PI, UCB) triplet."""

    n_obj = 3

    def __init__(self, model, config: AcquisitionConfig, best_f, train_x=None, outcome_warp=None):
        super().__init__(model, config)
        ref = model.train_inputs[0]
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=ref.dtype, device=ref.device))
        self.outcome_warp = outcome_warp

    def forward(self, X: Tensor) -> Tensor:
        if X.shape[-2] != self.q:
            raise ValueError(f"Expected q={self.q}, got X.shape={tuple(X.shape)}")
        return self._scores(X)

    @abstractmethod
    def _scores(self, X: Tensor) -> Tensor: ...


class AnalyticMACE(BaseMACE):
    """Exact Gaussian q=1 LogEI/PI/UCB on the acquisition outcome scale.

    A nonlinear active outcome warp makes that scale non-Gaussian; use
    QuadratureMACE for such points. Joint q>1 requires QMC MACE.
    """

    def __init__(self, model, config, best_f, train_x=None, outcome_warp=None):
        super().__init__(model, config, best_f, train_x, outcome_warp)
        if config.q != 1 or outcome_warp is not None:
            raise ValueError("AnalyticMACE requires q == 1 and Gaussian outcomes")
        if config.evaluation != "analytic":
            raise ValueError("AnalyticMACE requires analytic evaluation")
        self.logei = LogExpectedImprovement(model, self.best_f)
        self.pi = ProbabilityOfImprovement(model, self.best_f)
        self.ucb = UpperConfidenceBound(model, config.beta)

    def _scores(self, X):
        return torch.stack((self.logei(X), self.pi(X), self.ucb(X)), dim=-1)


class MACE(BaseMACE, _target_="MACE"):
    """Native QMC triplet for both single points and joint q-batches."""

    def __init__(self, model, config: AcquisitionConfig, best_f, train_x=None, outcome_warp=None):
        super().__init__(model, config, best_f, train_x, outcome_warp)
        if config.evaluation != "qmc":
            raise ValueError("QMC MACE requires qmc evaluation; use QuadratureMACE for quadrature")
        objective = self._original_scale_objective(outcome_warp)

        def sampler():
            return SobolQMCNormalSampler(torch.Size([config.mc_samples]), seed=config.qmc_seed)

        self.qLogEI = qLogExpectedImprovement(model, self.best_f, sampler=sampler(), objective=objective)
        self.qPI = qProbabilityOfImprovement(
            model, sampler=sampler(), best_f=self.best_f, objective=objective
        )
        self.qUCB = qUpperConfidenceBound(model, config.beta, sampler=sampler(), objective=objective)

    def _scores(self, X: Tensor) -> Tensor:
        return torch.stack((self.qLogEI(X), self.qPI(X), self.qUCB(X)), dim=-1)


class SharedMACE(MACE):
    """Reuse samples only when the three native samplers have the SAME seed.

    For seed=None the original MACE constructor chooses three different seeds;
    preserve that behavior rather than changing the estimator. BoTorch 0.14's
    native reductions retain its stable qLogEI smoothing and log-mean-exp.
    """

    def _scores(self, X):
        functions = (self.qLogEI, self.qPI, self.qUCB)
        if len({f.sampler.seed for f in functions}) != 1:
            return super()._scores(X)
        _, objectives = self.qLogEI._get_samples_and_objectives(X)
        return torch.stack(
            [f._sample_reduction(f._q_reduction(f._sample_forward(objectives))) for f in functions], dim=-1
        )


class QuadratureMACE(BaseMACE):
    """Single-point triplet from one posterior and deterministic quadratures."""

    def __init__(self, model, config: AcquisitionConfig, best_f, train_x=None, outcome_warp=None):
        super().__init__(model, config, best_f, train_x, outcome_warp)
        if config.q != 1:
            raise ValueError("QuadratureMACE requires q == 1")
        if outcome_warp is None:
            raise ValueError("Quadrature requires active Yeo-Johnson warping after the gate")
        if config.evaluation != "quadrature":
            raise ValueError("QuadratureMACE requires quadrature evaluation")
        self.beta_prime = math.sqrt(config.beta * math.pi / 2)
        leg_nodes, leg_weights = np.polynomial.legendre.leggauss(config.quadrature_size)
        self.register_buffer("legendre_nodes", self.best_f.new_tensor(leg_nodes))
        self.register_buffer("legendre_weights", self.best_f.new_tensor(leg_weights))
        # SciPy's Hermite rule remains stable at the supported large orders.
        herm_nodes, herm_weights = roots_hermite(config.quadrature_size)
        self.register_buffer("hermite_nodes", self.best_f.new_tensor(herm_nodes))
        self.register_buffer("hermite_weights", self.best_f.new_tensor(herm_weights))

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

    def _pi(self, mean: Tensor, sigma: Tensor) -> Tensor:
        """Analytic PI in latent Gaussian space under the monotone outcome warp."""
        best_z = self.outcome_warp(self.best_f)
        standardized = (mean - best_z) / sigma
        return torch.special.ndtr(standardized)

    @staticmethod
    def _normal_quantile_from_log_prob(log_prob: Tensor) -> Tensor:
        """Inverse normal CDF without underflow in the far left tail."""
        tail = log_prob < -20.0
        # Clamp the unused branch, so ndtri never sees an underflowed zero.
        result = torch.special.ndtri(log_prob.clamp_min(-20.0).exp())
        tail_log_prob = log_prob[tail]
        z = -torch.sqrt(-2.0 * tail_log_prob)
        # Newton solve log(Phi(z)) = log_prob. The asymptotic initial guess
        # converges rapidly; all operations retain input gradients.
        for _ in range(6):
            log_cdf = torch.special.log_ndtr(z)
            slope = (-0.5 * z.square() - 0.5 * math.log(2 * math.pi) - log_cdf).exp()
            z = z - (log_cdf - tail_log_prob) / slope
        return result.masked_scatter(tail, z)

    def _logei(self, mean: Tensor, sigma: Tensor) -> Tensor:
        """Log EI via Gauss-Legendre quadrature in survival-probability space."""
        best_z = self.outcome_warp(self.best_f)
        log_tail = torch.special.log_ndtr((mean - best_z) / sigma)
        u = 0.5 * (self.legendre_nodes + 1.0)
        log_survival = log_tail.unsqueeze(-1) + u.log()
        latent = mean.unsqueeze(-1) - sigma.unsqueeze(-1) * self._normal_quantile_from_log_prob(log_survival)
        values = self._inverse_warp(latent)
        # Positive weights and improvements are summed directly in log space.
        # A floor only guards roundoff in the inverse-warp subtraction.
        improvement = (values - self.best_f).clamp_min(torch.finfo(values.dtype).tiny)
        return log_tail + torch.logsumexp(
            improvement.log() + (0.5 * self.legendre_weights).log(),
            dim=-1,
        )

    def _ucb(self, mean: Tensor, sigma: Tensor) -> Tensor:
        """Upper Confidence Bound (UCB) via Gauss‑Hermite quadrature."""
        nodes = self.hermite_nodes
        latent = mean.unsqueeze(-1) + math.sqrt(2.0) * sigma.unsqueeze(-1) * nodes
        values = self._inverse_warp(latent)

        norm_weights = self.hermite_weights / math.sqrt(math.pi)
        outcome_mean = (values * norm_weights).sum(dim=-1)
        mean_abs_dev = ((values - outcome_mean.unsqueeze(-1)).abs() * norm_weights).sum(dim=-1)

        return outcome_mean + self.beta_prime * mean_abs_dev

    def _scores(self, X: Tensor) -> Tensor:
        mean, sigma = self._posterior_parameters(X)
        return torch.stack((self._logei(mean, sigma), self._pi(mean, sigma), self._ucb(mean, sigma)), dim=-1)


class BoTorchScalarAcquisition(BaseAcquisition):
    """Native analytic Gaussian acquisition, with an explicit objective axis."""

    n_obj = 1
    native_type = ExpectedImprovement

    def __init__(self, model, config: AcquisitionConfig, best_f, train_x, outcome_warp=None):
        super().__init__(model, config)
        if outcome_warp is not None:
            raise ValueError(
                "Analytic EI/LogEI requires Gaussian outcomes; use output_transform='standardize'"
            )
        if config.evaluation != "analytic":
            raise NotImplementedError("Scalar EI/LogEI supports analytic EI only")
        self.native_acqf = self.native_type(model=model, best_f=best_f)

    def forward(self, X: Tensor) -> Tensor:
        return self.native_acqf(X).unsqueeze(-1)


class EI(BoTorchScalarAcquisition, _target_="EI"):
    """Native analytic expected improvement."""


class LogEI(BoTorchScalarAcquisition, _target_="LogEI"):
    """Native analytic log expected improvement."""

    native_type = LogExpectedImprovement


class qLogEI(BaseAcquisition, _target_="qLogEI"):
    """Monte Carlo log-EI in original outcome units, including a retained YJ warp."""

    n_obj = 1

    def __init__(self, model, config: AcquisitionConfig, best_f, train_x=None, outcome_warp=None):
        super().__init__(model, config)
        if config.evaluation != "qmc":
            raise ValueError("qLogEI requires qmc evaluation")
        self.native_acqf = qLogExpectedImprovement(
            model=model, best_f=best_f,
            sampler=SobolQMCNormalSampler(torch.Size([config.mc_samples]), seed=config.qmc_seed),
            objective=self._original_scale_objective(outcome_warp),
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.native_acqf(X).unsqueeze(-1)

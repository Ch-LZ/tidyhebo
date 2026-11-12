import logging
import torch
from abc import ABC, abstractmethod

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import Warp
from botorch.models.utils import gpytorch_modules
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior

from .common import GP_Tensor
from .config import ModelConfig
from .mll import BaseMLL, ExactMLL, GPMLLFunction
from .registry import Registrable
from .yeo_johnson_warp import YeoJohnsonGPytorch, YJWarpedExactMLL


logger = logging.getLogger(__name__)


class SurrogateModel(Registrable, ABC):
    """
    Interface and registry for surrogate-model implementations.
    """

    @abstractmethod
    def build_gp_mll(self) -> tuple[SingleTaskGP, BaseMLL]:
        """Build the compatible GP/MLL pair for the current observations."""
        ...

    @abstractmethod
    def train_model(self, state_dict=None) -> dict[str, torch.Tensor] | None:
        """Fit and return an independent parameter dictionary, or None for a cold fit."""
        ...


class SimpleModel(SurrogateModel, _target_="joint"):
    """Fit a single-task GP using ModelConfig settings.

    Configuration is retained and read when training. The active outcome-warp
    flag is separate runtime state: the gate may disable warping without
    changing the configuration. Inputs must be in normalized GP space.
    """

    def __init__(self,
                 train_x: GP_Tensor,
                 train_y,
                 config: ModelConfig,
                 ) -> None:
        self.model: SingleTaskGP = None
        self.train_y = train_y
        self.train_x: GP_Tensor = train_x
        self.mll = None
        self.config = config
        self.use_power_outcome_transform = self.config.output_transform == "yeo_johnson"

        assert self.config.kernel in ("RBF", "Matern52_Gamma", "Matern52", "default")

        n_x = self.train_x.shape[1]
        self.bounds = torch.repeat_interleave(
            torch.tensor([[0,1],], device=self.train_x.device, dtype=torch.double), n_x, dim=0).T


    @staticmethod
    def _make_me_a_kernel(kind, train_x, train_y) -> Kernel | None:
        """
        Создание кернела как в BoTorch (https://github.com/pytorch/botorch/blob/main/botorch/models/utils/gpytorch_modules.py)
        но более удобным для использования образом.
        Заметка: в BoTorch именно удобное использование этого не имплементировано.
        Ссылка:
        [Hvarfner2024vanilla]
            C. Hvarfner, E. O. Hellsten, L. Nardi,
            Vanilla Bayesian Optimization Performs Great in High Dimensions.
            In International Conference on Machine Learning, 2024.

        В работе показывают, что обычный сетап байесовской оптимизации (GP). При правильном прайоре работает не хуже, чем более сложные многосоставные модели
        """
        arg_num = train_x.shape[-1]
        _, _aug_batch_shape = BatchedMultiOutputGPyTorchModel.get_batch_dimensions(train_x, train_y)
        match kind:
            case "Matern52_Gamma":
                return gpytorch_modules.get_matern_kernel_with_gamma_prior(ard_num_dims=arg_num, batch_shape=_aug_batch_shape)
            case "Matern52":
                return gpytorch_modules.get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=arg_num,
                    batch_shape=_aug_batch_shape,
                    use_rbf_kernel=False
                )
            case "RBF":
                return gpytorch_modules.get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=arg_num,
                    batch_shape=_aug_batch_shape,
                    use_rbf_kernel=True
                )
            case "default":
                return None


    def train_model(self, state_dict=None) -> dict[str, torch.Tensor] | None:
        if self.config.warp_fit != "joint":
            raise NotImplementedError("Separate output-warp fitting is not implemented yet")
        self.use_power_outcome_transform = self.config.output_transform == "yeo_johnson"
        self._train_model_inner(state_dict if self.config.warm_start else None)
        if (self.config.gate and self.use_power_outcome_transform
                and self.model.train_targets.std() < 0.5):
            self.use_power_outcome_transform = False
            self._train_model_inner()
        if not self.config.warm_start:
            return None
        return {name: value.detach().clone()
                for name, value in self.model.named_parameters()}


    def build_gp_mll(self) -> tuple[SingleTaskGP, BaseMLL]:
        # building a model
        ## input transform
        input_transform = (self._get_kumaraswamy_warp(self.train_x, self.bounds)
                            if self.config.input_warp else None)

        ## make a GP
        kernel = self._make_me_a_kernel(kind=self.config.kernel, train_x=self.train_x, train_y=self.train_y)
        self.model = SingleTaskGP(
            train_X=self.train_x,
            train_Y=self.train_y,
            covar_module=kernel,
            input_transform=input_transform,
            # outcome transform - default (scale) and not learnable
        ).to(self.train_x)

        self.mll = self._create_mll().to(self.train_x)
        return self.model, self.mll

    def _train_model_inner(self, state_dict=None) -> None:
        self.model, self.mll = self.build_gp_mll()
        # load state dict if any for faster convergence
        if state_dict:
            self._load_model_state(state_dict)

        self._fit_mll()
        if self.use_power_outcome_transform:
            # During optimization YJWarpedExactMLL evaluates the current
            # transformed targets directly against the GP prior.
            # But the ExactGP model by default stores `train_targets` from previous step.
            # Here we use final fitted lambda, so ExactGP will calculate
            # posterior conditioned on correct (correctly transformed) targets.
            with torch.no_grad():
                train_targets = self.model.outcome_warp_yj(self.train_y).squeeze(-1)
            self.model.set_train_data(
                inputs=self.model.train_inputs,
                targets=train_targets.detach(),
                strict=False,
            )
        self.model.prediction_strategy = None
        self.model.eval()
        self.mll.eval()


    def _fit_mll(self) -> None:
        fit_gpytorch_mll(self.mll,
                        max_attempts=self.config.max_attempts,
                        pick_best_of_all_attempts=self.config.pick_best)


    def _create_mll(
        self, gp_mll: GPMLLFunction = ExactMarginalLogLikelihood.forward,
    ) -> BaseMLL:
        """Create the training objective for the active outcome transformation."""
        if self.use_power_outcome_transform:
            return YJWarpedExactMLL(
                self.model.likelihood, self.model,
                warping=YeoJohnsonGPytorch(), train_Y_raw=self.train_y,
                gp_mll=gp_mll,
            )
        return ExactMLL(self.model.likelihood, self.model, gp_mll=gp_mll)


    def _load_model_state(self, state_dict) -> None:
        """Restore parameters only; retain current data-dependent buffers."""
        parameters = dict(self.model.named_parameters())
        if any(name not in state_dict or state_dict[name].shape != value.shape
               for name, value in parameters.items()):
            logger.warning("Incompatible warm-start parameters; using a cold fit")
            return
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(state_dict[name].to(value))


    def _get_kumaraswamy_warp(self, train_x, bounds) -> Warp:
        """
        Kumaraswamy преобразованеи для учёта гетероскедастичности
        """
        return Warp(  # уже содержит внутри kumaraswamy преобразование (и нормализацию)
                d=train_x.shape[-1],
                indices=list(range(train_x.shape[-1])),
                # use a prior with median at 1.
                # when a=1 and b=1, the Kumaraswamy CDF is the identity function
                concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
                concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
                bounds=bounds,
            )


# region Centered

def _centered_gp_mll(self, output, target, *params, **kwargs) -> torch.Tensor:
    """Evaluate m = beta + mean(target) during fitting, physical m otherwise."""
    if self.optimizing_beta:
        # Adding mean(target) to beta is equivalent to centering the targets.
        output = MultivariateNormal(
            output.mean + target.mean(-1, keepdim=True), output.lazy_covariance_matrix)
    return ExactMarginalLogLikelihood.forward(self, output, target, *params, **kwargs)


class CenteredModel(SimpleModel, _target_="centered"):
    """Fit m = beta + mean(current GP targets); expose physical m after fitting.

    Background: Rasmussen & Williams (2006), GPML, sec. 2.7, eq. (2.38),
    https://gaussianprocess.org/gpml/chapters/RW2.pdf
    The lambda-dependent empirical offset is our reparameterization of that
    mean-shift identity; it remains differentiable during joint warp fitting.
    """

    def _create_mll(self) -> BaseMLL:
        mll = super()._create_mll(gp_mll=_centered_gp_mll)
        # Parameters represent physical m outside _fit_mll, beta inside it.
        mll.optimizing_beta = False
        return mll

    def _target_mean(self) -> torch.Tensor:
        targets = (self.model.outcome_warp_yj(self.train_y)
                   if self.use_power_outcome_transform else self.model.train_targets)
        return targets.mean()

    def _fit_mll(self) -> None:
        with torch.no_grad():
            mean = self.model.mean_module.constant
            if self.config.centered_init == "zero_beta":
                mean.zero_()
            else:
                mean.sub_(self._target_mean())
        self.mll.optimizing_beta = True
        try:
            super()._fit_mll()
        finally:
            # Lambda may have changed; restore m with the final transformed mean.
            with torch.no_grad():
                self.model.mean_module.constant.add_(self._target_mean())
            self.mll.optimizing_beta = False


# endregion


# region Profiled

def _profiled_gp_mll(self, output, target, *params, **kwargs) -> torch.Tensor:
    """Evaluate GP log likelihood with the constant mean maximized analytically.

    Let z = target be the CURRENT targets in GP space (YJ-transformed or
    standardized), 1 the all-ones vector, and n the number of observations.
    The likelihood adds observation noise before we factor the covariance:

        C = K_theta(X, X) + Sigma_noise = L L^T
        log p(z | m, C) = -1/2 * [(z - m*1)^T C^-1 (z - m*1)
                                 + log|C| + n*log(2*pi)]

    Setting its derivative with respect to m to zero gives the GLS mean:

        m_hat = (1^T C^-1 z) / (1^T C^-1 1)

    Compute it without forming C^-1: one Cholesky and a triangular solve
    with two right-hand sides give u = L^-1 z and v = L^-1 1. Then:

        m_hat = (v^T u) / (v^T v)
        residual = u - m_hat*v
        (z - m_hat*1)^T C^-1 (z - m_hat*1) = ||residual||^2
        log|C| = 2 * sum_i log(L_ii)
        value = -1/2 * [||residual||^2 + 2*sum_i log(L_ii) + n*log(2*pi)]

    Return (value + GPyTorch prior/added-loss terms) / n. For YJ, the outer
    YJWarpedExactMLL adds its Jacobian term separately. This assumes no prior
    on m: it is profile likelihood, not integration over m or a REML objective.

    Reference: Lophaven, Nielsen & Sondergaard (2002), DACE: A MATLAB
    Kriging Toolbox, IMM-TR-2002-12, sec. 3, eqs. (3.7)-(3.10): GLS and
    Cholesky whitening. https://www.omicron.dk/dace/dace.pdf
    This is the constant-basis F = 1 case, using the full noisy covariance C;
    only the mean is profiled here, not the covariance scale.

    m_hat remains in the autograd graph of residual, so gradients reach both
    covariance parameters and the transformed targets. The no_grad copy only
    exposes physical m to the native GP/posterior and warm-state dictionary.
    """
    observed = self.likelihood(output, *params, **kwargs)
    chol = observed.lazy_covariance_matrix.cholesky().to_dense()
    rhs = torch.stack((target, torch.ones_like(target)), dim=-1)
    u, v = torch.linalg.solve_triangular(chol, rhs, upper=False).unbind(-1)
    mean = (v * u).sum(-1) / v.square().sum(-1)
    residual = u - mean.unsqueeze(-1) * v
    value = -0.5 * (residual.square().sum(-1)
                    + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                    + target.shape[-1] * torch.log(target.new_tensor(2 * torch.pi)))
    # The native GP and its parameter snapshot always expose physical m.
    with torch.no_grad():
        self.model.mean_module.constant.copy_(mean)
    return self._add_other_terms(value, params) / target.shape[-1]


class ProfiledModel(SimpleModel, _target_="profiled"):
    """Profile m = (1^T C^-1 z)/(1^T C^-1 1), without a mean prior or REML term.

    GLS reference: Lophaven, Nielsen & Sondergaard (2002), DACE: A MATLAB
    Kriging Toolbox, IMM-TR-2002-12, sec. 3, eqs. (3.7)-(3.10), with F = 1.
    https://www.omicron.dk/dace/dace.pdf
    """

    def _create_mll(self) -> BaseMLL:
        self.model.mean_module.raw_constant.requires_grad_(False)
        return super()._create_mll(gp_mll=_profiled_gp_mll)

    def _fit_mll(self) -> None:
        super()._fit_mll()
        # Retry/best-attempt selection may restore a different parameter snapshot.
        with torch.no_grad():
            self.model.train()
            inputs = self.model.train_inputs
            self.mll(self.model(*inputs), self.model.train_targets,
                     *(self.model.transform_inputs(X=x) for x in inputs))

# endregion

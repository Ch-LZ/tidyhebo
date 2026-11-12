# region yeo-johnson
import torch
from gpytorch.distributions import MultivariateNormal  # typing only


def _expm1_over_x(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Continuous expm1(x)/x, including its derivatives at zero."""
    # Balance roundoff O(eps/|x|) in the regular derivative against
    # the first omitted derivative term, x**9 / 3991680.
    small = x.abs() < max(epsilon, (3991680 * torch.finfo(x.dtype).eps) ** (1 / 10))
    # Keep *both* evaluated branches finite: torch.where alone does not stop
    # autograd from propagating NaNs from a zero denominator or unused overflow.
    t = torch.where(small, x, torch.zeros_like(x))
    series = 1 + t * (1 / 2 + t * (1 / 6 + t * (1 / 24 + t * (
        1 / 120 + t * (1 / 720 + t * (1 / 5040 + t * (
        1 / 40320 + t * (1 / 362880 + t / 3628800))))))))
    safe_x = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, series, torch.expm1(safe_x) / safe_x)


def _log1p_over_x(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Continuous log1p(x)/x on x > -1, with derivatives at zero."""
    # The first omitted derivative term is 8*x**7/9; balance its
    # magnitude against roundoff O(eps/|x|) in the regular derivative.
    small = x.abs() < max(epsilon, (9 * torch.finfo(x.dtype).eps / 8) ** (1 / 8))
    t = torch.where(small, x, torch.zeros_like(x))
    series = 1 + t * (-1 / 2 + t * (1 / 3 + t * (-1 / 4 + t * (
        1 / 5 + t * (-1 / 6 + t * (1 / 7 - t / 8))))))
    safe_x = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, series, torch.log1p(safe_x) / safe_x)


def yeo_johnson_transform(y: torch.Tensor, lam: float, epsilon: float = 1e-14) -> torch.Tensor:
    """Yeo–Johnson transform with differentiable limits at lambda=0 and 2.

    Lambda may be a scalar or a tensor expandable to y's shape. Degree-nine
    series are used for |lambda*log1p(y)| < (3991680*dtype.eps)**(1/10)
    on the positive branch; use (2-lambda)*log1p(-y) for y <= 0.
    The cutoff balances truncation and roundoff in the first derivative;
    epsilon sets a minimum cutoff.
    """
    if isinstance(lam, (int, float)):
        lam = torch.tensor(lam, dtype=y.dtype, device=y.device)
    if lam.shape != y.shape:
        lam = lam.expand_as(y)

    positive = y > 0
    # Unlike abs(), this preserves the input derivative at y == 0.
    magnitude = torch.log1p(torch.where(positive, y, -y))
    exponent = torch.where(positive, lam, 2 - lam)
    signed_log = torch.where(positive, magnitude, -magnitude)
    return signed_log * _expm1_over_x(exponent * magnitude, epsilon)


def inv_yeo_johnson_transform(y: torch.Tensor, lam: float, epsilon: float = 1e-14) -> torch.Tensor:
    """Inverse Yeo–Johnson, with differentiable limits at lambda=0 and 2.

    With autograd active, degree-seven series use (9*dtype.eps/8)**(1/8)
    as a cutoff on the dimensionless argument, with epsilon as a minimum.
    Value-only calls use log1p and its continuous limit directly. The original
    domain is unchanged: 1 + lambda*y > 0 on the positive branch and
    1 - (2-lambda)*y > 0 on the negative branch.
    """
    if isinstance(lam, (int, float)):
        lam = torch.tensor(lam, dtype=y.dtype, device=y.device)
    if lam.shape != y.shape:
        lam = lam.expand_as(y)

    positive = y > 0
    magnitude = torch.where(positive, y, -y)
    exponent = torch.where(positive, lam, 2 - lam)
    if torch.is_grad_enabled() and (y.requires_grad or lam.requires_grad):
        # The ratio needs a series for accurate derivatives by lambda.
        inner = magnitude * _log1p_over_x(exponent * magnitude, epsilon)
    else:
        # Values alone need no polynomial: log1p avoids cancellation.
        # Divide by its own argument so subnormal products remain accurate.
        argument = exponent * magnitude
        zero = argument == 0
        safe_argument = torch.where(zero, 1.0, argument)
        ratio = torch.where(zero, 1.0, torch.log1p(argument) / safe_argument)
        inner = magnitude * ratio
    result = torch.expm1(inner)
    return torch.where(positive, result, -result)
# endregion

# region Gpytroch module
import torch

from gpytorch import Module as GPytorchModule
from gpytorch.priors import NormalPrior
from gpytorch.constraints import Interval


class YeoJohnsonGPytorch(GPytorchModule):
    """
    A module for Yeo-Johnson learnable transformation.
    Lambda is constrained to [0, 2], making the inverse valid on all real inputs.
    The sigmoid maps finite raw parameters into the open interval;
    exact endpoints can be assigned explicitly. Tested for single outcome only.
    """
    def __init__(self) -> None:
        super().__init__()

        # _scale is set by YJWarpedExactMLL and does not require gradients.
        # A non-persistent buffer follows dtype / device conversions without
        # changing the model state_dict format. 
        # *Note: It will not appear in model.parameters()
        self.register_buffer(
            "_scale",
            torch.tensor([1.0], dtype=torch.double),
            persistent=False,
        )

        # raw_lam=0 maps to lambda=1: start from the identity transform.
        self.register_parameter("raw_lam", torch.nn.Parameter(torch.tensor(0.0, dtype=torch.double)))
        self.register_constraint("raw_lam", Interval(0.0, 2.0))
        self.register_prior("lam_prior", NormalPrior(loc=1.0, scale=0.5),
                            lambda m: m.lam,
                            lambda m, v: m._set_lam(v),)

    @property
    def lam(self) -> torch.Tensor:
        return self.raw_lam_constraint.transform(self.raw_lam)


    @lam.setter
    def lam(self, value) -> None:
        return self._set_lam(value)


    def _set_lam(self, value) -> None:
        # only because seem to unable to use setter in a constructor
        value = torch.as_tensor(value, dtype=self.raw_lam.dtype, device=self.raw_lam.device)
        if not self.raw_lam_constraint.check(value):
            raise ValueError("Yeo-Johnson lambda must be within [0, 2]")
        self.raw_lam.data.copy_(
            self.raw_lam_constraint.inverse_transform(value))


    def forward(self, y) -> torch.Tensor:
        y = y / self._scale
        rv = yeo_johnson_transform(y, lam=self.lam)
        return rv


    def untransform(self, y_hat) -> torch.Tensor:
        rv = inv_yeo_johnson_transform(y_hat, lam=self.lam)
        rv = rv * self._scale
        return rv


    def log_jacobian(self, y0) -> torch.Tensor:
        """
        Log jacobian: log( |dT/dy| ) at point y0
        """
        y0 = y0 / self._scale

        y_pos = y0.clamp_min(0.0)
        y_neg = (-y0).clamp_min(0.0)

        s_pos = torch.log1p(y_pos).sum()
        s_neg = torch.log1p(y_neg).sum()

        jacobian_term = (self.lam - 1) * (s_pos - s_neg)

        return jacobian_term

# endregion

# region likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from .mll import BaseMLL, GPMLLFunction


class YJWarpedExactMLL(BaseMLL):
    """
    The class to train SingleTaskGP and YeoJohnson transform together
    and perform changes requried for that.
    Tested for single outcome tasks only.
    """
    def __init__(self, likelihood, model: SingleTaskGP, 
                 warping: YeoJohnsonGPytorch, train_Y_raw: torch.Tensor,
                 gp_mll: GPMLLFunction = ExactMarginalLogLikelihood.forward) -> None:
        # Set warping parameter to the model
        # This is required to be within the model 
        # to restore previous state_dict in training time at fit_gpytorch_mll
        assert not hasattr(model, "outcome_warp_yj")
        model.outcome_warp_yj = warping

        # Delete BoTorch (native) outcome transformation from attributes
        # It seemed to interfere with trainable outcome warping 
        # with attempts to preserve it together with `warping` being unsuccessful
        delattr(model, "outcome_transform")

        # Register assembled model
        super().__init__(likelihood, model, gp_mll=gp_mll)
        self.model: SingleTaskGP

        self.train_Y_raw: torch.Tensor = train_Y_raw

        # Introduce scaling factor to the transform
        # (max - min) * std(y / (max - min)) is exactly std(y), but the
        # expanded expression divides by zero for constant observations.
        _scale = train_Y_raw.std()
        if (_scale <= 1e-8) or torch.isnan(_scale).any():
            _scale = train_Y_raw.new_tensor(1.0)
        _scale = _scale.detach()
        warping._scale = _scale  # should not require grad


    def forward(self, output: MultivariateNormal, _target,  *params, **kwargs) -> torch.Tensor:
        """
        Is used at botorch.optim.closures.model_closures -> @GetLossClosure.register(ExactMarginalLogLikelihood, object, object, NoneType)
        Params:
            output - GP prior at the training inputs, calculated by the
                BoTorch loss closure. In ExactGP training mode this does not
                depend on train_targets or on lambda.
                p(f| X,θ) = N(m(X),Kff(X,X;θ))
            _target - model.train_targets just passed through. Can't be used since they are already warped with previous lambda
            params - transformed input for consistency. Just to pass into forward as in ExactMarginalLogLikelihood
        """
        y_raw = self.train_Y_raw
        trainable_warp: YeoJohnsonGPytorch = self.model.outcome_warp_yj

        y_transformed = trainable_warp.forward(y_raw)
        y_transformed = y_transformed.reshape(self.train_Y_raw.shape)
        y_transformed = y_transformed.squeeze(-1)

        gp_mll_term = self._gp_mll(
            output,
            y_transformed,
            *params,
            **kwargs,
        )

        log_jacobian = trainable_warp.log_jacobian(y_raw)
        #note: prior term handled automatically by GPytorch Module, so only jacobian is required

        # Scale by the amount of data
        # as it is done in ExactMarginalLogLikelihood for consistency of loss scales
        num_data = output.event_shape.numel()
        yj_term = log_jacobian.div_(num_data)

        return gp_mll_term + yj_term

# endregion

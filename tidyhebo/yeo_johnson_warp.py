# region yeo-johnson
import torch


def yeo_johnson_transform(y: torch.Tensor, lam: float, epsilon:float = 1e-14):
    """
    YJ direct transformation. So far tested only for lam being a single value (not a vector)
    """
    # y allower to be any real number

    # prepare lam
    if isinstance(lam, (int, float)):
        lam = torch.tensor(lam, dtype=y.dtype, device=y.device)
    # Expand lam to match y's shape if needed
    if lam.shape != y.shape:
        lam = lam.expand_as(y)

    # print(f"y shape: {y.shape}, lam shape: {lam.shape}")

    y_hat = torch.zeros_like(y)
    y_gt_0 = (y > 0)
    lam_near_0 = torch.abs(lam) < epsilon
    lam_near_2 = torch.abs(lam - 2) < epsilon 
    # For y >= 0:
    # case 1: y >= 0, lam != 0
    mask1 = y_gt_0 & ~lam_near_0
    if mask1.any():
        # y_hat = ((y + 1) ** lam - 1) / lam
        inner = torch.log1p(y[mask1]) * lam[mask1]
        y_hat[mask1] = torch.expm1(inner) / lam[mask1]

    # case 2: y >= 0, lam == 0
    mask2 = y_gt_0 & lam_near_0
    if mask2.any():
        y_hat[mask2] = torch.log1p(y[mask2])

    # For y < 0
    # case3: y < 0, lam != 2
    mask3 = (~y_gt_0) & ~lam_near_2
    if mask3.any():
        # y_hat = -((-y + 1)**(2-lam)-1) / (2-lam)
        inner = torch.log1p(-y[mask3]) * (2 - lam[mask3])
        y_hat[mask3] = -1 * torch.expm1(inner) / (2 - lam[mask3])
    
    # case 4: y < 0, lam == 2
    mask4 = (~y_gt_0) & lam_near_2
    if mask4.any():
        y_hat[mask4] = -1 * torch.log1p(-y[mask4])
    
    return y_hat


def inv_yeo_johnson_transform(y: torch.Tensor, lam: float, epsilon: float = 1e-14):
    """
    YJ inverse transformation. So far tested only for lam being a single value (not a vector)
    """
    # prepare lam
    if isinstance(lam, (int, float)):
        lam = torch.tensor(lam, dtype=y.dtype, device=y.device)
    # Expand lam to match y's shape if needed
    if lam.shape != y.shape:
        lam = lam.expand_as(y)

    # print(f"y shape: {y.shape}, lam shape: {lam.shape}")

    y_hat = torch.zeros_like(y)

    y_gt_0 = (y > 0)
    lam_near_0 = torch.abs(lam) < epsilon
    lam_near_2 = torch.abs(lam - 2) < epsilon
    # For y >= 0:
    # case 1: y >= 0, lam != 0
    mask1 = y_gt_0 & ~lam_near_0
    if mask1.any():
        # y_hat = (y * lam) ** (1/lam) - 1
        inner = y[mask1] * lam[mask1]
        y_hat[mask1] = torch.expm1(torch.log1p(inner) / lam[mask1])

    # case 2: y >= 0, lam == 0
    mask2 = y_gt_0 & lam_near_0
    if mask2.any():
        y_hat[mask2] = torch.expm1(y[mask2])

    # For y < 0
    # case3: y < 0, lam != 2
    mask3 = (~y_gt_0) & ~lam_near_2
    if mask3.any():
        # y_hat = -((-y + 1)**(2-lam)-1) / (2-lam)
        inner = -(2 - lam[mask3]) * y[mask3]
        log_term = torch.log1p(inner) / (2 - lam[mask3])
        y_hat[mask3] = -1 * torch.expm1(log_term)

    # case 4: y < 0, lam == 2
    mask4 = (~y_gt_0) & lam_near_2
    if mask4.any():
        y_hat[mask4] = -1 * torch.expm1(-y[mask4])

    return y_hat
# endregion

# region Gpytroch module
import torch

from gpytorch import Module as GPytorchModule
from gpytorch.priors import NormalPrior
from gpytorch.constraints import Interval


class YeoJohnsonGPytorch(GPytorchModule):
    """
    A module for Yeo-Johnson learnable transformation.
    Tested for single outcome only.
    """
    def __init__(self):
        super().__init__()

        #*note: _scale is modified externally at YJWarpedExactMLL at training time; should not require gard
        self._scale = torch.tensor([1.0], dtype=torch.double)

        self.register_parameter("raw_lam", torch.nn.Parameter(torch.tensor(1.0 + 1e-6, dtype=torch.double)))
        self.register_constraint("raw_lam", Interval(-3.0, 3.0))
        self.register_prior("lam_prior", NormalPrior(loc=1.0 + 1e-6, scale=0.5),
                            lambda m: m.lam,
                            lambda m, v: m._set_lam(v),)

    @property
    def lam(self):
        return self.raw_lam_constraint.transform(self.raw_lam)


    @lam.setter
    def lam(self, value):
        return self._set_lam(value)


    def _set_lam(self, value):
        # only because seem to unable to use setter in a constructor
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lam)
        self.raw_lam.data.copy_(
            self.raw_lam_constraint.inverse_transform(value))


    def forward(self, y):
        y = y / self._scale
        rv = yeo_johnson_transform(y, lam=self.lam)
        return rv


    def untransform(self, y_hat):
        rv = inv_yeo_johnson_transform(y_hat, lam=self.lam)
        rv = rv * self._scale
        return rv


    def log_jacobian(self, y0):
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


class YJWarpedExactMLL(ExactMarginalLogLikelihood):
    """
    The class to train SingleTaskGP and YeoJohnson transform together
    and perform changes requried for that.
    Tested only for single outcome only.
    """
    def __init__(self, likelihood, model: SingleTaskGP, 
                 warping: YeoJohnsonGPytorch, train_Y_raw: torch.Tensor):
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
        super().__init__(likelihood, model)
        self.model: SingleTaskGP

        self.train_Y_raw: torch.Tensor = train_Y_raw

        # Introduce scaling factor to the transform
        _scale = (train_Y_raw.max() - train_Y_raw.min())
        _scale = _scale * (train_Y_raw / _scale).std()
        if (_scale <= 1e-8) or torch.isnan(_scale).any():
            _scale = torch.tensor([1.0])
        _scale = _scale.detach()
        warping._scale = _scale  # should not require grad


    def forward(self, _output, _target,  *params, **kwargs):
        """
        Is used at botorch.optim.closures.model_closures -> @GetLossClosure.register(ExactMarginalLogLikelihood, object, object, NoneType)
        Params:
            _output - prediction of GP. With previous lambda involved.
            _target - model.train_targets just passed through. Can't be used since they are already warped with previous lambda
            params - transformed input for consistency. Just to pass into forward as in ExactMarginalLogLikelihood
        """
        y_raw = self.train_Y_raw
        trainable_warp: YeoJohnsonGPytorch = self.model.outcome_warp_yj

        y_transformed = trainable_warp.forward(y_raw)
        y_transformed = y_transformed.reshape(self.train_Y_raw.shape)
        y_transformed = y_transformed.squeeze(-1)

        self.model.set_train_data(
            inputs=self.model.train_inputs,
            targets=y_transformed,
            strict=False,  # no redundant checks
        )

        output = self.model(*self.model.train_inputs)

        gp_mll_term = super().forward(
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

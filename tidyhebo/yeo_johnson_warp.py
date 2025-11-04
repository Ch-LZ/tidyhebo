"""
Code for Yeo-Johnson (YJ) trainable outcome transform. 
There might be a more elegant way of imlpementing it, but this works.

For implmenetation guidelines thanks to https://github.com/meta-pytorch/botorch/issues/2747

Note:
    Чтобы это заработало пришлось сделать некоторые вещи
    (1) данные `y` должны быть отшкалированы  примерно в диапазон [0, 1] по модулю для лучшей численной обусловленности.
    (2) среднее неинформативного прайора преобразования стоит ровно в точке нестабильности градиента - в 1.0. Именно при lambda=1 YJ преобразование совпадает с нейтральным (ничего не делает)
        Поэтмоу и начальное значение тоже стоит в единице. Поэтому пришлось инициализировать 1 + 1у-6
    И всё равно у такого сетапа иногда (часто) падает обучение. Нужно что-то с этим сделать (
"""

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

# region Gpytroch outcome transform
import torch

from gpytorch import Module as GPytorchModule
from gpytorch.priors import Prior, NormalPrior
from gpytorch.constraints import Interval


class YeoJohnsonGPytorch(GPytorchModule):
    """
    A module for Yeo-Johnson learnable transformation
    """
    def __init__(self, 
                 prior: Prior | None = NormalPrior(loc=1.0 + 1e-6, scale=0.5)):
        super().__init__()
        self.register_parameter("lam", torch.nn.Parameter(torch.tensor(1.0 + 1e-6, dtype=torch.double)))
        if prior is not None:
            self.register_prior("lam_prior", prior, "lam")

        self.register_constraint(
            "lam", Interval(-1.0, 2.0)
        )

    def forward(self, y):
        """
        *Here `y` strongly recommended to be scaled to *approximately* [0, 1] to improve numerical conditioning during optimization.
        """
        return yeo_johnson_transform(y, lam=self.lam)


    def _neg_log_like(self, y0):
        """
        Loss for YJ(y) ~ N(mu, sigma) likelihood
        DON'T use while training with GP simultaneously.
        Mainly for testing purposes.
        """
        if y0.ndim != 2:
            y0 = y0.unsqueeze(-1)

        x_trans = self(y0)
        n_samples = x_trans.shape[0]
        x_trans_var = x_trans.var(axis=0)

        log_var = torch.log(x_trans_var)
        loglike = -n_samples / 2 * log_var
        loglike += (self.lam - 1) * (torch.sign(y0) * torch.log1p(torch.abs(y0))).sum()

        loglike += self.regularization_term()

        return -loglike
    

    def term_to_gp_loss(self, y0):
        """
        Term ot add to GP loss. A part of *positive* log-likelihood.
        Incorporates jacobian term and regularization term, requried for MAP estimation.
        """
        sy_ = torch.sign(y0)
        jacobian_term = sy_ * (self.lam - 1) * torch.log1p(sy_ * y0)
        return jacobian_term + self.regularization_term()


    def regularization_term(self):
        rv = 0.0
        if hasattr(self, "lam_prior"):
            rv = self.lam_prior.log_prob(self.lam)
        return rv


    def _optimize(self, y0):
        """
        For YJ(y) ~ N(mu, sigma) likelihood. Mainly for testing purposes.
        """
        from torch.optim import Adam
        optimizer = Adam(self.parameters(), lr=1e-2)
        for i in range(1000):
            optimizer.zero_grad()

            loss = self._neg_log_like(y0)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.lam.clamp_(0.0, 2.0)

        self.eval()


# endregion
# region closure
from botorch.optim.closures.core import ForwardBackwardClosure
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Any
from botorch.optim.utils import get_parameters


"""
The easiest found workaround to train GP and YJ simultaneously. 
The closure should compute a loss term for the optimzier.
"""
def get_closure(mll: ExactMarginalLogLikelihood, outcome_warp: YeoJohnsonGPytorch):
    """
    outcome_warp: YeoJohnsonGPytorch - a module with parameters

    outcome_warp(y) ~ GP(m, covar[x,x'])
    """
    def closure_wapring(**kwargs: Any):
        """
        adds a loss part from outcome transformation (with prior for regularization)
        """
        model = mll.model
        model_output = model(*model.train_inputs)
        with torch.no_grad():
            warped_targets = outcome_warp(model.train_targets)
        warped_inputs = (model.transform_inputs(X=t_in) for t_in in model.train_inputs)

        log_likelihood = mll(
            model_output,
            warped_targets,
            *warped_inputs,
            **kwargs
        )

        # power_term = outcome_warp.neg_log_like(model.train_targets)
        power_term = outcome_warp.term_to_gp_loss(model.train_targets)

        return -(log_likelihood + power_term)

    return ForwardBackwardClosure(
        forward=closure_wapring,
        parameters=get_parameters(mll, requires_grad=True),
    )
# endregion
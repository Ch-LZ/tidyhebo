import torch

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import Warp
from botorch.models.utils import gpytorch_modules
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior

from .common import dtype, device, GP_Tensor
from .yeo_johnson_warp import YeoJohnsonGPytorch, YJWarpedExactMLL


class SimpleModel:
    def __init__(self,
                 train_x: GP_Tensor,
                 train_y,
                 kernel,
                 power_outcome=True,
                 use_gate=True,
                 warp_input=True,
                 _best_train=False,
                 _max_attempts=10,
                 ):
        """
        kernel - which kind of kernel to use: "RBF" or "Matern52"
        Works for x in (0, 1) - normalized space
        """
        self.model: SingleTaskGP = None
        self.train_y = train_y
        self.train_x: GP_Tensor = train_x
        self.mll = None
        self.kernel_kind = kernel
        self.use_power_outcome_transform = power_outcome
        self.use_gate = use_gate
        self.should_warp_input = warp_input
        self._max_attempts = _max_attempts
        self._best_train = _best_train

        assert self.kernel_kind in ("RBF", "Matern52_Gamma", "Matern52", "default")

        n_x = self.train_x.shape[1]
        self.bounds = torch.repeat_interleave(
            torch.tensor([[0,1],], device=device, dtype=dtype), n_x, dim=0).T


    @staticmethod
    def _make_me_a_kernel(kind, train_x, train_y):
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


    def train_model(self, state_dict=None):
        self._train_model_inner(state_dict=state_dict)

        if self.use_gate:
            if self.model.train_targets.std() < 0.5:
                self.use_power_outcome_transform = False
                self._train_model_inner(state_dict=state_dict)


    def _train_model_inner(self, state_dict=None):
        # building a model
        ## input transform
        input_transform = (self._get_kumaraswamy_warp(self.train_x, self.bounds)
                            if self.should_warp_input else None)

        ## make a GP
        kernel = self._make_me_a_kernel(kind=self.kernel_kind, train_x=self.train_x, train_y=self.train_y)
        self.model = SingleTaskGP(
            train_X=self.train_x,
            train_Y=self.train_y,
            covar_module=kernel,
            input_transform=input_transform,
            # outcome transform - default (scale) and not learnable
        ).to(self.train_x)

        ## outcome transformation
        if self.use_power_outcome_transform:
            outcome_warp_yj = YeoJohnsonGPytorch()

        # pose surrogate quality criterion
        if self.use_power_outcome_transform:
            self.mll = YJWarpedExactMLL(self.model.likelihood, self.model, warping=outcome_warp_yj, train_Y_raw=self.train_y)
        else:
            self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        # load state dict if any for faster convergence
        if state_dict:
            load_state_dict(model=self.model, state_dict=state_dict, silent=True)

        # fit model
        fit_gpytorch_mll(self.mll,
                        max_attempts = self._max_attempts,
                        pick_best_of_all_attempts=self._best_train,
                         )
        if self.use_power_outcome_transform:
            # removes warped outcome from computational graph => they don't require grad
            self.model.train_targets = self.model.train_targets.detach()


    def _get_kumaraswamy_warp(self, train_x, bounds):
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


def load_state_dict(model: SingleTaskGP, state_dict, safe=True, silent=True):
    """
    If model differs from the state dict, don't load it
    This is useful in case of dinamic model building
    """
    if safe:
        model_keys = model.state_dict().keys()
        st_d_keys = state_dict.keys()
        if model_keys == st_d_keys:
            model.load_state_dict(state_dict)
        else:
            if not silent:
                print("State dict doesn't match")
                assert False
    else:
        model.load_state_dict(state_dict)

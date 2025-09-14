import torch

from abc import ABC
from botorch import fit_gpytorch_mll
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.utils import gpytorch_modules
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from .common import dtype, device, GP_Tensor


class SimpleModel(ABC):
    def __init__(self,
                 train_x: GP_Tensor,
                 train_y,
                 kernel,
                 power_outcome=True,
                 warp_input=True,
                 ):
        """
        kernel - which kind of kernel to use: "RBF" or "Matern52"
        Работает в (0, 1) - нормализованом пространстве
        """
        self.model: SingleTaskGP | MixedSingleTaskGP = None
        self.train_y = train_y
        self.train_x: GP_Tensor = train_x
        self.mll = None
        self.kernel_kind = kernel
        self.use_power_outcome_transform = power_outcome
        self.should_warp_input = warp_input

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
        # assert kind in ("RBF", "Matern52_Gamma", "Matern52")
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
        # data transformations
        ## outome transform
        #* трансформ y написан именно таким образом для оптимизации. Потому что в "трансформированном" пространстве оптимизация проходит заметно лучше. 
        #* Если использовать обратное преобразование и дельта-метод для variance, то результат сразу сходится сильно хуже.
        y_to_train = (self._output_transformation(self.train_y) 
                      if self.use_power_outcome_transform else self.train_y)
        input_transform = (self._get_kumaraswamy_warp(self.train_x, self.bounds)
                            if self.should_warp_input else None)
        # initialie model
        # make a GP
        kernel = self._make_me_a_kernel(kind=self.kernel_kind, train_x=self.train_x, train_y=self.train_y)
        self.model = SingleTaskGP(
            train_X=self.train_x,
            train_Y=y_to_train,
            covar_module=kernel,
            input_transform=input_transform
        ).to(self.train_x)
        # pose surrogate quality criterion
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        # load state dict if any for faster convergence
        if state_dict:
            self.model.load_state_dict(state_dict=state_dict)
        # fit model
        fit_gpytorch_mll(self.mll)


    def _output_transformation(self, y_as_is):
        """
        Должен применяться на каждом шаге BO перед 
        Преобразование которое нормализует данные.
        Также по-умолчанию стандартизует их.
        Xarray-like of shape (n_samples, n_features)
        """
        from sklearn.preprocessing import power_transform

        try:
            if y_as_is.min() <= 0:
                y = torch.from_numpy(power_transform(y_as_is / y_as_is.std(), method="yeo-johnson"))
            else:
                y = torch.from_numpy(power_transform(y_as_is / y_as_is.std(), method="box-cox"))
                if y.std() < 0.5:
                    y = torch.from_numpy(power_transform(y_as_is / y_as_is.std(), method="yeo-johnson"))
            if y.std() < 0.5:
                raise RuntimeError("Power transformation failed")
        except:
            y = y_as_is.clone()  # для консистентности выхода, чтобы вегда возвращал копию.
        y = y.type(dtype=dtype)
        return y


    def _get_kumaraswamy_warp(self, train_x, bounds):
        """
        Kumaraswamy преобразованеи для учёта гетероскедастичности
        """
        from botorch.models.transforms.input import Warp
        from gpytorch.priors.torch_priors import LogNormalPrior

        return Warp(  # уже содержит внутри kumaraswamy преобразование (и нормализацию)
                d=train_x.shape[-1],
                indices=list(range(train_x.shape[-1])),
                # use a prior with median at 1.
                # when a=1 and b=1, the Kumaraswamy CDF is the identity function
                concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
                concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
                bounds=bounds,
            )

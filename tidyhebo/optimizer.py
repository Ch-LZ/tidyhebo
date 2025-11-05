import numpy as np
import pandas as pd
import torch
from typing import List, overload
from botorch.acquisition.objective import GenericMCObjective
from botorch.exceptions import ModelFittingError

from .acquisition import HedgedPareto
from .base_model import SimpleModel
from .common import dtype, device, DS_df, DS_Tensor, GA_Tensor, GP_Tensor
from .design_space import MixedDesignSpace
from .evolution_optimizer import optimize_real_hedged_acquisition, optimize_brute_force
from .mixed_params import AbstractParameter
from .yeo_johnson_warp import inv_yeo_johnson_transform

class RealOptimizer:
    def __init__(self, parameters: List[AbstractParameter],
                 y_scale: float, 
                 train_x: DS_df | DS_Tensor = None, 
                 train_y: torch.Tensor | np.ndarray = None,
                 _state_dict=None) -> None:
        """
        y_scale: flaot - a crude but constant estimate of largest y exexpected. 
            For example, for yield should be 100%; for most valudation metrics should be 1.0. 
            Is required to scale outcome to *approximately* [0, 1] (in abs units) for better numerical conditioning in Yeo-Johnson transformation.
        """
        self._y_scale = y_scale
        self._check_user_input(params=parameters, x=train_x, y=train_y)
        self.design_space: MixedDesignSpace = MixedDesignSpace(parameters)  # bounds, parameters
        if isinstance(train_x, pd.DataFrame):
            train_x = train_x[self.design_space.ordered_params]

        self.dim = len(parameters)
        self.train_x: GA_Tensor = self.design_space.from_DS_to_GA(train_x)  # хранятся в пространстве оптимизатора
        self.train_y: torch.Tensor = train_y if isinstance(train_y, torch.Tensor) else torch.tensor(train_y, dtype=dtype).reshape(-1, 1)
        
        # design space config
        self._return_df: bool = isinstance(train_x, DS_df(pd.DataFrame))

        # model config
        self.kernel = "Matern52"
        self.power_outcome: bool = True
        self.warp_input: bool = True
        self.refit_from_scratch: bool = False
        self.model = None  # save for inference
        self.state_dict = _state_dict

        # acquisition function config
        self.use_logei: bool = True
        self.beta: int = 4
        ## beta scheduling 
        self.use_beta_schedule: bool = False
        self.use_srinivas: bool = False
        self.use_beta_upscaled: bool = False

        # GA config
        self.use_brute_force: bool = False
        self.pop_size=128
        self.n_generations=100


    def suggest(self, return_all=False, _from_ds_points: DS_df | DS_Tensor=None) -> DS_df | DS_Tensor:
        power_outcome_this_time = self.power_outcome

        model: SimpleModel = SimpleModel(
            train_x=self.design_space.from_GA_to_GP(self.train_x),
            train_y=self.train_y / self._y_scale,  # in scaled space
            kernel=self.kernel,
            power_outcome=self.power_outcome,
            warp_input=self.warp_input,
        )
        # train model
        state_dict = self.state_dict if not self.refit_from_scratch else None
        try:
            model.train_model(state_dict=state_dict)  # might fail due to YJ transform, but is not expected to fail
            # save to future runs
            self.state_dict = model.model.state_dict()
        except ModelFittingError as e:
            with open("train_x.txt", "a") as f:
                np.savetxt(f, self.train_x)
            with open("train_y.txt", "a") as f:
                np.savetxt(f, self.train_y)
            raise e

        self.model = model
        # create hedged acquisition
        if self.use_beta_schedule:
            current_beta = _schedule_beta(iter=self.train_x.shape[-1] + 1, dim=self.dim,
                           use_srinivas=self.use_srinivas, upscale=self.use_beta_upscaled)
        else:
            current_beta = self.beta

        # objective to optimize acquisition functions in direct space if power_outcome is set        
        objective = None
        if power_outcome_this_time:
            objective = GenericMCObjective(
                lambda y, X: inv_yeo_johnson_transform(
                    y, lam=model.model.outcome_warp_yj.lam).squeeze(-1))

        acqf = HedgedPareto(
            model=model.model,
            best_f=self.train_y.max().item() / self._y_scale,  # in scaled space
            beta=current_beta,
            train_x=self.design_space.from_GA_to_GP(self.train_x),
            objective=objective,
            use_logei=self.use_logei,
        )
        # optimize acquisition function
        if self.use_brute_force or (_from_ds_points is not None):
            cand: GA_Tensor = optimize_brute_force(
                acqf=acqf, opt_space=self.design_space, train_x=self.train_x,
                from_ds_points=_from_ds_points)
        else:
            cand: GA_Tensor = optimize_real_hedged_acquisition(
                acqf=acqf, opt_space=self.design_space, train_x=self.train_x,
                pop_size=self.pop_size, n_generations=self.n_generations,
            )
        # если нет кандидатов - пространство поиска слишком маленькое (почти наверняка)
        # либо что-то пошло не так с оптимизаторами паретто-фронта (вряд ли)
        if cand.numel() == 0:
            print("No candidates was found after paretto-front serach!")
            cand = self.design_space._large_GA_sample
            cand = cand[self.design_space.are_values_new(cand=cand, X_old=self.train_x)]
            if cand.numel() == 0:
                raise RuntimeError("Search space provided by parameters is too small, no new values can be found")

        # reformat back to design space
        res: DS_df | DS_Tensor = self.design_space.from_GA_to_DS(cand, return_df=self._return_df)
        if return_all:
            return res
        if self._return_df:
            return res.sample(n=1)
        else:
            return res[torch.randint(low=0, high=len(res), size=(1,))]


    def observe(self, X, y) -> None:
        """
        Получает наблюдения в пространстве дизайна
        """
        self._check_user_input(params=self.design_space.params, x=X, y=y)
        if isinstance(X, pd.DataFrame):
            X = X[self.design_space.ordered_params]

        y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=dtype)
        if y.dim() == 1:
            y = y.reshape(-1, 1)  # column
        
        x = self.design_space.from_DS_to_GA(X)
        self.train_x = torch.cat([self.train_x, x], axis=0)
        self.train_y = torch.cat([self.train_y, y], axis=0)


    def _check_user_input(self, params, x, y):
        # соответствие типов x, y: pd.DataFrame, np.array || torch.Tensor, torch.Tensor
        # соответствие названий параметров названиям колонок
        if isinstance(x, pd.DataFrame):
            # not empty
            assert len(y) > 0, "Empty y is not allowed"
            assert x.shape[0] > 0, "Empty x is not allowed"
            # lens match
            assert len(y) == x.shape[0], "Number of samples in X and y mismatch"
            # right type
            assert isinstance(y, np.ndarray), "If x is DataFrame, y should be np.array"
            assert len(y.shape) in (1, 2)
            # right column names
            params_names = [p.name for p in params]
            assert frozenset(x.columns.to_list()) == frozenset(params_names), "Parameter names and column names doesn't match"
            assert x.columns.is_unique, "Column names should be unique"


def _schedule_beta(iter, dim, use_srinivas=False, upscale=False):
    if use_srinivas:
        dim = 6
        delta = 0.01
        beta = 2 * ((2 + 2*dim) * np.log(iter) + np.log((2*np.pi**2) / (2 * delta)))
    else:
        delta = 0.01
        upsi = 0.5
        # формула из кода HEBO, на практике растет разумно.
        beta = upsi * 2 * ((2.0 + dim / 2.0) * np.log(iter) + np.log(3 * np.pi**2 / (3 * delta)))

    if upscale:
        beta = beta * 2 / np.pi
    return beta

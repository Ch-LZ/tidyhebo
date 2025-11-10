import numpy as np
import pandas as pd
import torch
from typing import List
from botorch.acquisition.objective import GenericMCObjective

from .acquisition import HedgedPareto
from .base_model import SimpleModel
from .common import dtype, device, DS_df, DS_Tensor, GA_Tensor, GP_Tensor
from .design_space import MixedDesignSpace
from .evolution_optimizer import optimize_real_hedged_acquisition, optimize_brute_force
from .mixed_params import AbstractParameter
from .yeo_johnson_warp import inv_yeo_johnson_transform


class RealOptimizer:
    def __init__(self, parameters: List[AbstractParameter],
                 train_x: DS_df | DS_Tensor = None, 
                 train_y: torch.Tensor | np.ndarray = None,
                 _state_dict=None) -> None:

        self._y_scale = 1.0
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
        self.shift_outcome: bool = False    # was never better with it
        self.use_two_lambdas: bool = False  # was never better with it
        self._use_gate: bool = True

        self.warp_input: bool = True
        self.refit_from_scratch: bool = False
        self.model = None
        self.state_dict = _state_dict

        # acquisition function config
        self.use_logei: bool = True
        self.beta: int = 4

        # GA config
        self.use_brute_force: bool = False
        self.pop_size=128
        self.n_generations=100


    def suggest(self, return_all=False, _from_ds_points: DS_df | DS_Tensor=None) -> DS_df | DS_Tensor:
        # select a model based on criterion
        power_outcome_this_time = self.power_outcome
        try:
            self._y_scale = self.train_y.max().item() - self.train_y.min().item()
            _std0 = (self.train_y / self._y_scale).std()
            self._y_scale = self._y_scale * _std0

            model: SimpleModel = SimpleModel(
                train_x=self.design_space.from_GA_to_GP(self.train_x),
                train_y=self.train_y / self._y_scale,  # in scaled space
                kernel=self.kernel,
                warp_input=self.warp_input,
                power_outcome=self.power_outcome,
                shift_outcome=self.shift_outcome,
                use_two_lambdas=self.use_two_lambdas,
            )
            model.train_model(state_dict=self.state_dict)

            # evaluate the criterion
            if self._use_gate:
                with torch.no_grad():
                    y_tr = model.outcome_warp_yj.forward(self.train_y / self._y_scale)
                    assert y_tr.std().item() >= 0.5  # else retry
        except:
            power_outcome_this_time = False
            self._y_scale = 1.0

            model: SimpleModel = SimpleModel(
                train_x=self.design_space.from_GA_to_GP(self.train_x),
                train_y=self.train_y / self._y_scale,  # in scaled space
                kernel=self.kernel,
                power_outcome=power_outcome_this_time,
                warp_input=self.warp_input,
            )
            model.train_model(state_dict=self.state_dict)

        self.model = model
        self.state_dict = self.model.model.state_dict() if not self.refit_from_scratch else None
        # self.state_dict = None

        # objective to optimize acquisition functions in direct space if power_outcome is set        
        objective = None
        if power_outcome_this_time:
            objective = GenericMCObjective(
                lambda y, X: model.model.outcome_warp_yj.untransform(y).squeeze(-1))

        acqf = HedgedPareto(
            model=model.model,
            best_f=self.train_y.max().item() / self._y_scale,  # in scaled space
            beta=self.beta,
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


#todo: пофиксить эту функцию или вообще убрать за ненадобностью
def train_model(
        train_x,
        train_y,
        kernel="Matern52",
        warp_input=True,
        power_outcome=True,
        use_logei=True,
        beta=4,
):
    power_outcome_this_time = power_outcome
    _y_scale = None
    try:
        _y_scale = train_y.max().item() - train_y.min().item()
        _std0 = (train_y / _y_scale).std()
        _y_scale = _y_scale * _std0

        model: SimpleModel = SimpleModel(
            train_x=train_x,
            train_y=train_y / _y_scale,  # in scaled space
            kernel=kernel,
            power_outcome=power_outcome,
            warp_input=warp_input,
        )
        model.train_model(state_dict=None)  # restart from scratch every time

        # criterion
        with torch.no_grad():
            y_tr = model.outcome_warp_yj.forward(train_y / _y_scale)
            assert y_tr.std().item() >= 0.5  # else retry
    except:
        power_outcome_this_time = False
        _y_scale = 1.0

        model: SimpleModel = SimpleModel(
            train_x=train_x,
            train_y=train_y / _y_scale,  # in scaled space
            kernel=kernel,
            power_outcome=power_outcome,
            warp_input=warp_input
        )
        model.train_model(state_dict=None)  # restart from scratch every time

    objective = None
    if power_outcome_this_time:
        objective = GenericMCObjective(
            lambda y, X: inv_yeo_johnson_transform(
                y, lam=model.model.outcome_warp_yj.lam).squeeze(-1))

    acqf = HedgedPareto(
        model=model.model,
        best_f=train_y.max().item() / _y_scale,  # in scaled space
        beta=beta,
        train_x=train_x,
        objective=objective,
        use_logei=use_logei,
    )
    return model, acqf

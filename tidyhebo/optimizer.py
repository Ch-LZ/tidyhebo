import numpy as np
import pandas as pd
import torch
from typing import List
from botorch.acquisition.objective import GenericMCObjective

from .acquisition import HedgedPareto
from .base_model import SimpleModel
from .common import dtype, device, DS_df, DS_Tensor, GA_Tensor, GP_Tensor
from .design_space import MixedDesignSpace
from .evolution_optimizer import optimize_real_hedged_acquisition
from .mixed_params import AbstractParameter


class tidyHEBO:
    def __init__(self, parameters: List[AbstractParameter],
                 train_x: DS_df | DS_Tensor = None, 
                 train_y: torch.Tensor | np.ndarray = None,
                 n_init: int | None = None,
                 _state_dict=None) -> None:
        """
        n_init - сколько точек равномерно (соболем) семплировать из пространства дизайна. 
        Введено для целей равномерной инициализации (sobol sampling ~lhs) при отсутствии train_x и train_y.
        """
        self._check_user_input(params=parameters, x=train_x, y=train_y)
        self.design_space: MixedDesignSpace = MixedDesignSpace(parameters)  # bounds, parameters

        if isinstance(train_x, pd.DataFrame):
            train_x = train_x[self.design_space.ordered_params]

        # train_x 
        dim = len(parameters)
        self.train_x: GA_Tensor = torch.empty((0, dim), dtype=torch.double) if (train_x is None) else self.design_space.from_DS_to_GA(train_x)

        # train_y
        self.train_y: torch.Tensor
        if train_y is None:
            self.train_y = torch.empty((0, 1), dtype=torch.double)
        elif isinstance(train_y, torch.Tensor):
            self.train_y = train_y
        else:
            self.train_y = torch.tensor(train_y, dtype=dtype).reshape(-1, 1)

        # init generator
        self._init_design_generator: Generator[DS_df, None, None] = None
        if n_init:
            assert (train_x is None) and (train_y is None), "n_init for initialization purposes only"
            assert n_init > 0
            #* note: there is return_df always on.
            self._init_design_generator = gen_init_design(n_init, self.design_space, return_df=True)
        else:
            assert (train_x is not None) and (train_y is not None), "No initialization for optimizer observed"
        
        self.model: SimpleModel = None
        self.state_dict = _state_dict

        # design space config
        self._return_df: bool = isinstance(train_x, DS_df(pd.DataFrame)) or (n_init is not None)

        # model config
        self.kernel = "Matern52"
        self.power_outcome: bool = True
        self._use_gate: bool = True

        self.warp_input: bool = True
        self.refit_from_scratch: bool = False
        self._best_train: bool = False
        self._max_attempts = 10

        # acquisition function config
        self.use_logei: bool = True
        self.beta: int = 4
        self.quadrature_size: int = 64
        self.mc_samples: int = 4096

        # GA config
        self.pop_size=128
        self.n_generations=100


    def suggest(self, return_all=False) -> DS_df | DS_Tensor:
        # if initialization sampling required
        if self._init_design_generator:
            another_init_cand = next(self._init_design_generator, None)
            if another_init_cand is not None:
                return another_init_cand

        # train model
        model, acqf, _state_dict = train_model_and_AF(
            train_x=self.design_space.from_GA_to_GP(self.train_x),
            train_y=self.train_y,
            kernel=self.kernel,
            warp_input=self.warp_input,
            power_outcome=self.power_outcome,
            _use_gate=self._use_gate,
            state_dict=self.state_dict,
            refit_from_scratch=self.refit_from_scratch,
            beta=self.beta,
            use_logei=self.use_logei,
            _best_train=self._best_train,
            _max_attempts=self._max_attempts,
            quadrature_size=self.quadrature_size,
            mc_samples=self.mc_samples,
        
        )
        self.model = model,
        self.state_dict = _state_dict        

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
            assert len(y.shape) in (0, 1, 2)
            # right column names
            params_names = [p.name for p in params]
            assert frozenset(x.columns.to_list()) == frozenset(params_names), "Parameter names and column names doesn't match"
            assert x.columns.is_unique, "Column names should be unique"


def train_model_and_AF(
    train_x, train_y,
    kernel="Matern52",
    warp_input=True,
    power_outcome=True,
    _use_gate=True,
    state_dict=None,
    refit_from_scratch=False,
    _best_train=False,
    _max_attempts=10, 
    beta=4, 
    use_logei=True,
    quadrature_size=64,
    mc_samples=4096,
    ):
        # train model
    model: SimpleModel = SimpleModel(
        train_x=train_x,
        train_y=train_y,
        kernel=kernel,
        warp_input=warp_input,
        power_outcome=power_outcome,
        use_gate=_use_gate,
        _best_train=_best_train,
        _max_attempts=_max_attempts,
    )
    model.train_model(state_dict=state_dict)


    new_state_dict = model.model.state_dict() if not refit_from_scratch else None

    # objective to optimize acquisition functions in direct space if power_outcome is set        
    objective = None
    if model.use_power_outcome_transform:
        objective = GenericMCObjective(
            lambda y, X: model.model.outcome_warp_yj.untransform(y).squeeze(-1))

    acqf = HedgedPareto(
        model=model.model,
        best_f=train_y.max().item(),
        beta=beta,
        train_x=train_x,
        objective=objective,
        outcome_warp=(
            model.model.outcome_warp_yj
            if model.use_power_outcome_transform
            else None
        ),
        use_logei=use_logei,
        quadrature_size=quadrature_size,
        mc_samples=mc_samples,
    )
    return model, acqf, new_state_dict


from typing import Generator
from torch.quasirandom import SobolEngine

def gen_init_design(n_init: int, design_space: MixedDesignSpace, return_df: bool):
    assert n_init > 0
    # prepare design
    dim = len(design_space.params)
    sampler = SobolEngine(dim, scramble=True)
    for i in range(n_init):
        x_init_GP = sampler.draw(1, dtype=torch.double)
        yield design_space.from_GP_to_DS(x_init_GP, return_df=return_df)

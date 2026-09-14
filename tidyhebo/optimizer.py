import numpy as np
import pandas as pd
import torch
from typing import Generator, List
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
                 _state_dict=None
                 ) -> None:
        """
        n_init - how many points to sample uniformly (Sobol) from the design space.
        Used for uniform initialization when train_x and train_y are not provided.
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
        self.consider_fast_acq_path: bool = True
        self.quadrature_size: int = 512  # for fast acquisition calculation if available
        self.mc_samples: int = 4096

        # GA config
        self.pop_size = 128
        self.n_generations = 100

    # -------------------------------------------------------------------------
    # Main public methods
    # -------------------------------------------------------------------------

    def suggest(self, candidates: DS_df | DS_Tensor = None, return_all=False) -> DS_df | DS_Tensor:
        """
        Suggest next point(s) to evaluate.

        Args:
            candidates: Optional discrete set of candidate points (in design space).
                        If provided, acquisition is evaluated directly on these points.
            return_all: If True, return all points from the Pareto frontier
                        (discrete) or all candidates found by the evolutionary optimizer
                        (continuous). Otherwise, return a single randomly chosen point.
        """
        init_pts_if_any: DS_df | DS_Tensor | None = self._suggest_init_points_if_remoained(return_all=return_all)
        if init_pts_if_any is not None:
            return init_pts_if_any

        # Train model and build acquisition function
        model, acqf, state_dict = self._train_and_get_acqf()
        self.model = model
        self.state_dict = state_dict

        # Get full set of candidates in GA space
        if candidates is not None:
            _check_input_x(self.design_space.params, candidates, _y=None)
            if isinstance(candidates, pd.DataFrame):
                candidates = candidates[self.design_space.ordered_params]
            cand_ga = self._suggest_discrete(candidates, acqf)   # returns all Pareto points
        else:
            cand_ga = self._suggest_continuous(acqf)             # returns all points from EA

        # Convert to design space
        result = self.design_space.from_GA_to_DS(cand_ga, return_df=self._return_df)

        # Decide how many to return
        if return_all:
            return result
        else:
            if self._return_df:
                return result.sample(n=1)
            else:
                return result[torch.randint(len(result), (1,))]

    def observe(self, X, y) -> None:
        """
        Add new observations to the training data.

        X: 2D array-like, where one row for observations
        y: 2D array-like, where one row for one y value (a column)
        
        Expected data types for pair (X, y) are (1) for usage (pd.DataFrame, np.array) or (2) for testing (torch.tensor, torch.tensor)
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

    # -------------------------------------------------------------------------
    # Private helper methods for suggestion
    # -------------------------------------------------------------------------
    def _suggest_init_points_if_remoained(self, return_all=False) -> DS_df | DS_Tensor | None:
        # If initialization sampling is still ongoing
        if self._init_design_generator:
            if return_all:  # gather all remained points from the generator
                all_points = list(self._init_design_generator)
                self._init_design_generator = None  # генератор исчерпан
                if self._return_df:  # DS_df
                    return pd.concat(all_points, ignore_index=True) if all_points else None
                else:   # DS_Tensor
                    return torch.cat(all_points, dim=0) if all_points else None
            else:  # return one
                another_init_cand = next(self._init_design_generator, None)
                if another_init_cand is not None:  
                    return another_init_cand  # already formatted to required type
                else:  # might be first time we know generator is exhausted
                    self._init_design_generator = None
                    return None
        return None


    def _train_and_get_acqf(self):
        """Train the GP model and return the acquisition function."""
        return train_model_and_AF(
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
            consider_fast_acq_path=self.consider_fast_acq_path,
            quadrature_size=self.quadrature_size,
            mc_samples=self.mc_samples,
        )

    def _suggest_discrete(self, candidates: DS_df | DS_Tensor, acqf) -> GA_Tensor:
        """
        Evaluate acquisition on a discrete set of candidates and return
        the Pareto frontier (all non‑dominated points) in GA space.
        """
        from botorch.utils.multi_objective.pareto import is_non_dominated

        cand_ga = self.design_space.from_DS_to_GA(candidates)  # shape (n, d)
        print(f"cand_ga_shape: {cand_ga.shape}")

        with torch.no_grad():
            acq_values = acqf(cand_ga.unsqueeze(1))  # shape (n, 3)
        print(f"acq_values_shape: {acq_values.shape}")
        pareto_mask = is_non_dominated(acq_values, deduplicate=False).detach().cpu().numpy()
        print(f"pareto_mask.shape: {pareto_mask}")
        return cand_ga[pareto_mask]  # always have at least one point

    def _suggest_continuous(self, acqf) -> GA_Tensor:
        """
        Run evolutionary optimization over the entire design space and return
        all candidate points found (typically a Pareto frontier) in GA space.
        """
        cand_ga: GA_Tensor = optimize_real_hedged_acquisition(
            acqf=acqf,
            opt_space=self.design_space,
            train_x=self.train_x,
            pop_size=self.pop_size,
            n_generations=self.n_generations,
        )
        if cand_ga.numel() == 0:
            print("No candidates found after Pareto-front search!")
            cand_ga = self.design_space._large_GA_sample
            cand_ga = cand_ga[self.design_space.are_values_new(cand=cand_ga, X_old=self.train_x)]
            if cand_ga.numel() == 0:
                raise RuntimeError("Search space provided by parameters is too small, no new values can be found")
        return cand_ga

    def _check_user_input(self, params, x, y):
        # соответствие типов x, y: pd.DataFrame, np.array || torch.Tensor, torch.Tensor
        # соответствие названий параметров названиям колонок
        _check_input_x(params, x, y)
        _check_input_y(params, x, y)
        if isinstance(x, pd.DataFrame):
            assert len(y) == x.shape[0], "Number of samples in X and y mismatch"


# -----------------------------------------------------------------------------
# External helper functions (unchanged)
# -----------------------------------------------------------------------------
def _check_input_x(params, x, _y):
    if isinstance(x, pd.DataFrame):
        assert x.shape[0] > 0, "Empty x is not allowed"
        # right column names
        params_names = [p.name for p in params]
        # must much in order
        assert frozenset(x.columns.to_list()) == frozenset(params_names), f"Parameter names: {params_names} and columns {x.columns.to_list()} names doesn't match"
        assert x.columns.is_unique, "Column names should be unique"        

def _check_input_y(_params, x, y):
    if isinstance(x, pd.DataFrame):
        assert len(y) > 0, "Empty y is not allowed"
        assert isinstance(y, np.ndarray), "If x is DataFrame, y should be np.array"
        assert len(y.shape) in (0, 1, 2)



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
    consider_fast_acq_path=True,
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
        consider_fast_acq_path=consider_fast_acq_path,
        quadrature_size=quadrature_size,
        mc_samples=mc_samples,
    )
    return model, acqf, new_state_dict


def gen_init_design(n_init: int, design_space: MixedDesignSpace, return_df: bool):
    """Generator for Sobol initialization points."""
    assert n_init > 0
    dim = len(design_space.params)
    sampler = torch.quasirandom.SobolEngine(dim, scramble=True)
    for i in range(n_init):
        x_init_GP = sampler.draw(1, dtype=torch.double)
        yield design_space.from_GP_to_DS(x_init_GP, return_df=return_df)

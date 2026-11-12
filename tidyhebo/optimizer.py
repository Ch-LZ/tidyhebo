import numpy as np
import pandas as pd
import torch
from typing import Generator, List
from numpy.typing import ArrayLike
from itertools import islice

from .acquisition import MACE, EI, LogEI  # Register acquisition implementations.
from .base_acquisition import BaseAcquisition
from .base_model import SurrogateModel
from .config import TidyHEBOConfig
from .candidate_search import CandidateSearch
from .common import GA_Tensor
from .design_space import MixedDesignSpace
from .mixed_params import AbstractParameter


class tidyHEBO:
    def __init__(self, parameters: List[AbstractParameter],
                 train_x: pd.DataFrame | None = None,
                 train_y: ArrayLike | None = None,
                 n_init: int | None = None,
                 _state_dict=None,
                 *, config: TidyHEBOConfig | None = None,
                 ) -> None:
        """Initialize from observations or a positive-size Sobol design.

        Observations take priority over n_init when both are supplied.
        """
        self.config = TidyHEBOConfig() if config is None else config
        self.config.check_supported()
        self._validator = InputValidator([p.name for p in parameters])
        train_x, train_y, n_init = self._validator.ensure_user_input(train_x, train_y, n_init)
        self.design_space: MixedDesignSpace = MixedDesignSpace(parameters)
        self.train_x: GA_Tensor = self.design_space.from_DS_to_GA(train_x)
        self.train_y = torch.tensor(train_y, dtype=torch.double)
        self._init_design_generator: Generator[pd.DataFrame, None, None] | None = (
            gen_init_design(n_init, self.design_space) if n_init is not None else None
        )

        self.model: SurrogateModel = None
        self.state_dict = _state_dict
        # Snapshot compatibility for benchmark adapters predating the removal
        # of adaptive-direction search. Retained layered search leaves it empty.
        self.direction_state = {}

    @classmethod
    def from_config(cls, config: TidyHEBOConfig, parameters: List[AbstractParameter], *,
                    n_init: int | None = None, train_x: pd.DataFrame | None = None,
                    train_y: ArrayLike | None = None) -> "tidyHEBO":
        """Construct an optimizer from a configuration and a problem domain."""
        return cls(parameters, train_x=train_x, train_y=train_y, n_init=n_init, config=config)

    # -------------------------------------------------------------------------
    # Main public methods
    # -------------------------------------------------------------------------

    def suggest(self) -> pd.DataFrame:
        """Suggest using the current immutable config as the sole settings source.

        q comes from config.acquisition; return_all and the search candidate
        pool also come from config. A front element is one complete q-batch.
        During initialization, return up to q remaining Sobol points, or all
        remaining initialization points when config.return_all is true.
        """
        self.config.check_supported()
        q, return_all = self.config.acquisition.q, self.config.return_all
        search = CandidateSearch.from_config(self.config.search, design_space=self.design_space)
        init_pts_if_any: pd.DataFrame | None = self._suggest_init_points_if_remoained(return_all=return_all, q=q)
        if init_pts_if_any is not None:
            return init_pts_if_any
        if q > 1 and len(self.train_y) == 0:
            raise RuntimeError("Observe initialization points before requesting a model-based batch")

        # Train model and build acquisition function
        model, acqf, state_dict = self._train_and_get_acqf()
        self.model = model
        self.state_dict = state_dict
        cand_ga = search.search(acqf, self.train_x)
        # Each front element is a complete q-batch, including q=1.
        batches = cand_ga.unsqueeze(1) if q == 1 else cand_ga
        if not return_all:
            selected = torch.randint(len(batches), (1,)).item()
            cand_ga = batches[selected]

        # DataFrames are flat tables; retain joint front boundaries in the index.
        result = self.design_space.from_GA_to_DS(
            cand_ga.reshape(-1, cand_ga.shape[-1]), return_df=True,
        )
        if return_all and q > 1:
            result.index = pd.MultiIndex.from_product(
                [range(len(batches)), range(q)], names=["batch", "point"],
            )
        return result

    def observe(self, X: pd.DataFrame, y: ArrayLike) -> None:
        """Append observations; y must have shape (n, 1), matching X by row position."""
        X, y, _ = self._validator.ensure_user_input(X, y)
        x = self.design_space.from_DS_to_GA(X)
        y = torch.tensor(y, dtype=torch.double)
        self.train_x = torch.cat([self.train_x, x], axis=0)
        self.train_y = torch.cat([self.train_y, y], axis=0)

    # -------------------------------------------------------------------------
    # Private helper methods for suggestion
    # -------------------------------------------------------------------------
    def _suggest_init_points_if_remoained(self, return_all=False, q=1) -> pd.DataFrame | None:
        if self._init_design_generator is None:
            return None
        points = list(self._init_design_generator if return_all else islice(self._init_design_generator, q))
        if return_all or len(points) < q:
            self._init_design_generator = None
        return pd.concat(points, ignore_index=True) if points else None

    def _train_and_get_acqf(self):
        """Fit the selected surrogate and build the configured acquisition."""
        return train_model_and_AF(
            self.design_space.from_GA_to_GP(self.train_x), self.train_y,
            config=self.config, state_dict=self.state_dict,
        )


class InputValidator:
    """Validate public observations and normalize their column order without mutation."""

    def __init__(self, columns: List[str]):
        self.columns = list(columns)

    def ensure_user_input(self, X: pd.DataFrame | None, y: ArrayLike | None,
                          n_init: int | None = None):
        """Return ordered X, array y and the initialization size, if needed."""
        if X is None and y is None:
            if isinstance(n_init, bool) or not isinstance(n_init, (int, np.integer)) or n_init <= 0:
                raise ValueError("Provide observations or a positive integer n_init")
            return pd.DataFrame(columns=self.columns), np.empty((0, 1)), n_init
        if X is None or y is None:
            raise ValueError("Provide both X and y")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame")
        if len(X) == 0:
            raise ValueError("X must not be empty")
        if not X.columns.is_unique:
            raise ValueError("X columns must be unique")
        if set(X.columns) != set(self.columns):
            raise ValueError(f"X columns must match {self.columns}")
        y = np.asarray(y)
        if y.shape != (len(X), 1):
            raise ValueError("y must have shape (n, 1), matching the number of X rows")
        return X[self.columns], y, None


def train_model_and_AF(train_x, train_y, *, config: TidyHEBOConfig, state_dict=None):
    """Build model/acquisition from the same per-run configuration."""
    model = SurrogateModel.from_config(config.model, train_x=train_x, train_y=train_y)
    new_state_dict = model.train_model(state_dict=state_dict)
    warp = model.model.outcome_warp_yj if model.use_power_outcome_transform else None
    acqf = BaseAcquisition.from_config(
        config.acquisition, model=model.model, best_f=train_y.max().item(),
        train_x=train_x, outcome_warp=warp,
    )
    return model, acqf, new_state_dict


def gen_init_design(n_init: int, design_space: MixedDesignSpace):
    """Generator for Sobol initialization points."""
    assert n_init > 0
    dim = len(design_space.params)
    sampler = torch.quasirandom.SobolEngine(dim, scramble=True)
    for i in range(n_init):
        x_init_GP = sampler.draw(1, dtype=torch.double)
        yield design_space.from_GP_to_DS(x_init_GP, return_df=True)

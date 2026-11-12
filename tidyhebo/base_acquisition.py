"""Common callable interface and registry for acquisition functions."""

from abc import ABC, abstractmethod

from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import GenericMCObjective
from torch import Tensor

from .config import AcquisitionConfig
from .registry import Registrable


class BaseAcquisition(Registrable, AcquisitionFunction, ABC):
    """Construct from config and return (..., n_obj) maximization scores.

    Inputs have shape (..., q, d). The fitted GP and the outcome-warp retained
    by its gate are explicit dependencies; configuration contains no fit state.
    """

    n_obj: int

    def __init__(self, model, config: AcquisitionConfig):
        super().__init__(model)
        self.config = config

    @classmethod
    def from_config(cls, config, **dependencies):
        """Dispatch the explicitly configured MACE evaluator."""
        if cls is BaseAcquisition and config._target_ in ("MACE", "qMACE"):
            from .acquisition import AnalyticMACE, MACE, QuadratureMACE

            implementation = {"analytic": AnalyticMACE, "quadrature": QuadratureMACE, "qmc": MACE}[
                config.evaluation
            ]
            return implementation(config=config, **dependencies)
        return super().from_config(config, **dependencies)

    @property
    def q(self):
        return self.config.q

    @property
    def n_objectives(self):
        return self.n_obj

    @property
    def scale(self):
        return self.config.scale

    def _original_scale_objective(self, outcome_warp):
        if outcome_warp is None:
            return None
        return GenericMCObjective(lambda y, X: outcome_warp.untransform(y).squeeze(-1))

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate complete q-batches, retaining the objective axis."""
        ...

"""Immutable per-run settings. Runtime caches and learned state are not configs."""

from dataclasses import dataclass, field
from typing import Literal, Protocol

import pandas as pd


class RegistryConfig(Protocol):
    """Configuration identifying a registered implementation."""

    @property
    def _target_(self) -> str: ...


@dataclass(frozen=True)
class ModelConfig:
    _target_: Literal["joint", "centered", "profiled"] = "joint"
    centered_init: Literal["preserve_mean", "zero_beta"] = "preserve_mean"
    warm_start: bool = True
    kernel: Literal["Matern52", "Matern52_Gamma", "RBF", "default"] = "Matern52"
    input_warp: bool = True
    warp_fit: Literal["joint", "separate"] = "joint"
    output_transform: Literal["standardize", "yeo_johnson"] = "yeo_johnson"
    gate: bool = True
    max_attempts: int = 10
    pick_best: bool = False


@dataclass(frozen=True)
class AcquisitionConfig:
    _target_: Literal["LogEI", "EI", "MACE", "qMACE", "qLogEI"] = "MACE"
    q: int = 1
    scale: Literal["original", "latent"] = "original"
    evaluation: Literal["analytic", "quadrature", "qmc"] | None = None
    quadrature_size: int = 512  # Preserve the established sequential runtime default.
    mc_samples: int = 4096
    qmc_seed: int | None = None
    beta: float = 4.0
    cache_invariants: bool = True
    reuse_inverse_warp: bool = True
    posterior: Literal["native", "frozen_q1"] = "native"

    def __post_init__(self):
        # Family defaults are fixed, never inferred from fitted model state.
        if self.evaluation is None:
            object.__setattr__(self, 'evaluation',
                               'qmc' if self._target_ in ('MACE', 'qMACE') else 'analytic')
        if self.evaluation not in ('analytic', 'quadrature', 'qmc'):
            raise ValueError('evaluation must be analytic, quadrature or qmc')
        if not isinstance(self.q, int) or isinstance(self.q, bool) or self.q < 1:
            raise ValueError("q must be a positive integer")
        if self._target_ == "qMACE" and self.q < 2:
            raise ValueError("qMACE requires q > 1")
        if self._target_ not in ("qMACE", "qLogEI") and self.q != 1:
            raise ValueError(f"{self._target_} requires q == 1")


@dataclass(frozen=True)
class StopConfig:
    mode: Literal["budget", "hv_patience"] = "budget"
    check_every: int = 5
    patience: int = 4
    relative_tolerance: float = 1e-3
    absolute_tolerance: float = 0.0


@dataclass(frozen=True)
class SearchConfig:
    _target_: Literal["pymoo", "discrete", "botorch"]


@dataclass(frozen=True)
class GASearchConfig(SearchConfig):
    _target_: Literal["pymoo"] = "pymoo"
    pop_size: int = 128
    max_generations: int = 100
    n_offsprings: int = 128
    reference_partitions: int = 12
    fallback_samples: int = 4096  # Maximum random proposals if GA finds no feasible result.
    matching_distance: Literal["euclidean", "kernel"] = "euclidean"
    stop: StopConfig = field(default_factory=StopConfig)


@dataclass(frozen=True)
class DiscreteSearchConfig(SearchConfig):
    candidates: pd.DataFrame = field(kw_only=True, repr=False)
    _target_: Literal["discrete"] = "discrete"


@dataclass(frozen=True)
class BoTorchSearchConfig(SearchConfig):
    _target_: Literal["botorch"] = "botorch"
    num_restarts: int = 5
    raw_samples: int = 128
    inference_device: str = "cpu"
    maxiter: int = 100
    batch_limit: int = 4
    sequential: bool = True

    def __post_init__(self):
        if self.inference_device != "cpu":
            raise ValueError("BoTorch search currently supports CPU inference only")
        for name in ("num_restarts", "raw_samples", "maxiter", "batch_limit"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        if type(self.sequential) is not bool:
            raise ValueError("sequential must be boolean")


@dataclass(frozen=True)
class TidyHEBOConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    search: SearchConfig = field(default_factory=GASearchConfig)
    # For qMACE, one front element is a complete q-batch.
    return_all: bool = False

    def check_supported(self) -> None:
        """Reject declared but unimplemented execution paths before any fitting.

        Advanced cache switches are reserved: only their default behavior is
        currently supported. Their presence does not enable a frozen posterior
        or shared inverse-sample implementation.
        """
        a = self.acquisition
        if self.model.warp_fit != "joint":
            raise NotImplementedError("Separate warp fitting is not implemented")
        if a.scale != "original" or a.posterior != "native":
            raise NotImplementedError("Only original-scale acquisition with the native posterior is implemented")
        if not a.cache_invariants or not a.reuse_inverse_warp:
            raise NotImplementedError("Nondefault cache switches are not implemented")
        if a._target_ in ("qMACE", "qLogEI") and a.evaluation != "qmc":
            raise NotImplementedError("qMACE/qLogEI requires qmc evaluation")
        if isinstance(self.search, GASearchConfig):
            if self.search.stop.mode != "budget":
                raise NotImplementedError("Only generation-budget termination is implemented")
        elif isinstance(self.search, DiscreteSearchConfig):
            if a.q != 1:
                raise NotImplementedError("Explicit discrete candidate sets support q=1 only")
        elif isinstance(self.search, BoTorchSearchConfig):
            if a._target_ not in ("EI", "LogEI", "qLogEI"):
                raise ValueError("BoTorch search requires scalar EI/LogEI/qLogEI acquisition")
        else:
            from .layered.config import LayeredSearchConfig
            if not isinstance(self.search, LayeredSearchConfig):
                raise NotImplementedError("Unknown candidate search configuration")
            if a._target_ not in ('MACE','qMACE'):
                raise ValueError('Layered leader requires the MACE triplet')

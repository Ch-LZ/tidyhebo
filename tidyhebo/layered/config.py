"""Configuration for the three-stage retained qMACE search."""

from dataclasses import dataclass, field
from typing import Literal
from ..config import SearchConfig


@dataclass(frozen=True)
class ScalarizationConfig:
    method: Literal["augmented_chebyshev"] = "augmented_chebyshev"
    normalize_weights: bool = False
    calibration_samples: int = 256
    rho: float = 0.01
    direction_floor: float = 0.02

    def __post_init__(self):
        if self.method != "augmented_chebyshev":
            raise ValueError("The retained leader uses augmented_chebyshev")
        if self.calibration_samples < 2 or self.rho < 0 or not 0 < self.direction_floor < 1:
            raise ValueError("Invalid scalarization calibration/weights")


@dataclass(frozen=True)
class InitializationConfig:
    anchors: bool = True
    directions: int = 6
    restarts: int = 3
    local_steps: int = 30
    use_point_pool: bool = False
    pool_generations: int = 5

    def __post_init__(self):
        if self.directions < -1 or min(self.restarts, self.local_steps, self.pool_generations) < 1:
            raise ValueError("Invalid initialization budget")


@dataclass(frozen=True)
class LevelConfig:
    algorithm: Literal["initial_only", "nsga3"] = "nsga3"
    population: int = 48
    generations: int = 20
    reference_partitions: int = 6
    memetic: Literal["none", "end"] = "none"
    refine_fraction: float = 0.2
    local_steps: int = 15
    pool_size: int = 96
    deeper_fronts: bool = True

    def __post_init__(self):
        if self.algorithm not in ("initial_only", "nsga3") or self.memetic not in ("none", "end"):
            raise ValueError("Unsupported leader level")
        if (
            min(
                self.population, self.generations, self.reference_partitions, self.local_steps, self.pool_size
            )
            < 1
            or not 0 < self.refine_fraction <= 1
        ):
            raise ValueError("Invalid level budget")


@dataclass(frozen=True)
class LayeredSearchConfig(SearchConfig):
    _target_: str = "layered"
    point_init: InitializationConfig = field(default_factory=InitializationConfig)
    batch_init: InitializationConfig = field(
        default_factory=lambda: InitializationConfig(use_point_pool=True)
    )
    points: LevelConfig = field(default_factory=lambda: LevelConfig(algorithm="initial_only"))
    batches: LevelConfig = field(default_factory=lambda: LevelConfig(memetic="end"))
    scalarization: ScalarizationConfig = field(default_factory=ScalarizationConfig)
    matching_distance: Literal["euclidean", "kernel"] = "kernel"
    inference_device: str = "cpu"
    evaluation_chunk: int = 32
    progress_every: int = 5

    def __post_init__(self):
        if self._target_ != "layered" or self.matching_distance not in ("euclidean", "kernel"):
            raise ValueError("Invalid layered search target/matching")
        if not isinstance(self.inference_device, str):
            raise ValueError("inference_device must be a device string")
        if self.evaluation_chunk < 1 or self.progress_every < 1:
            raise ValueError("Invalid evaluation/progress chunk")
        if self.points.algorithm != "initial_only" or self.batches.algorithm != "nsga3":
            raise ValueError("Leader requires initial_only points and NSGA-III batches")
        if self.point_init.use_point_pool:
            raise ValueError("Point initialization cannot use a point pool")


def leader_config(q=4, *, inference_device="cpu", qmc_seed=None):
    """Explicit opt-in; leave the established tidyHEBO defaults unchanged."""
    from ..config import AcquisitionConfig, TidyHEBOConfig

    return TidyHEBOConfig(
        acquisition=AcquisitionConfig(
            q=q,
            _target_="MACE" if q == 1 else "qMACE",
            mc_samples=512,
            quadrature_size=128,
            qmc_seed=qmc_seed,
            evaluation="qmc",
        ),
        search=LayeredSearchConfig(inference_device=inference_device),
    )

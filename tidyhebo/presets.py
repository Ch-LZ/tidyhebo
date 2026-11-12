"""Reusable optimizer configurations from the paper campaigns."""

from dataclasses import replace
from typing import Final

from .config import AcquisitionConfig, BoTorchSearchConfig, ModelConfig, TidyHEBOConfig
from .layered.config import leader_config

tidyHEBO_QMACE: Final[TidyHEBOConfig] = leader_config(q=4)

tidyHEBO_QLOGEI: Final[TidyHEBOConfig] = TidyHEBOConfig(
    acquisition=AcquisitionConfig(
        _target_="qLogEI", q=4, evaluation="qmc", mc_samples=512, quadrature_size=128,
    ),
    search=BoTorchSearchConfig(num_restarts=8),
)
VANILLA_GP_QLOGEI: Final[TidyHEBOConfig] = replace(
    tidyHEBO_QLOGEI,
    model=ModelConfig(input_warp=False, output_transform="standardize", gate=False),
)

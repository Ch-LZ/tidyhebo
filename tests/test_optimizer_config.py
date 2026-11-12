"""The configured objects, not legacy optimizer attributes, control execution."""

from dataclasses import replace
from math import sqrt, pi

import pandas as pd
import pytest
import torch

from tidyhebo import tidyHEBO
from tidyhebo.acquisition import MACE
from tidyhebo.base_model import SimpleModel, CenteredModel, ProfiledModel
from tidyhebo.config import (
    AcquisitionConfig, BoTorchSearchConfig, DiscreteSearchConfig, GASearchConfig,
    ModelConfig, StopConfig, TidyHEBOConfig,
)
from tidyhebo.evolution_optimizer import optimize_real_hedged_acquisition
from tidyhebo.optimizer import train_model_and_AF
from tidyhebo.qmace import optimize_qmace
from tests.test_qmace import Tradeoffs, real_space


@pytest.mark.parametrize("target,cls", [("joint", SimpleModel), ("centered", CenteredModel), ("profiled", ProfiledModel)])
@pytest.mark.parametrize("acquisition", [AcquisitionConfig(evaluation="qmc", mc_samples=16, qmc_seed=37, beta=2.5),
    AcquisitionConfig(_target_="qMACE", q=3, mc_samples=16, qmc_seed=37, beta=2.5)])
def test_model_and_acquisition_receive_configured_choices(monkeypatch, target, cls, acquisition):
    def prepare(self, state_dict=None):
        self.build_gp_mll()
        self.model.eval()
        return {"fit": torch.tensor(1.)}

    monkeypatch.setattr(SimpleModel, "train_model", prepare)
    config = TidyHEBOConfig(model=ModelConfig(_target_=target, output_transform="standardize", input_warp=False),
                           acquisition=acquisition)
    x = torch.tensor([[0., .1], [.3, .6], [.7, .2], [1., 1.]], dtype=torch.double)
    y = torch.tensor([[0.], [1.], [.2], [-.3]], dtype=torch.double)
    model, acq, state = train_model_and_AF(x, y, config=config)
    assert isinstance(model, cls) and model.config is config.model
    assert "fit" in state
    assert isinstance(acq, MACE) and acq.q == acquisition.q
    sampler = acq.qLogEI.sampler
    assert acq.qUCB.beta_prime == pytest.approx(sqrt(acquisition.beta * pi / 2))
    assert sampler.sample_shape == torch.Size([acquisition.mc_samples])
    assert sampler.seed == acquisition.qmc_seed


@pytest.mark.parametrize("q", [1, 2])
def test_search_budget_and_offspring_come_from_shared_config(q):
    config = GASearchConfig(pop_size=96, max_generations=3, n_offsprings=32, reference_partitions=4)
    acquisition, space = Tradeoffs(q), real_space()
    observed = torch.empty((0, 2), dtype=torch.double)
    search = optimize_real_hedged_acquisition if q == 1 else optimize_qmace
    front = search(acquisition, space, observed, config=config)
    assert acquisition.sizes == [96, 32, 32]
    assert len(front) > 0


def test_configs_reject_conflicts_and_unimplemented_paths():
    with pytest.raises(TypeError, match="improvement"):
        AcquisitionConfig(improvement="nei")
    for kwargs in [{"q": 2}, {"_target_": "qMACE", "q": 1}]:
        with pytest.raises(ValueError, match="requires q"):
            AcquisitionConfig(**kwargs)
    unsupported = [
        TidyHEBOConfig(acquisition=AcquisitionConfig(scale="latent")),
        TidyHEBOConfig(acquisition=AcquisitionConfig(posterior="frozen_q1")),
        TidyHEBOConfig(acquisition=AcquisitionConfig(cache_invariants=False)),
        TidyHEBOConfig(acquisition=AcquisitionConfig(_target_="qMACE", q=2, evaluation="quadrature")),
        TidyHEBOConfig(search=GASearchConfig(stop=StopConfig(mode="hv_patience"))),
        TidyHEBOConfig(model=ModelConfig(warp_fit="separate")),
    ]
    for config in unsupported:
        with pytest.raises(NotImplementedError):
            tidyHEBO(real_space().params, n_init=4, config=config)


def test_optimizer_stores_one_config_and_rechecks_replacements():
    config = TidyHEBOConfig(acquisition=AcquisitionConfig(_target_="qMACE", q=2))
    optimizer = tidyHEBO(real_space().params, n_init=4, config=config)
    assert optimizer.config is config
    assert not ({"mc_seed", "matching_distance", "pop_size", "n_generations", "beta", "mc_samples"} & vars(optimizer).keys())
    assert len(optimizer.suggest()) == 2
    optimizer.config = replace(config, return_all=True)
    assert len(optimizer.suggest()) == 2
    optimizer.config = replace(config, search=BoTorchSearchConfig())
    with pytest.raises(ValueError, match="search"):
        optimizer.suggest()

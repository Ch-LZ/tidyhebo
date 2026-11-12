"""Keep the paper configurations usable through the public optimizer API."""

import numpy as np
import pytest
import torch

from tidyhebo import RealP, tidyHEBO
from tidyhebo.presets import (
    VANILLA_GP_QLOGEI,
    tidyHEBO_QLOGEI,
    tidyHEBO_QMACE,
)


@pytest.mark.parametrize(("config", "family", "q", "warped"), [
    (tidyHEBO_QMACE, "qMACE", 4, True),
    (tidyHEBO_QLOGEI, "qLogEI", 4, True),
    (VANILLA_GP_QLOGEI, "qLogEI", 4, False),
], ids=["qMACE", "warped-qLogEI", "vanilla-qLogEI"])
def test_paper_preset_suggest_and_observe(config, family, q, warped):
    acquisition = config.acquisition
    assert (acquisition._target_, acquisition.q, acquisition.evaluation) == (family, q, "qmc")
    assert (acquisition.mc_samples, acquisition.quadrature_size) == (512, 128)
    assert (config.model._target_, config.model.kernel) == ("joint", "Matern52")
    assert config.model.input_warp is warped
    assert config.model.gate is warped
    assert config.model.output_transform == ("yeo_johnson" if warped else "standardize")
    search = config.search
    assert search.inference_device == "cpu"
    if family == "qLogEI":
        assert (search.num_restarts, search.raw_samples, search.maxiter, search.batch_limit) == (8, 128, 100, 4)
        assert search.sequential is True
    else:
        assert search.point_init.directions == search.batch_init.directions == 6
        assert (search.points.population, search.points.pool_size) == (48, 96)
        assert (search.batches.population, search.batches.generations) == (48, 20)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        optimizer = tidyHEBO.from_config(config, [RealP(0, 0.0, 1.0, name="x")], n_init=4)
        for _ in range(4 // q + 1):
            X = optimizer.suggest()
            assert X.shape == (q, 1) and np.isfinite(X.to_numpy()).all()
            optimizer.observe(X, -(X.to_numpy() - 0.35) ** 2)
        assert len(optimizer.train_y) == 4 + q
        assert optimizer.model is not None

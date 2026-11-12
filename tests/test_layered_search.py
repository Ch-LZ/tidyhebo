from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from botorch.models import SingleTaskGP
from tidyhebo import RealP, CategoricalP, tidyHEBO
from tidyhebo.acquisition import MACE
from tidyhebo.candidate_search import CandidateSearch
from tidyhebo.config import AcquisitionConfig
from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.layered import leader_config
from tidyhebo.qmace.ga_search import distinct_points


def small_config(q):
    cfg = leader_config(q)
    s = cfg.search
    s = replace(
        s,
        point_init=replace(s.point_init, directions=2, restarts=2, local_steps=2),
        batch_init=replace(s.batch_init, directions=2, restarts=2, local_steps=2, pool_generations=3),
        points=replace(s.points, population=8, pool_size=8, reference_partitions=1),
        batches=replace(s.batches, population=8, generations=3, reference_partitions=1, local_steps=2),
        scalarization=replace(s.scalarization, calibration_samples=16),
    )
    return replace(
        cfg, search=s, acquisition=replace(cfg.acquisition, mc_samples=16, quadrature_size=16, qmc_seed=19)
    )


def source_and_space(q):
    torch.manual_seed(73)
    np.random.seed(73)
    space = MixedDesignSpace([RealP(0, 0, 1), RealP(1, 0, 1)])
    x = torch.rand(6, 2, dtype=torch.double)
    y = (-(x - 0.4).square().sum(-1)).unsqueeze(-1)
    model = SingleTaskGP(x, y).eval()
    cfg = small_config(q)
    source = MACE(model, cfg.acquisition, y.max())
    return cfg, source, space, x


@pytest.mark.parametrize("q", [1, 2, 4, 8, 16])
def test_complete_layered_search_has_novel_whole_batches_and_full_ga_budget(q, monkeypatch):
    from tidyhebo.qmace.ga_search import BatchProblem

    counts = []
    evaluate = BatchProblem._evaluate

    def counted(self, records, out, **kwargs):
        counts.append(len(records))
        return evaluate(self, records, out, **kwargs)

    monkeypatch.setattr(BatchProblem, "_evaluate", counted)
    cfg, source, space, x = source_and_space(q)
    cfg.check_supported()
    search = CandidateSearch.from_config(cfg.search, design_space=space)
    result = search.search(source, x)
    batches = result[:, None] if q == 1 else result
    assert batches.shape[1:] == (q, 2) and len(batches) > 0
    assert result.dtype == torch.double and result.device.type == "cpu"
    assert space.are_values_new(batches.flatten(0, 1), x).all()
    assert distinct_points(batches, space).all()
    assert (batches >= 0).all() and (batches <= 1).all()
    assert counts[:3] == [8, 8, 8]  # full search budget every generation
    assert len(counts) == 4 and counts[-1] > 0  # final memetic evaluations


def test_public_dataframe_api_and_return_all(monkeypatch):
    cfg, source, space, x = source_and_space(2)
    cfg = replace(cfg, return_all=True)
    optimizer = tidyHEBO(
        space.params,
        train_x=space.from_GA_to_DS(x, return_df=True),
        train_y=np.arange(6)[:, None],
        config=cfg,
    )
    monkeypatch.setattr(optimizer, "_train_and_get_acqf", lambda: (object(), source, {"test": True}))
    all_points = optimizer.suggest()
    assert isinstance(all_points, pd.DataFrame) and all_points.index.names == ["batch", "point"]
    assert len(all_points) % 2 == 0 and not hasattr(optimizer, "search_state")
    optimizer.config = replace(cfg, return_all=False)
    assert len(optimizer.suggest()) == 2

"""L1: build a single-point acquisition and retain its Pareto candidate pool.

L1 optimizes individual points before L2 builds q-point batches from those
points and L3 evolves complete batches. Two operations belong here:

* ``make_point_acquisition`` reuses the fitted GP, best observed value, and
  outcome warp from the batch acquisition. Without a warp it uses analytic
  q=1 MACE; with a warp it uses quadrature. The acquisition factory itself
  may initialize stochastic state; its invocation order must be preserved.
* ``select_point_pool`` retains points from nondominated fronts in their
  original order, optionally traversing deeper fronts to fill the budget.
  It does not sample, score or clean candidates.

The orchestrator runs candidate initialization and feasibility cleanup and
calculates nondominated fronts before calling the second operation. All
coordinates here are normalized GP coordinates; each candidate is a
single-point batch with shape (1, d). No LayeredSearch dependency is needed.
"""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import torch

from ..base_acquisition import BaseAcquisition
from .config import LevelConfig


def make_point_acquisition(source: BaseAcquisition) -> BaseAcquisition:
    """Create L1 MACE for one point, using the existing fitted GP.

    ``source`` is the acquisition prepared for the requested batch size.
    Its ``model``, ``best_f`` and ``outcome_warp`` are reused without fitting
    or copying them. The acquisition config is cloned with target ``MACE``
    and ``q=1``. Evaluation is ``analytic`` when there is no output warp,
    otherwise ``quadrature`` to handle the warped response distribution.

    The result is a q=1 acquisition (three maximization objectives) created
    through ``BaseAcquisition.from_config``. Preserve this factory call and
    its ordering: changing to q=1 QMC/SharedMACE would change the algorithm.
    """
    evaluation = "quadrature" if source.outcome_warp is not None else "analytic"
    point_config = replace(source.config, _target_="MACE", q=1, evaluation=evaluation)
    return BaseAcquisition.from_config(
        point_config, model=source.model, best_f=source.best_f, outcome_warp=source.outcome_warp
    )


def select_point_pool(
    candidates: torch.Tensor,
    fronts: Sequence[np.ndarray],
    level: LevelConfig,
) -> torch.Tensor:
    """Retain the best L1 points without changing their relative order.

    ``candidates`` is a cleaned, feasible and duplicate-free tensor of
    shape ``(N, 1, d)``. ``fronts`` contains ordered arrays of candidate
    indices, obtained by nondominated sorting on acquisition *maximization*
    scores: front zero is best. ``level`` defines ``pool_size`` and
    ``deeper_fronts``; no other fields are used.

    Collect each entire front until the budget is reached when
    ``deeper_fronts`` is enabled, or only the first front otherwise. Then
    truncate to ``pool_size`` and return shape ``(M, 1, d)``. No RNG calls,
    AF evaluations, or filtering occur in this function.
    """
    ids = []
    for front in fronts:
        ids.extend(front.tolist())
        if not level.deeper_fronts or len(ids) >= level.pool_size:
            break
    return candidates[ids[: level.pool_size]]

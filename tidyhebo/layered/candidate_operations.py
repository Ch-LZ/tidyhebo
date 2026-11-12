"""Common candidate operations used by point and whole-batch searches.

All tensors use normalized GP coordinates. Candidate batches are ``(N, q, d)``;
objective values are ``(N, n_obj)`` and are *maximized*. A q-batch represents
an unordered set, so lexicographic canonicalization is required to keep the
QMC estimator invariant under a permutation of its points. The functions
below contain no optimizer state or new random draws. Keeping these numerical
operations separate allows LayeredSearch to own only stage orchestration.

The order of transformations, QMC calls, feasibility checks and sorting is
intentionally unchanged: optimization trajectories and RNG consumption must
match the pre-refactor implementation exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from ..qmace.ga_search import canonicalize

if TYPE_CHECKING:
    from ..design_space import MixedDesignSpace


def snap_to_design_space(x: torch.Tensor, space: MixedDesignSpace) -> torch.Tensor:
    """Round GP-coordinate candidates via the existing GP→GA→GP converters.

    The same conversion is used on new Sobol samples and before canonicalizing
    and filtering candidates. It may snap values to discrete design-space
    boundaries; it must not change device, dtype, or sampling order.
    """
    return space.from_GA_to_GP(space.from_GP_to_GA(x))


def canonical_gp(x: torch.Tensor) -> torch.Tensor:
    """Lexicographically reorder q points using a differentiable gather.

    Sorting from the last coordinate to the first produces a stable
    lexicographic ordering. This prevents reordered batches from receiving
    different finite-QMC/Cholesky evaluations. The operation is piecewise
    differentiable, including inside Adam; it does not consume RNG.
    """
    if x.shape[-2] < 2:
        return x
    for j in range(x.shape[-1] - 1, -1, -1):
        order = torch.argsort(x[..., j], dim=-1, stable=True)
        x = torch.gather(x, -2, order.unsqueeze(-1).expand_as(x))
    return x


def evaluate_in_chunks(
    acq: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    reference: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """Evaluate the same acquisition object for ordered chunks of candidates.

    ``acq`` is the prepared q-point acquisition: ``acq(part)`` returns
    maximization scores ``(chunk_len, n_obj)``. ``reference`` supplies its
    fitted GP's dtype/device; ``chunk`` controls memory use, never sampling.
    Canonicalization runs independently inside each chunk, exactly as before.
    """
    return torch.cat([acq(canonical_gp(part.to(reference))) for part in x.split(chunk)], 0)


def feasible_batches(
    x: torch.Tensor, observed: torch.Tensor, tolerances: torch.Tensor
) -> torch.Tensor:
    """Return an ``(N,)`` mask of feasible, novel, internally distinct q-sets.

    Each coordinate must be finite and in [0,1]. Each point in every batch
    must differ from every observed training point by more than the per-axis
    ``tolerances`` in *at least one* coordinate. Points inside the same batch
    must be distinct by the same criterion. All work stays on the input
    device; detaching candidates avoids recording constraints in Adam's graph.
    """
    x = x.detach()
    valid = torch.isfinite(x).all((-1, -2)) & (x >= 0).all((-1, -2)) & (x <= 1).all((-1, -2))
    if len(observed):
        novel = ((x.unsqueeze(-2) - observed).abs() > tolerances).any(-1).all((-1, -2))
        valid &= novel
    pairwise = ((x.unsqueeze(-2) - x.unsqueeze(-3)).abs() > tolerances).any(-1)
    diagonal = torch.eye(x.shape[-2], dtype=torch.bool, device=x.device)
    return valid & (pairwise | diagonal).all((-1, -2))


def unique_batches(x: torch.Tensor, space: MixedDesignSpace) -> torch.Tensor:
    """Keep first occurrence of each unordered set, preserving candidate order.

    ``space`` is the design-space converter from GP to GA coordinates. Identity
    uses existing ``canonicalize`` bytes of that GA representation. The
    original candidate tensors and their gradients are not changed; only
    indexing into the retained rows is performed.
    """
    if not len(x):
        return x
    ga = space.from_GP_to_GA(x.detach()).cpu().numpy()
    seen, keep = set(), []
    for i, b in enumerate(ga):
        key = canonicalize(b).tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return x[keep]


def rank_scores(scores: np.ndarray) -> list[np.ndarray]:
    """Sort maximization AF scores into nondominated fronts, best first.

    Input shape is ``(N, n_obj)``. Pymoo expects minimization, hence the
    sign reversal. No tiebreaking, random seeding or front truncation is added.
    """
    return NonDominatedSorting().do(-scores)


def refinement_targets(
    scores: torch.Tensor,
    directions: np.ndarray,
    scalar: Callable[[torch.Tensor, np.ndarray], torch.Tensor],
    fraction: float
) -> tuple[list[int], list[np.ndarray]]:
    """Select unique final-generation representatives for memetic Adam.

    ``scores`` are the already-evaluated acquisition objectives (maximized).
    Each reference direction yields one scalarized minimizer via ``scalar``.
    A selected candidate appears at most once, with the first direction that
    chose it. The selection is truncated to ceil(N*fraction), at least one.
    No sampling or fitness re-evaluation occurs here.
    """
    n = max(1, math.ceil(len(scores) * fraction))
    selected, rays = [], []
    for ray in directions:
        index = int(scalar(scores, ray).argmin())
        if index not in selected:
            selected.append(index)
            rays.append(ray)
    return selected[:n], rays[:n]

"""Initialize feasible L1 point and L3 batch populations in normalized GP coordinates.

Both levels use the same algorithm: draw/clean Sobol batches, optimize MACE
anchor and scalarization targets, append any supplied L2 batches, and refill
with fresh Sobol batches. Only the supplied callbacks consume randomness.
The original order of callback invocations is preserved.
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch

from .config import InitializationConfig


TargetOptimizer = Callable[[torch.Tensor, np.ndarray, Sequence[int], int, int], torch.Tensor]


def _target_specifications(
    config: InitializationConfig, reference_directions: np.ndarray
) -> tuple[list[np.ndarray], list[int]]:
    """Build MACE anchor targets (0,1,2) followed by scalarized targets (-1).

    A target consists of an acquisition objective index and a reference
    direction used by the shared target optimizer. This function uses no RNG.
    """
    rays: list[np.ndarray] = []
    objectives: list[int] = []
    if config.anchors:
        rays.extend([reference_directions[0]] * 3)
        objectives.extend(range(3))
    if config.directions:
        count = (
            len(reference_directions)
            if config.directions == -1
            else min(config.directions, len(reference_directions))
        )
        rays.extend(
            reference_directions[i]
            for i in np.linspace(0, len(reference_directions) - 1, count, dtype=int)
        )
        objectives.extend([-1] * count)
    return rays, objectives


def _refill_with_sobol(
    seeds: torch.Tensor,
    population: int,
    sobol: Callable[[int], torch.Tensor],
    clean: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Refill with at most 40 Sobol draws; preserve cleanup and draw ordering.

    Returns exactly ``population`` batches or raises ``RuntimeError`` when
    the configured feasibility constraints make that impossible.
    """
    for _ in range(40):
        if len(seeds) >= population:
            break
        seeds = clean(torch.cat((seeds, sobol(max(population, population - len(seeds))))))[:population]
    if len(seeds) < population:
        raise RuntimeError(f"Only {len(seeds)} feasible batches for population={population}")
    return seeds


def initialize_population(
    config: InitializationConfig,
    *,
    population: int,
    reference_directions: np.ndarray,
    sobol: Callable[[int], torch.Tensor],
    clean: Callable[[torch.Tensor], torch.Tensor],
    optimize_targets: TargetOptimizer,
    supplied: torch.Tensor | None = None,
) -> torch.Tensor:
    """Produce a feasible initial population of shape ``(population, q, d)``.

    ``config`` gives restarts/Adam steps, MACE anchors and direction count;
    ``reference_directions`` is an ordered ``(R, n_obj)`` weight array.
    ``sobol(n)`` produces ``(n,q,d)`` GP-coordinate batches; ``clean`` filters
    infeasible/duplicate batches, preserving order. ``optimize_targets`` accepts
    ``(raw, rays, objective_indices, restarts, local_steps)`` and returns one
    ``(q,d)`` batch per target. ``supplied`` adds optional L2 candidates after
    optimization but before cleanup/refill.

    The original algorithm's Sobol, Adam, filter and refill order is preserved.
    The function has no hidden state and consumes no RNG of its own.
    """
    raw = clean(sobol(max(population, config.restarts * 8)))
    if not len(raw):
        raise RuntimeError("No feasible Sobol seeds")

    rays, objectives = _target_specifications(config, reference_directions)
    parts: list[torch.Tensor] = []
    if rays:
        parts.append(
            optimize_targets(raw, np.asarray(rays), objectives, config.restarts, config.local_steps)
        )
    if supplied is not None and len(supplied):
        parts.append(supplied)

    seeds = clean(torch.cat(parts))[:population] if parts else raw[:0]
    return _refill_with_sobol(seeds, population, sobol, clean)

"""NSGA-III search whose individuals are complete, unordered q-batches."""

import numpy as np
import torch
from pymoo.algorithms.moo.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from torch.quasirandom import SobolEngine
from pymoo.util.ref_dirs import get_reference_directions

from ..archive import ParetoAccumulator
from ..config import GASearchConfig
from ..design_space import MixedDesignSpace
from ..base_acquisition import BaseAcquisition
from ..evolution_optimizer import BatchLayout, _pymoo_seed_options


def canonicalize(batch: np.ndarray) -> np.ndarray:
    """Sort points lexicographically without changing a batch's membership."""
    return batch[np.lexsort(batch.T[::-1])]


def distinct_points(batches: torch.Tensor, space: MixedDesignSpace) -> torch.Tensor:
    """One mask value per batch: all point pairs differ beyond GA tolerances."""
    q = batches.shape[1]
    delta = (batches[:, :, None, :] - batches[:, None, :, :]).abs()
    distinct = (delta > space.tolerances.squeeze(-1).to(batches)).any(dim=-1)
    diagonal = torch.eye(q, dtype=torch.bool, device=batches.device)
    return (distinct | diagonal).all(dim=-1).all(dim=-1)


class BatchSampler(Sampling):
    """Group one scrambled d-dimensional Sobol sequence into canonical q-sets."""

    def __init__(self, space, layout):
        super().__init__()
        self.space, self.layout = space, layout

    def _do(self, problem, n_samples, **kwargs):
        q, d = self.layout.q, self.layout.d
        gp = SobolEngine(d, scramble=True).draw(n_samples * q, dtype=torch.double).reshape(n_samples, q, d)
        batches = self.space.from_GP_to_GA(gp).numpy()
        return [self.layout.to_records(canonicalize(batch)) for batch in batches]


class BatchProblem(Problem):
    """A batch is feasible iff all points are novel and pairwise distinct.

    evaluator optionally supplies canonical/chunked AF evaluation without
    wrapping the acquisition object; it receives (n, q, d) GP coordinates.
    """

    def __init__(self, acqf, space, layout, train_x, *, evaluator=None):
        super().__init__(vars=layout.vars, n_obj=acqf.n_obj, n_ieq_constr=1)
        self.acqf, self.space, self.layout, self.train_x = acqf, space, layout, train_x
        self.evaluate_acquisition = acqf if evaluator is None else evaluator

    def _evaluate(self, records, out, *args, **kwargs):
        batches = torch.from_numpy(self.layout.from_records(records))
        novel = self.space.are_values_new(batches.flatten(0, 1), self.train_x).reshape(batches.shape[:2]).all(dim=1)
        feasible = novel & distinct_points(batches, self.space)
        with torch.no_grad():
            out["F"] = -self.evaluate_acquisition(self.space.from_GA_to_GP(batches)).cpu().numpy()
        out["G"] = (0.5 - feasible.to(torch.double)).numpy()


def optimize_qmace(acqf: BaseAcquisition, space: MixedDesignSpace, train_x: torch.Tensor,
                   *, config: GASearchConfig) -> torch.Tensor:
    """Return (n_front, q, d) batches from post-survival populations.

    Flatten only for archive storage: one archive row still represents one
    complete batch and its three AF values. Different batches with equal AF
    values survive. No generation history or second archive is introduced.
    """
    # Operators use the batch helpers above; load them at the search entry point.
    from .ga_components import BatchMatcher, SetDuplicates, SetMating

    layout = BatchLayout(space, acqf.q)
    duplicates = SetDuplicates(layout)
    mating = SetMating(layout, BatchMatcher(space, acqf.model, config.matching_distance),
                       selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                       eliminate_duplicates=duplicates)
    algorithm = NSGA3(pop_size=config.pop_size, n_offsprings=config.n_offsprings,
                      sampling=BatchSampler(space, layout), mating=mating,
                      eliminate_duplicates=duplicates,
                      ref_dirs=get_reference_directions("das-dennis", acqf.n_obj, n_partitions=config.reference_partitions))
    archive = ParetoAccumulator()

    def save_front(algorithm):
        population = algorithm.pop
        batches = layout.from_records(population.get("X"))
        archive.add(batches.reshape(len(batches), -1), population.get("F"), population.get("FEAS"))

    minimize(BatchProblem(acqf, space, layout, train_x), algorithm, ("n_gen", config.max_generations),
             copy_algorithm=False, copy_termination=False, save_history=False, callback=save_front,
             **_pymoo_seed_options(algorithm))
    if not archive.has_data:
        return torch.empty((0, layout.q, layout.d), dtype=torch.double)
    batches, _ = archive.take_front()
    batches = batches.reshape(-1, layout.q, layout.d)
    return batches[torch.randperm(len(batches))]

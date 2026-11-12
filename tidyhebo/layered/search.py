"""Orchestrate L1 point search, L2 index-pool GA and L3 whole-batch NSGA-III.

``LayeredSearch`` is the public candidate-search adapter registered under
``_target_='layered'``. It prepares the fitted acquisition, feasibility data,
calibrated scalarization and sampling for each level, then runs the three
stages in order. Shared deterministic candidate math lives in
``candidate_operations``; common population construction lives in
``population_initialization``. The remaining methods either own prepared
level state or supply state-bound callbacks to the three search stages.
"""

import numpy as np
import torch
from pymoo.util.ref_dirs import get_reference_directions
from ..candidate_search import CandidateSearch
from ..evolution_optimizer import BatchLayout
from ..mixed_params import RealParam
from ..qmace.ga_search import BatchProblem
from ..qmace.ga_components import BatchMatcher
from .scalarization import Scalarizer
from .candidate_operations import (
    canonical_gp,
    evaluate_in_chunks,
    feasible_batches,
    rank_scores,
    refinement_targets,
    snap_to_design_space,
    unique_batches,
)
from .local import refine, optimize_targets
from .inference import resolve_device, inference_copy
from .pool_batches_l2 import PoolProblem, solve_pool_problem
from .batch_search_l3 import BatchSearchL3
from .point_search_l1 import make_point_acquisition, select_point_pool
from .population_initialization import initialize_population


class LayeredSearch(CandidateSearch, _target_="layered"):
    """Run point initialization, pool-index GA and whole-batch NSGA-III.

    One instance owns one search invocation's working population and frozen
    scalarization at a time. Point and batch levels have independent scales.
    No populations, timing data or diagnostics are passed to the optimizer.
    """

    def __init__(self, design_space, config):
        super().__init__(design_space, config)
        resolve_device(config.inference_device)
        if not all(isinstance(p, RealParam) for p in design_space.params):
            raise ValueError("Layered search requires continuous parameters")

    def search(self, acqf, train_x):
        """Return the final feasible Pareto front in GA coordinates."""
        matcher_model = acqf.model
        source = inference_copy(acqf, resolve_device(self.config.inference_device))
        point_acq = make_point_acquisition(source)
        self._prepare_level(point_acq, train_x, self.config.points)
        point_candidates = self.clean(
            initialize_population(
                self.config.point_init,
                population=self.level.population,
                reference_directions=self.directions,
                sobol=self.sobol,
                clean=self.clean,
                optimize_targets=self._optimize_targets,
            )
        )
        point_pool = select_point_pool(
            point_candidates, self.rank(point_candidates), self.config.points
        )
        self._prepare_level(point_acq if source.q == 1 else source, train_x, self.config.batches)
        self.matcher_model = matcher_model
        self.matching_distance = self.config.matching_distance
        supplied = self.small_batches_ga(point_pool) if self.config.batch_init.use_point_pool else None
        initial = initialize_population(
            self.config.batch_init,
            population=self.level.population,
            reference_directions=self.directions,
            sobol=self.sobol,
            clean=self.clean,
            optimize_targets=self._optimize_targets,
            supplied=supplied,
        )
        result = self.clean(self.batches_ga(initial))
        if not len(result):
            raise RuntimeError("Layered search returned no feasible batches")
        result = self.space.from_GP_to_GA(result[self.rank(result)[0]]).detach().cpu()
        return result[:, 0] if source.q == 1 else result

    def _prepare_level(self, acq, train_x, level):
        """Freeze the AF scale and feasibility data for this point/batch level."""
        space = self.design_space
        scalar_config = self.config.scalarization
        chunk, progress_every = self.config.evaluation_chunk, self.config.progress_every
        self.acq, self.space, self.train_x, self.level = acq, space, train_x, level
        self.q, self.d, self.chunk = acq.q, len(space.params), chunk
        self.ref = acq.model.train_inputs[0]
        self.observed = space.from_GA_to_GP(train_x.to(self.ref))
        self.tolerances = space.tolerances.squeeze(-1).to(self.ref)
        self.directions = (
            get_reference_directions("das-dennis", acq.n_obj, n_partitions=level.reference_partitions)
            if acq.n_obj > 1
            else np.ones((1, 1))
        )
        self.progress_every = progress_every
        self.continuous = torch.tensor(
            [isinstance(p, RealParam) for p in space.params], device=self.ref.device
        )
        samples = self.sobol(scalar_config.calibration_samples)
        self.scalar = Scalarizer(self.scores(samples).detach(), scalar_config)

    def log(self, message):
        print(f"AF q={self.q} {message}", flush=True)

    def sobol(self, n):
        x = torch.quasirandom.SobolEngine(self.d * self.q, scramble=True).draw(n, dtype=self.ref.dtype)
        x = x.to(self.ref).reshape(n, self.q, self.d)
        return snap_to_design_space(x, self.space)

    def scores(self, x):
        """Evaluate MACE objectives for candidate batches in fixed-size chunks."""
        return evaluate_in_chunks(self.acq, x, self.ref, self.chunk)

    def _refine(self, starts, rays, steps):
        return refine(
            starts,
            rays,
            steps,
            scalar=self.scalar,
            scores=self.scores,
            feasible=self.feasible,
            continuous=self.continuous,
        )

    def _optimize_targets(self, raw, rays, objectives, restarts, steps):
        return optimize_targets(
            raw,
            rays,
            objectives,
            restarts,
            steps,
            scalar=self.scalar,
            scores=self.scores,
            feasible=self.feasible,
            continuous=self.continuous,
        )

    def feasible(self, x):
        """Check bounds, novelty and pairwise distinctness for GP-coordinate batches."""
        return feasible_batches(x, self.observed, self.tolerances)

    def clean(self, x):
        x = canonical_gp(snap_to_design_space(x.detach(), self.space))
        return unique_batches(x[self.feasible(x)], self.space)

    def rank(self, x):
        """Rank candidates by nondominated fronts (maximization objectives)."""
        with torch.no_grad():
            scores = self.scores(x).detach().cpu().numpy()
        return rank_scores(scores)

    def refine_population(self, x):
        """Adam-refine one representative per scalarization ray from the last front."""
        with torch.no_grad():
            values = self.scores(x)
            selected, rays = refinement_targets(
                values, self.directions, self.scalar, self.level.refine_fraction
            )
        return self._refine(x[selected], np.asarray(rays), self.level.local_steps)

    def batches_ga(self, initial):
        """Run L3 NSGA-III on whole q-batches, with optional end refinement."""
        if self.level.algorithm == "initial_only":
            return initial
        layout = BatchLayout(self.space, self.q)
        problem = BatchProblem(self.acq, self.space, layout, self.train_x.cpu(), evaluator=self.scores)
        matcher = BatchMatcher(self.space, self.matcher_model, self.matching_distance)
        result = BatchSearchL3(
            problem, matcher, level=self.level, directions=self.directions, reference=self.ref
        ).solve(
            initial,
            refine=lambda gp: self.clean(self.refine_population(gp)),
            report=self.log,
            progress_every=self.progress_every,
        )
        return self.clean(result)

    def small_batches_ga(self, point_pool):
        """Build a feasible q-batch from the point-level Pareto pool (L2)."""
        pool = point_pool.squeeze(1)
        if len(pool) < self.q:
            self.log(f"point pool has {len(pool)} < q={self.q}; extending with Sobol points")
            extra = self.sobol(self.q * 2).reshape(-1, self.d)
            pool = torch.unique(torch.cat((pool, extra)), dim=0)

        problem = PoolProblem(
            pool, q=self.q, n_obj=self.acq.n_obj,
            acquisition_scores=self.scores, feasible=self.feasible,
        )
        batches = solve_pool_problem(
            problem,
            population=self.level.population,
            generations=self.config.batch_init.pool_generations,
            ref_dirs=self.directions,
        )
        return self.clean(batches)

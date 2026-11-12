"""L3: evolve full q-point batches with NSGA-III and optional final memetic step.

Unlike L2, each individual here encodes an entire q-by-d batch (pymoo's
mixed-variable records). The fixed ``BatchProblem`` evaluates these batches
with the acquisition function and enforces novelty/distinctness. Parents are
aligned by ``BatchMatcher`` before crossover and mutation; reordered batches
are treated as duplicates. The final generation can be refined by Adam, then
reinserted through pymoo's usual evaluation and survival path.

The library's ``Algorithm.run`` owns generation stepping; a read-only pymoo
``Callback`` reports progress. Only the final memetic pass runs afterward,
so logging reflects its surviving population without modifying evolution.

``BatchSearchL3`` owns the batch-level search, not the GP, point initialization,
layer config tree, or the public optimizer. All needed dependencies are typed
and supplied explicitly; no reference to LayeredSearch is retained.
"""

from collections.abc import Callable

import numpy as np
import torch
from pymoo.algorithms.moo.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.operators.selection.tournament import TournamentSelection

from ..qmace.ga_search import BatchProblem
from ..qmace.ga_components import BatchMatcher, SetDuplicates, SetMating
from .config import LevelConfig


class _GenerationProgress(Callback):
    """Report NSGA-III evaluations without changing the evolving population.

    Pymoo calls ``notify`` after each completed generation, before incrementing
    ``n_gen``. The last report is deferred until final memetic refinement has
    passed through survival; its ``new_evals`` still counts *only* evolution,
    as in the original loop. No RNG or fitness calls occur in this callback.
    """

    def __init__(self, generations: int, every: int, report: Callable[[str], None]) -> None:
        super().__init__()
        self.generations = generations
        self.every = every
        self.report = report
        self.previous_evals = 0
        self.last_new_evals = 0

    def notify(self, algorithm: NSGA3) -> None:
        generation = algorithm.n_gen
        count = algorithm.evaluator.n_eval
        self.last_new_evals = count - self.previous_evals
        self.previous_evals = count
        if generation < self.generations and (generation == 1 or generation % self.every == 0):
            self._report(algorithm, generation, self.last_new_evals)

    def finish(self, algorithm: NSGA3) -> None:
        """Report the final population *after* optional memetic survival."""
        self._report(algorithm, self.generations, self.last_new_evals)

    def _report(self, algorithm: NSGA3, generation: int, search_evals: int) -> None:
        self.report(
            f"NSGA-III generation={generation}/{self.generations} "
            f"pop={len(algorithm.pop)} new_evals={search_evals}"
        )


class BatchSearchL3:
    """Run one whole-batch search using a fixed problem, geometry and budget.

    ``problem`` carries the acquisition evaluator, design space, batch layout
    and training points. ``matcher`` aligns parent q-sets in the chosen
    geometry (kernel or Euclidean). ``level`` sets population size, generation
    budget and optional final memetic refinement; ``directions`` supplies the
    NSGA-III reference vectors. ``reference`` fixes the dtype/device of
    decoded batches to match the fitted GP.

    ``solve`` receives two narrow callbacks: ``refine`` maps final-population
    GP batches to improved feasible GP batches; ``report`` prints progress.
    It returns the final population in normalized GP coordinates, leaving
    ordinary feasibility cleanup to the caller. No random generators are
    created or reset here; pymoo receives one NumPy-derived seed.
    """

    def __init__(
        self,
        problem: BatchProblem,
        matcher: BatchMatcher,
        *,
        level: LevelConfig,
        directions: np.ndarray,
        reference: torch.Tensor,
    ) -> None:
        self.problem = problem
        self.matcher = matcher
        self.level = level
        self.directions = directions
        self.reference = reference

    def solve(
        self,
        initial: torch.Tensor,
        *,
        refine: Callable[[torch.Tensor], torch.Tensor],
        report: Callable[[str], None],
        progress_every: int,
    ) -> torch.Tensor:
        """Evolve ``initial`` (N,q,d), optionally refine last gen, return GP batches.

        The ``refine`` callback is only called when ``level.memetic == 'end'``
        at the last generation. It must return cleaned feasible GP batches.
        ``report`` receives the same per-generation text as before, at gen 1,
        multiples of ``progress_every``, and the last generation.

        RNG sequence remains: construct mating/GA, draw one NumPy seed,
        then let pymoo.run() call next() exactly ``level.generations`` times.
        The last population is refined only after pymoo's final next(), just
        as before. Progress reports contain the same generation counts.
        """
        level = self.level
        problem = self.problem
        layout, space = problem.layout, problem.space
        duplicates = SetDuplicates(layout)
        mating = SetMating(
            layout,
            self.matcher,
            selection=TournamentSelection(func_comp=comp_by_cv_then_random),
            eliminate_duplicates=duplicates,
        )
        ga = space.from_GP_to_GA(initial).detach().cpu().numpy()
        progress = _GenerationProgress(level.generations, progress_every, report)
        algorithm = NSGA3(
            pop_size=level.population,
            n_offsprings=level.population,
            sampling=Population.new("X", layout.to_records(ga)),
            ref_dirs=self.directions,
            mating=mating,
            eliminate_duplicates=duplicates,
            callback=progress,
        )
        algorithm.setup(
            problem, termination=("n_gen", level.generations), seed=int(np.random.randint(2**31 - 1))
        )
        algorithm.run()
        if level.memetic == "end":
            self._refine_final_population(algorithm, refine)
        progress.finish(algorithm)
        return self._decode(algorithm.pop)

    def _decode(self, population: Population) -> torch.Tensor:
        """Convert pymoo's mixed-variable records into normalized GP batches."""
        space, layout = self.problem.space, self.problem.layout
        return space.from_GA_to_GP(
            torch.as_tensor(layout.from_records(population.get("X"))).to(self.reference)
        )

    def _refine_final_population(
        self, algorithm: NSGA3, refine: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """Inject refined last-generation batches via pymoo's evaluator/survival.

        ``refine`` returns feasible GP-coordinate batches. The procedure
        deliberately uses the original survival and optimum update, without
        mutating pymoo state from its progress callback.
        """
        problem = self.problem
        refined = refine(self._decode(algorithm.pop))
        if not len(refined):
            return
        extra = Population.new(
            "X", problem.layout.to_records(problem.space.from_GP_to_GA(refined).cpu().numpy())
        )
        algorithm.evaluator.eval(problem, extra)
        # New pymoo versions use a per-algorithm generator for survival.
        state = getattr(algorithm, "random_state", None)
        rng_options = {} if state is None else {"random_state": state}
        algorithm.pop = algorithm.survival.do(
            problem,
            Population.merge(algorithm.pop, extra),
            n_survive=self.level.population,
            algorithm=algorithm,
            **rng_options,
        )
        algorithm._set_optimum()

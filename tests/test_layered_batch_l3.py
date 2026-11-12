"""L3 refactor regression: original loop versus extracted NSGA-III stage.

The pymoo double exercises the same constructor/setup/generation/refinement
sequence without a costly real acquisition function. Full optimization tests
still run against actual pymoo in test_layered_search.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tidyhebo.layered import batch_search_l3
from tidyhebo.layered import search as search_module
from tidyhebo.layered.search import LayeredSearch


class _Layout:
    def __init__(self, space, q):
        self.q = q
        self.space = space

    def to_records(self, x):
        return np.asarray(x)

    def from_records(self, x):
        return np.asarray(x)


class _Space:
    def from_GP_to_GA(self, x):
        return x

    def from_GA_to_GP(self, x):
        return x


class _Problem:
    def __init__(self, acq, space, layout, train_x, *, evaluator):
        self.acqf = acq
        self.space = space
        self.layout = layout
        self.train_x = train_x
        self.evaluator = evaluator


class _Population:
    def __init__(self, x):
        self.x = np.array(x, copy=True)

    def get(self, key):
        assert key == "X"
        return self.x

    def __len__(self):
        return len(self.x)

    @classmethod
    def new(cls, key, x):
        assert key == "X"
        return cls(x)

    @classmethod
    def merge(cls, left, right):
        return cls(np.concatenate((left.x, right.x)))


class _Evaluator:
    def __init__(self):
        self.n_eval = 0

    def eval(self, problem, population):
        self.n_eval += len(population)


class _Survival:
    def do(self, problem, population, *, n_survive, algorithm):
        return _Population(population.x[-n_survive:])


class _Algorithm:
    def __init__(self, *, pop_size, n_offsprings, sampling, ref_dirs, mating,
                 eliminate_duplicates, callback=None):
        self.pop = sampling
        self.evaluator = _Evaluator()
        self.survival = _Survival()
        self.pop_size = pop_size
        self.n_offsprings = n_offsprings
        self.optimum_updated = False
        self.callback = callback
        self.n_gen = 0
        self.steps_done = 0

    def setup(self, problem, *, termination, seed):
        assert termination[0] == "n_gen"
        self.problem = problem
        self.seed = seed
        self.max_generations = termination[1]
        return self

    def run(self):
        """Pymoo-shaped run/next/callback lifecycle, including last callback."""
        while self.steps_done < self.max_generations:
            self.next()
            self.steps_done += 1
            self.n_gen = self.steps_done
            self.callback(self)
        return None

    def next(self):
        order = np.random.permutation(len(self.pop))
        jitter = torch.rand(1).item() * 0.00001
        self.pop = _Population(self.pop.x[order] + jitter)
        self.evaluator.n_eval += self.n_offsprings

    def _set_optimum(self):
        self.optimum_updated = True


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


class _Search:
    def __init__(self, q, memetic, *, generations=4, empty_refinement=False):
        self.q = q
        self.empty_refinement = empty_refinement
        self.level = SimpleNamespace(
            algorithm="nsga3", population=5, generations=generations, memetic=memetic
        )
        self.space = _Space()
        self.acq = object()
        self.train_x = torch.zeros((2, 2), dtype=torch.double)
        self.matcher_model = object()
        self.matching_distance = "kernel"
        self.directions = np.ones((2, 3))
        self.ref = torch.zeros((1, 2), dtype=torch.double)
        self.progress_every = 2
        self.messages = []
        self.clean_count = 0
        self.refine_count = 0

    def scores(self, x):
        return x.sum(dim=(-1, -2)).unsqueeze(-1)

    def clean(self, x):
        self.clean_count += 1
        return x.clone()

    def refine_population(self, x):
        self.refine_count += 1
        return x[:0] if self.empty_refinement else x[:2] + 0.001

    def log(self, message):
        self.messages.append(message)


def _legacy_batches_ga(search, initial):
    """Frozen pre-extraction method; pymoo dependencies are patched doubles."""
    if search.level.algorithm == "initial_only":
        return initial
    level = search.level
    layout = _Layout(search.space, search.q)
    duplicates = _Dummy(layout)
    mating = _Dummy(
        layout,
        _Dummy(search.space, search.matcher_model, search.matching_distance),
        selection=_Dummy(func_comp=None),
        eliminate_duplicates=duplicates,
    )
    ga = search.space.from_GP_to_GA(initial).detach().cpu().numpy()
    algorithm = _Algorithm(
        pop_size=level.population,
        n_offsprings=level.population,
        sampling=_Population.new("X", layout.to_records(ga)),
        ref_dirs=search.directions,
        mating=mating,
        eliminate_duplicates=duplicates,
    )
    problem = _Problem(search.acq, search.space, layout, search.train_x.cpu(), evaluator=search.scores)
    algorithm.setup(
        problem, termination=("n_gen", level.generations), seed=int(np.random.randint(2**31 - 1))
    )
    for generation in range(1, level.generations + 1):
        before = algorithm.evaluator.n_eval
        algorithm.next()
        search_evals = algorithm.evaluator.n_eval - before
        if level.memetic == "end" and generation == level.generations:
            gp = search.space.from_GA_to_GP(
                torch.as_tensor(layout.from_records(algorithm.pop.get("X"))).to(search.ref)
            )
            refined = search.clean(search.refine_population(gp))
            if len(refined):
                extra = _Population.new(
                    "X", layout.to_records(search.space.from_GP_to_GA(refined).cpu().numpy())
                )
                algorithm.evaluator.eval(problem, extra)
                algorithm.pop = algorithm.survival.do(
                    problem,
                    _Population.merge(algorithm.pop, extra),
                    n_survive=level.population,
                    algorithm=algorithm,
                )
                algorithm._set_optimum()
        if generation == 1 or generation % search.progress_every == 0 or generation == level.generations:
            search.log(
                f"NSGA-III generation={generation}/{level.generations} pop={len(algorithm.pop)} new_evals={search_evals}"
            )
    gp = search.space.from_GA_to_GP(
        torch.as_tensor(layout.from_records(algorithm.pop.get("X"))).to(search.ref)
    )
    return search.clean(gp)


def _np_state_equal(a, b):
    return all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y
               for x, y in zip(a, b))


@pytest.mark.parametrize("q,memetic,generations,empty_refinement", [
    (1, "none", 1, False),
    (1, "end", 1, False),
    (1, "none", 4, False),
    (2, "end", 4, False),
    (2, "end", 4, True),
    (4, "end", 4, False),
    (4, "none", 4, False),
])
def test_l3_matches_previous_result_rng_progress_and_refinement(
    monkeypatch, q, memetic, generations, empty_refinement
):
    monkeypatch.setattr(search_module, "BatchLayout", _Layout)
    monkeypatch.setattr(search_module, "BatchProblem", _Problem)
    monkeypatch.setattr(search_module, "BatchMatcher", _Dummy)
    monkeypatch.setattr(batch_search_l3, "SetDuplicates", _Dummy)
    monkeypatch.setattr(batch_search_l3, "SetMating", _Dummy)
    monkeypatch.setattr(batch_search_l3, "TournamentSelection", _Dummy)
    monkeypatch.setattr(batch_search_l3, "Population", _Population)
    monkeypatch.setattr(batch_search_l3, "NSGA3", _Algorithm)

    initial = torch.arange(5 * q * 2, dtype=torch.double).reshape(5, q, 2) / 100

    def replay(run):
        np.random.seed(1729 + q)
        torch.manual_seed(239 + q)
        search = _Search(q, memetic, generations=generations, empty_refinement=empty_refinement)
        result = run(search, initial)
        return (result, np.random.get_state(), torch.random.get_rng_state(),
                search.messages, search.clean_count, search.refine_count)

    before = replay(_legacy_batches_ga)
    after = replay(LayeredSearch.batches_ga)
    assert torch.equal(before[0], after[0])
    assert _np_state_equal(before[1], after[1])
    assert torch.equal(before[2], after[2])
    assert before[3:] == after[3:]
    assert after[5] == (1 if memetic == "end" else 0)


def test_l3_initial_only_does_not_touch_rng():
    search = _Search(q=2, memetic="none")
    search.level.algorithm = "initial_only"
    initial = torch.zeros((5, 2, 2), dtype=torch.double)
    assert LayeredSearch.batches_ga(search, initial) is initial

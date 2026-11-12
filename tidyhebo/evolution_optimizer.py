from copy import deepcopy
from functools import singledispatchmethod

import numpy as np
import torch

from torch.quasirandom import SobolEngine

from pymoo.algorithms.moo.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from .base_acquisition import BaseAcquisition
from .archive import ParetoAccumulator
from .config import GASearchConfig
from .common import GA_Tensor, GP_Tensor
from .design_space import MixedDesignSpace

from numpy.typing import NDArray


def _pymoo_seed_options(algorithm):
    """Seed pymoo's local generator while preserving the older global-RNG path."""
    return {"seed": int(np.random.randint(2**31 - 1))} if hasattr(algorithm, "random_state") else {}


# region Genetic Algorithm
class BatchLayout:
    """Encode a (q, d) numeric set as pymoo's dictionary of scalar variables.

    Single points retain the parameter names. Joint batches add slot and
    parameter indices to avoid collisions. Each slot retains its variable type.
    """

    def __init__(self, space: MixedDesignSpace, q: int):
        self.q, self.d = q, len(space.params)
        self.names = ([space.ordered_params] if q == 1 else
                      [[f"__q{i}__p{j}__{p.name}" for j, p in enumerate(space.params)] for i in range(q)])
        self._flat_names = [name for slot in self.names for name in slot]
        self.vars = {self.names[i][j]: deepcopy(p.pymoo_var) for i in range(q) for j, p in enumerate(space.params)}

    def to_records(self, batches: np.ndarray) -> dict | list[dict]:
        """GA coordinates -> pymoo record(s): (q, d) -> dict, (n, q, d) -> list."""
        if batches.ndim not in (2, 3) or batches.shape[-2:] != (self.q, self.d):
            raise ValueError(f"Expected ({self.q}, {self.d}) or (n, {self.q}, {self.d}) GA coordinates")
        if batches.ndim == 2:
            return dict(zip(self._flat_names, batches.flat))
        return [dict(zip(self._flat_names, batch.flat)) for batch in batches]

    @singledispatchmethod
    def from_records(self, records) -> np.ndarray:
        """Pymoo records -> GA coordinates: collection -> (n, q, d), dict -> (q, d)."""
        return np.asarray([[record[name] for name in self._flat_names] for record in records],
                          dtype=np.float64).reshape(len(records), self.q, self.d)

    @from_records.register(dict)
    def _from_record(self, record: dict) -> np.ndarray:
        return np.asarray([record[name] for name in self._flat_names], dtype=np.float64).reshape(self.q, self.d)


class MixedProblem(Problem):
    def __init__(self, acqf_with_state: BaseAcquisition,
                layout: BatchLayout,
                opt_space: MixedDesignSpace,
                n_obj: int,
                x_train: GA_Tensor
                ):
        super().__init__(vars=layout.vars, n_obj=n_obj,
                         n_ieq_constr=1
                        )
        self.acqf_with_state = acqf_with_state
        self.opt_space = opt_space
        self.train_x: GA_Tensor = x_train
        self.layout = layout


    def _evaluate(self, x: NDArray, out: dict, *args, **kwargs) -> NDArray:
        xt = self.layout.from_records(x)
        torch_x_opt_ga: GA_Tensor = torch.from_numpy(xt)
        torch_x_gp_space: GP_Tensor = self.opt_space.from_GA_to_GP(torch_x_opt_ga) # shape required by acquisition function
        # optimize is minimization problem => -1 * f(x)
        with torch.no_grad():  # a bit faster
            torch_y = -1 * self.acqf_with_state(torch_x_gp_space)
            torch_constrain = self.opt_space.are_values_new(torch_x_opt_ga.flatten(0, 1), self.train_x) * (-1) + 0.5
        out["F"] = torch_y.detach().numpy()
        out["G"] = torch_constrain.detach().numpy()


class SobolMixedSampler(Sampling):
    def __init__(self, opt_space: MixedDesignSpace, layout: BatchLayout):
        super().__init__()
        self.opt_space: MixedDesignSpace = opt_space
        self.layout = layout


    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var
        sobol_sample: GP_Tensor = SobolEngine(n_var, scramble=True).draw(n_samples, dtype=torch.double)
        # rounding preserves low-discrepancy property.
        rounded_sample: GA_Tensor = self.opt_space.from_GP_to_GA(sobol_sample).detach().numpy()
        batches = rounded_sample.reshape(n_samples, self.layout.q, self.layout.d)
        return self.layout.to_records(batches)



def optimize_real_hedged_acquisition(acqf: BaseAcquisition, opt_space: MixedDesignSpace,
                                     train_x: GA_Tensor, *, config: GASearchConfig, verbouse=False) -> GA_Tensor:
    # Get all solutions from pareto frontier.
    layout = BatchLayout(opt_space, q=1)
    problem = MixedProblem(
        acqf_with_state=acqf,
        layout=layout,
        opt_space=opt_space,
        n_obj=acqf.n_obj,
        x_train=train_x,
    )
    # make algorithm to solve the problem
    algorithm = NSGA3(
            pop_size=config.pop_size,
            n_offsprings=config.n_offsprings,
            sampling=SobolMixedSampler(opt_space=opt_space, layout=layout),  #* the sampler is of major importance.
            mating=MixedVariableMating(
                selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
            ref_dirs = get_reference_directions("das-dennis",  acqf.n_obj, n_partitions=config.reference_partitions),
    )
    # termination criterion
    termination = get_termination("n_gen", config.max_generations)

    archive = ParetoAccumulator()

    def save_front(algorithm):
        population = algorithm.pop
        if population is None or not len(population):
            return
        archive.add(
            layout.from_records(population.get("X")).reshape(-1, layout.d),
            population.get("F"), population.get("FEAS"),
        )
    # optimize
    minimize(problem, algorithm, termination, verbose=verbouse,
                   save_history=False,
                   copy_algorithm=False,
                   copy_termination=False,
                   callback=save_front,
                   **_pymoo_seed_options(algorithm),
                   )
    if not archive.has_data:
        return torch.empty((0, len(opt_space.params)), dtype=train_x.dtype)
    resX, _ = archive.take_front()
    # shuffle just in case GA returns candidates in some specified order
    rand_resX = resX[torch.randperm(resX.shape[-2])]
    return rand_resX
# endregion

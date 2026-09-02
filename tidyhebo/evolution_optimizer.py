import numpy as np
import torch

from botorch.utils.multi_objective.pareto import is_non_dominated
from collections import OrderedDict
from torch.quasirandom import SobolEngine

from pymoo.algorithms.moo.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.variable import BoundedVariable
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from .acquisition import HedgedPareto
from .common import dtype, device, GA_Tensor, GP_Tensor
from .design_space import MixedDesignSpace

from numpy.typing import NDArray
from typing import List


# region Genetic Algorithm
def _from_dicts_to_np(x, ordered_params):
    # Explicit indexing preserves parameter order and avoids constructing a
    # pandas DataFrame in every NSGA-III generation.
    return np.asarray(
        [[record[name] for name in ordered_params] for record in x],
        dtype=float,
    )


def _get_pareto_from_archive(archive, ordered_params):
    x_hist_front = torch.from_numpy(
        np.vstack(
            [ _from_dicts_to_np(each_X, ordered_params) for each_X in archive._X]
        )
    )
    y_hist_front = -1 * torch.from_numpy(np.vstack(archive._F))
    x_hisorical_paretto_res = x_hist_front[is_non_dominated(y_hist_front, deduplicate=False)]
    return x_hisorical_paretto_res    


class Archive:
    """
    Light-weight saving history
    since pymoo native `save_history=True` saves a lot of additional information
    """
    def __init__(self):
        self._X = []
        self._F = []

    def callback(self, algorithm):
        self._X.append(algorithm.pop.get("X").copy())
        self._F.append(algorithm.pop.get("F").copy())


class MixedProblem(Problem):
    def __init__(self, acqf_with_state: HedgedPareto, 
                vars: List[BoundedVariable], 
                opt_space: MixedDesignSpace,
                n_obj: int,
                x_train: GA_Tensor
                ):
        super().__init__(vars=vars, n_obj=n_obj,
                         n_ieq_constr=1
                        )
        self.acqf_with_state = acqf_with_state
        self.opt_space = opt_space
        self.x_trian: GA_Tensor = x_train


    def _evaluate(self, x: NDArray, out: dict, *args, **kwargs) -> NDArray:
        # type of x - NDArray[Dict[str, Any]] - this is why pandas required
        xt = _from_dicts_to_np(x, self.opt_space.ordered_params)
        torch_x_opt_ga: GA_Tensor = torch.from_numpy(xt)
        torch_x_gp_space: GP_Tensor = self.opt_space.from_GA_to_GP(torch_x_opt_ga.unsqueeze(1)) # shape required by acquisition function
        # optimize is minimization problem => -1 * f(x)
        with torch.no_grad():  # a bit faster
            torch_y = -1 * self.acqf_with_state(torch_x_gp_space)
            torch_constrain = self.opt_space.are_values_new(torch_x_opt_ga, self.x_trian) * (-1) + 0.5
        out["F"] = torch_y.detach().numpy()
        out["G"] = torch_constrain.detach().numpy()


class SobolMixedSampler(Sampling):
    def __init__(self, opt_space: MixedDesignSpace):
        super().__init__()
        self.opt_space: MixedDesignSpace = opt_space


    def _do(self, problem, n_samples, **kwargs):
        n_var = problem.n_var
        sobol_sample: GP_Tensor = SobolEngine(n_var).draw(n_samples, dtype=dtype)
        # rounding preserves low-discrepancy property.
        rounded_sample: GA_Tensor = self.opt_space.from_GP_to_GA(sobol_sample).detach().numpy()
        V = {name: rounded_sample[:, i] for name, i in zip(problem.vars, range(n_var))}

        X = []
        for k in range(n_samples):
            X.append({name: V[name][k] for name in problem.vars.keys()})
        return X


def optimize_real_hedged_acquisition(acqf: HedgedPareto, opt_space: MixedDesignSpace,
                                     train_x: GA_Tensor, pop_size, n_generations, verbouse=False) -> GA_Tensor:
    # Get all solutions from pareto frontier.
    vars = OrderedDict([(p.name, p.pymoo_var) for p in opt_space.params])
    problem = MixedProblem(
        acqf_with_state=acqf,
        vars=vars,
        opt_space=opt_space,
        n_obj=acqf.n_obj,
        x_train=train_x,
    )
    # make algorithm to solve the problem
    algorithm = NSGA3(
            pop_size=pop_size,
            n_offsprings=10,
            sampling=SobolMixedSampler(opt_space=opt_space),  #* the sampler is of major importance.
            mating=MixedVariableMating(
                selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
            ref_dirs = get_reference_directions("das-dennis",  acqf.n_obj, n_partitions=12),
    )
    # termination criterion
    termination = get_termination("n_gen", n_generations)

    archive = Archive()
    # optimize
    res = minimize(problem, algorithm, termination, verbose=verbouse, 
                   save_history=False,
                   copy_algorithm=False,
                   copy_termination=False,
                   callback=archive.callback,
                   )
    resX = _get_pareto_from_archive(archive=archive, ordered_params=opt_space.ordered_params)
    # shuffle just in case GA returns candidates in some specified order
    rand_resX = resX[torch.randperm(resX.shape[-2])]
    return rand_resX
# endregion

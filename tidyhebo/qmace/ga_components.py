"""Match parent sets before applying pymoo's standard mixed-variable operators."""

from math import ceil

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.mixed import MixedVariableMating

from ..design_space import MixedDesignSpace
from ..evolution_optimizer import BatchLayout
from .ga_search import canonicalize


class BatchMatcher:
    """Minimum-cost point matching in normalized Euclidean or GP-kernel geometry.

    Kernel geometry includes the fitted input warp and ARD length scales.
    Matching uses sqrt(2 - 2*rho), where rho is kernel correlation, exactly as
    in the prototype. Canonicalizing both parents resolves ordering ambiguity
    before solving the assignment problem.
    """

    def __init__(self, space: MixedDesignSpace, model, distance: str = "euclidean"):
        if distance not in ("euclidean", "kernel"):
            raise ValueError("matching_distance must be 'euclidean' or 'kernel'")
        self.space, self.model, self.distance = space, model, distance

    def distance_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        ref = self.model.train_inputs[0] if self.distance == "kernel" else torch.empty(0, dtype=torch.double)
        A = self.space.from_GA_to_GP(torch.as_tensor(A).to(ref))
        B = self.space.from_GA_to_GP(torch.as_tensor(B).to(ref))
        with torch.no_grad():
            if self.distance == "euclidean":
                distances = self._euclidean_distance_matrix(A, B)
            else:
                distances = self._kernel_distance_matrix(A, B)
        return distances.cpu().numpy()

    @staticmethod
    def _euclidean_distance_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Pairwise Euclidean distances between points in normalized GP space."""
        return torch.cdist(A, B)

    def _kernel_distance_matrix(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Pairwise sqrt(2 - 2*rho), including the fitted input warp and kernel."""
        A, B = self.model.transform_inputs(A), self.model.transform_inputs(B)
        kernel = self.model.covar_module
        # Only the diagonal is needed for normalization, not two q-by-q matrices.
        aa = kernel(A, A, diag=True).reshape(-1)
        bb = kernel(B, B, diag=True).reshape(-1)
        ab = kernel(A, B).to_dense().reshape(len(A), len(B))
        tiny = torch.finfo(A.dtype).tiny
        rho = ab / (aa.clamp_min(tiny)[:, None] * bb.clamp_min(tiny)[None, :]).sqrt()
        return (2 - 2 * rho.clamp(-1, 1)).clamp_min(0).sqrt()

    def align(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A, B = canonicalize(A), canonicalize(B)
        rows, columns = linear_sum_assignment(self.distance_matrix(A, B))
        return A, B[columns[np.argsort(rows)]]


class SetMating(MixedVariableMating):
    """Align each selected parent pair, then use ordinary crossover/mutation."""

    def __init__(self, layout: BatchLayout, matcher: BatchMatcher, **kwargs):
        super().__init__(**kwargs)
        self.layout, self.matcher = layout, matcher

    def _do(self, problem, pop, n_offsprings, parents=False, **kwargs):
        if not parents:
            pop = self.selection(problem, pop, ceil(n_offsprings / 2), 2, **kwargs)
        aligned = []
        for pair in pop:
            A, B = self.matcher.align(self.layout.from_records(pair[0].X), self.layout.from_records(pair[1].X))
            aligned.append([Individual(X=self.layout.to_records(A)), Individual(X=self.layout.to_records(B))])
        offspring = super()._do(problem, aligned, n_offsprings, parents=True, **kwargs)
        for child in offspring:
            child.X = self.layout.to_records(canonicalize(self.layout.from_records(child.X)))
        return offspring


class SetDuplicates(ElementwiseDuplicateElimination):
    """Treat reordered versions of the same point set as duplicate individuals."""

    def __init__(self, layout: BatchLayout):
        super().__init__()
        self.layout = layout

    def is_equal(self, a, b):
        return np.array_equal(canonicalize(self.layout.from_records(a.X)), canonicalize(self.layout.from_records(b.X)))

"""Exact cumulative feasible Pareto front for CPU, minimizing objectives."""

import numpy as np
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class ParetoAccumulator:
    """Keep the exact front from trusted GA arrays, without generation history.

    Dominance is computed on unique F rows, then mapped back to candidates.
    Distinct X with equal F are retained; exact (X,F) repeats count once.
    take_front() transfers the arrays to tensors and clears the archive.
    """

    def __init__(self):
        self.X = self.F = None

    @property
    def has_data(self) -> bool:
        return self.X is not None and len(self.X) > 0

    def add(self, X, F, feasible=None) -> None:
        X, F = np.asarray(X, dtype=np.float64), np.asarray(F, dtype=np.float64)
        if feasible is not None:
            mask = np.asarray(feasible, dtype=bool).reshape(-1)
            X, F = X[mask], F[mask]
        if len(F):
            unique_F, inverse = np.unique(F, axis=0, return_inverse=True)
            front = NonDominatedSorting(method="efficient_non_dominated_sort").do(
                unique_F, only_non_dominated_front=True,
            )
            keep = np.isin(inverse, front)
            X, F = X[keep], F[keep]
        if self.has_data and len(F):
            old_F, old_inverse = np.unique(self.F, axis=0, return_inverse=True)
            new_F, new_inverse = np.unique(F, axis=0, return_inverse=True)
            keep_old, keep_new = self._cross_front_masks(old_F, new_F)
            self.X, self.F = self.X[keep_old[old_inverse]], self.F[keep_old[old_inverse]]
            X, F = X[keep_new[new_inverse]], F[keep_new[new_inverse]]
        if self.X is not None:
            X, F = np.concatenate((self.X, X)), np.concatenate((self.F, F))
        if len(F):
            # Drop exact repeated candidates, preserving first-seen order.
            _, first = np.unique(np.concatenate((X, F), axis=1), axis=0, return_index=True)
            X, F = X[np.sort(first)], F[np.sort(first)]
        self.X, self.F = X.copy(), F.copy()

    @staticmethod
    def _cross_front_masks(old_F: np.ndarray, new_F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return masks retaining the nondominated union of two minimization fronts.

        Each input is already internally nondominated, with objectives in columns.
        Compare only old/new pairs; equal objective rows survive in both inputs.
        The returned masks index old_F and new_F respectively. Inputs are unchanged.
        """
        old_le_new = np.ones((len(old_F), len(new_F)), dtype=bool)
        new_le_old = np.ones_like(old_le_new)
        for objective in range(old_F.shape[1]):
            old, new = old_F[:, objective, None], new_F[None, :, objective]
            old_le_new &= old <= new
            new_le_old &= new <= old
        keep_old = ~(new_le_old & ~old_le_new).any(axis=1)
        keep_new = ~(old_le_new & ~new_le_old).any(axis=0)
        return keep_old, keep_new

    def take_front(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.has_data:
            raise RuntimeError("Cannot extract a front from an empty archive")
        X, F = torch.from_numpy(self.X), torch.from_numpy(self.F)
        self.X = self.F = None
        return X, F

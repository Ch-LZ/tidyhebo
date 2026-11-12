"""Registered acquisition search; all results use GA coordinates."""

from abc import ABC, abstractmethod
import warnings

import torch
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated

from .evolution_optimizer import optimize_real_hedged_acquisition
from .mixed_params import RealParam
from .qmace import optimize_qmace
from .registry import Registrable


class CandidateSearch(Registrable, ABC):
    """Return candidates (k, d), or whole joint batches (k, q, d).

    A search returns nonempty, unobserved candidates or raises RuntimeError.
    Acquisition functions receive GP coordinates; results use GA coordinates.
    """

    def __init__(self, design_space, config):
        self.design_space, self.config = design_space, config

    @abstractmethod
    def search(self, acqf, train_x):
        ...


# Register the opt-in retained leader after the common search interface exists.
from .layered.search import LayeredSearch  # noqa: E402,F401

class DiscreteSearch(CandidateSearch, _target_='discrete'):
    """Evaluate the configured unobserved pool and retain its Pareto front."""

    def __init__(self, design_space, config):
        super().__init__(design_space, config)
        candidates = config.candidates
        columns = design_space.ordered_params
        assert len(candidates) and candidates.columns.is_unique
        assert set(candidates.columns) == set(columns), f'Candidate columns must match {columns}'
        self.candidates = design_space.from_DS_to_GA(candidates[columns])

    def search(self, acqf, train_x):
        candidates = self.candidates.to(train_x)
        candidates = candidates[self.design_space.are_values_new(candidates, train_x)]
        if not len(candidates):
            raise RuntimeError('No unobserved discrete candidates remain')
        with torch.no_grad():
            values = acqf(self.design_space.from_GA_to_GP(candidates).unsqueeze(1))
        return candidates[is_non_dominated(values, deduplicate=False)]


class PymooSearch(CandidateSearch, _target_='pymoo'):
    """Use the existing sequential or joint GA, with novel-point fallback."""

    def search(self, acqf, train_x):
        if acqf.q > 1:
            candidates = optimize_qmace(acqf, self.design_space, train_x, config=self.config)
        else:
            candidates = optimize_real_hedged_acquisition(
                acqf, self.design_space, train_x, config=self.config)
        if len(candidates):
            return candidates
        warnings.warn(
            "GA returned no feasible candidates; using random fallback points "
            "without acquisition optimization.",
            RuntimeWarning, stacklevel=2,
        )
        candidates = self.design_space.sample_unobserved(train_x, q=acqf.q,
            max_draws=self.config.fallback_samples)
        return candidates.unsqueeze(0) if acqf.q > 1 else candidates


class BoTorchSearch(CandidateSearch, _target_='botorch'):
    """Optimize scalar acquisitions; qLogEI uses sequential greedy batching."""

    def __init__(self, design_space, config):
        super().__init__(design_space, config)
        if not all(isinstance(p, RealParam) for p in design_space.params):
            raise ValueError('BoTorch search requires continuous parameters')

    def search(self, acqf, train_x):
        if getattr(getattr(acqf, "config", None), "_target_", None) == "qLogEI":
            return self._qlogei(acqf, train_x)
        if acqf.n_obj != 1 or acqf.q != 1:
            raise ValueError('BoTorch search requires a scalar q=1 acquisition')
        # Native BoTorch acquisitions return (...), our interface returns (..., 1).
        def scalar(X):
            return acqf(X).squeeze(-1)
        d = len(self.design_space.params)
        bounds = torch.stack((train_x.new_zeros(d), train_x.new_ones(d)))
        points, _ = optimize_acqf(scalar, bounds, q=1,
            num_restarts=self.config.num_restarts, raw_samples=self.config.raw_samples,
            return_best_only=False)
        candidates = self.design_space.from_GP_to_GA(points.squeeze(-2))
        candidates = candidates[self.design_space.are_values_new(candidates, train_x)]
        if not len(candidates):
            raise RuntimeError('BoTorch search returned only observed points')
        with torch.no_grad():
            values = scalar(self.design_space.from_GA_to_GP(candidates).unsqueeze(1))
        return candidates[values.argmax().reshape(1)]


    # region qLogEI batch search and novelty repair
    def _qlogei(self, acqf, train_x):
        native = acqf.native_acqf
        d = len(self.design_space.params)
        bounds = torch.stack((train_x.new_zeros(d), train_x.new_ones(d)))
        points, _ = optimize_acqf(
            acq_function=native, bounds=bounds, q=acqf.q,
            num_restarts=self.config.num_restarts, raw_samples=self.config.raw_samples,
            options={"maxiter": self.config.maxiter, "batch_limit": self.config.batch_limit},
            sequential=self.config.sequential, return_best_only=True,
        )
        candidates = self.design_space.from_GP_to_GA(points.detach())
        if candidates.shape != (acqf.q, d):
            raise RuntimeError("BoTorch returned an invalid qLogEI batch shape")
        selected = []
        repairs = 0
        original_pending = native.X_pending
        try:
            for point in candidates.split(1):
                old = torch.cat((train_x, *selected)) if selected else train_x
                if not bool(self._novel(point, old).all()):
                    pending = self.design_space.from_GA_to_GP(torch.cat(selected)) if selected else None
                    if original_pending is not None:
                        pending = (original_pending if pending is None else
                                   torch.cat((original_pending, pending)))
                    native.set_X_pending(pending)
                    point = self._novel_qlogei_point(native, bounds, old)
                    repairs += 1
                selected.append(point)
        finally:
            # Never leak pending points into a later suggest or the caller's acquisition.
            if repairs or native.X_pending is not original_pending:
                native.set_X_pending(original_pending)
        if repairs:
            warnings.warn(f"qLogEI replaced {repairs} repeated/invalid candidates using "
                          "conditional acquisition search", RuntimeWarning, stacklevel=2)
        candidates = torch.cat(selected)
        return candidates if acqf.q == 1 else candidates.unsqueeze(0)

    def _novel(self, candidates, old):
        return (torch.isfinite(candidates).all(dim=-1) &
                self.design_space.are_values_new(candidates, old))

    def _novel_qlogei_point(self, native, bounds, old):
        """Retain novel local optima, then score a random pool if all optima repeat.

        The pool uses the run's seeded Torch RNG. Every fallback point is ranked
        by the same conditional qLogEI; no objective evaluations are spent here.
        """
        points, _ = optimize_acqf(
            acq_function=native, bounds=bounds, q=1,
            num_restarts=self.config.num_restarts, raw_samples=self.config.raw_samples,
            options={"maxiter": self.config.maxiter, "batch_limit": self.config.batch_limit},
            return_best_only=False,
        )
        pool = self.design_space.from_GP_to_GA(points.detach().reshape(-1, old.shape[-1]))
        pool = pool[self._novel(pool, old)]
        if not len(pool):
            # Reuse the design-space sampler's bounded novelty handling.
            count = max(128, self.config.raw_samples)
            unit = torch.rand((count, old.shape[-1]), dtype=old.dtype, device=old.device)
            pool = self.design_space.from_GP_to_GA(unit)
            pool = pool[self._novel(pool, old)]
            if not len(pool):
                pool = self.design_space.sample_unobserved(old, q=1)
        values = []
        with torch.no_grad():
            for chunk in pool.split(self.config.batch_limit):
                values.append(native(self.design_space.from_GA_to_GP(chunk).unsqueeze(-2)))
        scores = torch.cat(values).reshape(-1)
        finite = torch.isfinite(scores)
        if not bool(finite.any()):
            raise RuntimeError("qLogEI has no finite acquisition values on novel candidates")
        scores = scores.masked_fill(~finite, -torch.inf)
        return pool[scores.argmax().reshape(1)]
    # endregion

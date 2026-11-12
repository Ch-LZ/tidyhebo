"""Augmented Chebyshev scalarization of maximization AF scores.

See docs/chebyshev-scalarization.md for equations and boundary conventions.
Each point/batch level creates its own instance from a frozen Sobol sample.
"""

import torch


class Scalarizer:
    """Convert (..., m) AF scores to (...) scalar minimization losses.

    Calibration fixes one minimum/range per criterion; it never adapts during
    Adam or GA. Rays and scores broadcast over all leading dimensions, so
    independent targets/restarts can share one tensor evaluation.
    """

    def __init__(self, scores, config):
        """Freeze bounds from finite (n, m) calibration scores, with n >= 1."""
        if scores.ndim != 2 or min(scores.shape) == 0 or not torch.isfinite(scores).all():
            raise ValueError("Calibration scores must be a nonempty finite (n, m) tensor")
        self.config = config
        self.low = scores.amin(0).detach()
        self.scale = (scores.amax(0) - self.low).clamp_min(1e-10).detach()
        # A reference point, not a bound on later scores: no clipping is applied.
        self.ideal = scores.new_full((scores.shape[-1],), -0.05)

    def losses(self, scores):
        """Reverse maximization: L_i = 1 - (s_i - low_i) / scale_i.

        Calibration extrema map to 1 and 0. New scores may map outside [0, 1].
        The scale floor prevents division by zero for a constant criterion.
        """
        return 1 - (scores - self.low) / self.scale

    def direction(self, ray):
        """Floor each ray coordinate independently; do not renormalize the ray.

        Zero coordinates on simplex boundary rays become direction_floor,
        making inverse-direction weights finite and strictly positive.
        """
        ray = torch.as_tensor(ray, dtype=self.low.dtype, device=self.low.device)
        return ray.clamp_min(self.config.direction_floor)

    def weights(self, ray):
        """Return w_i = 1 / max(r_i, floor), optionally divided by sum_i w_i.

        Inverse weights balance losses along a geometric ray. Direct ray
        coordinates would instead specify linear objective preferences.
        """
        weights = self.direction(ray).reciprocal()
        if self.config.normalize_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights

    def __call__(self, scores, ray, weights=None):
        """Minimize max_i a_i + rho * sum_i a_i, a_i = w_i * (L_i - z_i).

        The maximum controls the worst weighted deviation; the augmentation
        rewards improvement in the other criteria. torch.amax supplies the
        subgradient at ties. Explicit weights allow reuse across Adam steps.
        For m=1 retain the historical unweighted L_1-z_1 objective.
        """
        deviations = self.losses(scores) - self.ideal
        if deviations.shape[-1] == 1:
            return deviations.squeeze(-1)
        weights = self.weights(ray) if weights is None else weights
        terms = deviations * weights
        return terms.amax(-1) + self.config.rho * terms.sum(-1)

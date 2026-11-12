"""Projected Adam kernels: independent restarts with best-feasible retention."""

import torch


def refine(starts, rays, steps, *, scalar, scores, feasible, continuous, objective_index=None):
    """Batched projected Adam with best-feasible retention for EACH restart.

    Every score uses the same acquisition/QMC realization. This guarantees
    scalar non-degradation against each start for this numerical function.
    """
    if not len(starts) or not continuous.any():
        return starts.detach()
    rays = torch.as_tensor(rays).to(starts)
    if rays.ndim == 1:
        rays = rays.expand(len(starts), -1)
    weights = scalar.weights(rays)
    evaluate = scores
    valid = feasible

    def loss(x):
        scores = evaluate(x)
        values = scalar(scores, rays, weights)
        if objective_index is None:
            return values
        indices = torch.as_tensor(objective_index, device=x.device).expand(len(x))
        anchors = -scores.gather(-1, indices.clamp_min(0)[:, None]).squeeze(-1)
        return torch.where(indices >= 0, anchors, values)

    x = starts.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=0.04)
    with torch.no_grad():
        best_x, best = x.detach().clone(), loss(x).detach()
        best = torch.where(valid(x), best, torch.full_like(best, float("inf")))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        values = loss(x)
        if not torch.isfinite(values).all():
            raise RuntimeError("Non-finite scalarization during local search")
        (grad,) = torch.autograd.grad(values.sum(), x)
        if not torch.isfinite(grad).all():
            raise RuntimeError("Non-finite acquisition gradient")
        x.grad = grad * continuous.to(x)[None, None, :]
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0, 1)
            values = loss(x)
            improved = torch.isfinite(values) & (values < best) & valid(x)
            best_x[improved] = x[improved]
            best[improved] = values[improved]
    return best_x.detach()


def optimize_targets(raw, rays, objectives, restarts, steps, *, scalar, scores, feasible, continuous):
    """All anchor/direction restarts are independent rows of one Adam tensor."""
    q, d = raw.shape[-2:]
    rays = torch.as_tensor(rays, dtype=raw.dtype, device=raw.device)
    objectives = torch.as_tensor(objectives, dtype=torch.long, device=raw.device)

    def losses(scores):
        values = (
            scalar(scores, rays[:, None, :]) if scores.ndim == 3 else scalar(scores[None], rays[:, None, :])
        )
        # Negative objective index denotes a scalarized directional target.
        if scores.ndim == 2:
            anchors = -scores[:, objectives.clamp_min(0)].T
        else:
            anchors = -scores.gather(
                -1, objectives[:, None, None].clamp_min(0).expand(-1, scores.shape[1], 1)
            ).squeeze(-1)
        return torch.where(objectives[:, None] >= 0, anchors, values)

    with torch.no_grad():
        values = losses(scores(raw))
        count = min(restarts, len(raw))
        order = torch.argsort(values, dim=-1)[:, :count]
        starts = raw[order].reshape(-1, q, d)
    refined = refine(
        starts,
        rays.repeat_interleave(count, 0),
        steps,
        scalar=scalar,
        scores=scores,
        feasible=feasible,
        continuous=continuous,
        objective_index=objectives.repeat_interleave(count),
    )
    with torch.no_grad():
        values = losses(scores(refined).reshape(len(rays), count, -1))
        winners = values.argmin(-1)
    return refined.reshape(len(rays), count, q, d)[torch.arange(len(rays), device=raw.device), winners]

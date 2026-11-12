import numpy as np
import pytest
import torch

from tidyhebo.archive import ParetoAccumulator


def exact_front(X, F):
    """Independent dense minimization oracle; deduplicate only entire X/F rows."""
    dominates = (F[:, None] <= F[None, :]).all(-1) & (F[:, None] < F[None, :]).any(-1)
    return {tuple(row) for row in np.concatenate((X, F), axis=1)[~dominates.any(0)]}


def rows(archive):
    return {tuple(row) for row in np.concatenate((archive.X, archive.F), axis=1)}


@pytest.mark.parametrize("objectives", [1, 2, 3, 5])
def test_every_update_matches_full_feasible_history(objectives):
    rng = np.random.default_rng(11)
    archive, all_x, all_f = ParetoAccumulator(), [], []
    for generation in range(15):
        # Discrete values deliberately include objective ties and duplicates.
        X = rng.integers(0, 8, (32, 2)).astype(float)
        F = rng.integers(0, 10, (32, objectives)).astype(float)
        feasible = rng.random(32) > 0.25
        archive.add(X, F, feasible[:, None])
        all_x.append(X[feasible]); all_f.append(F[feasible])
        assert rows(archive) == exact_front(np.concatenate(all_x), np.concatenate(all_f))
        assert len(archive.X) == len(rows(archive))


def test_equal_objectives_keep_distinct_points_and_first_seen_order():
    archive = ParetoAccumulator()
    archive.add([[2], [1], [2], [3]], [[0, 1], [0, 1], [0, 1], [1, 0]])
    archive.add([[1], [4]], [[0, 1], [2, 2]])
    assert archive.X.tolist() == [[2], [1], [3]]
    archive.add([[5]], [[0, 0]])
    assert archive.take_front()[0].tolist() == [[5]]


def test_take_front_transfers_storage_and_clears_archive():
    archive = ParetoAccumulator()
    X, F = np.array([[1.0]]), np.array([[2.0, 3.0]])
    archive.add(X, F)
    X[:] = F[:] = 99
    assert archive.has_data
    stored_x, stored_f = archive.X, archive.F
    x, f = archive.take_front()
    assert x.tolist() == [[1]] and f.tolist() == [[2, 3]]
    assert x.dtype == f.dtype == torch.float64
    assert np.shares_memory(x.numpy(), stored_x)
    assert np.shares_memory(f.numpy(), stored_f)
    assert not archive.has_data
    assert archive.X is archive.F is None
    with pytest.raises(RuntimeError, match="empty archive"):
        archive.take_front()
    archive.add([[4]], [[5, 6]])
    assert archive.has_data
    new_x, new_f = archive.take_front()
    assert new_x.tolist() == [[4]] and new_f.tolist() == [[5, 6]]
    assert x.tolist() == [[1]] and f.tolist() == [[2, 3]]
    assert not archive.has_data


def test_empty_and_infeasible_updates_preserve_front_and_dimensions():
    archive = ParetoAccumulator()
    assert not archive.has_data
    with pytest.raises(RuntimeError, match="empty archive"):
        archive.take_front()
    archive.add(np.empty((0, 2)), np.empty((0, 3)))
    assert not archive.has_data
    assert archive.X.shape == (0, 2)
    with pytest.raises(RuntimeError, match="empty archive"):
        archive.take_front()
    archive.add([[np.nan, 0]], [[np.nan, np.inf, 0]], [False])
    assert not archive.has_data
    assert archive.F.shape == (0, 3)
    with pytest.raises(RuntimeError, match="empty archive"):
        archive.take_front()
    archive.add([[1, 2]], [[0, 0, 0]])
    archive.add(np.empty((0, 2)), np.empty((0, 3)))
    archive.add([[9, 9]], [[-1, -1, -1]], [False])
    assert archive.has_data
    assert archive.take_front()[0].tolist() == [[1, 2]]
    assert not archive.has_data


def test_pymoo_python_fallback_has_the_same_front(monkeypatch):
    loader = pytest.importorskip("pymoo.util.function_loader", reason="Legacy pymoo sorting backend")
    monkeypatch.setattr(loader.FunctionLoader.get_instance(), "mode", "python")
    rng = np.random.default_rng(23)
    archive = ParetoAccumulator()
    X = rng.integers(0, 12, (60, 2)).astype(float)
    F = rng.integers(0, 5, (60, 3)).astype(float)
    for start in range(0, len(X), 10):
        archive.add(X[start:start + 10], F[start:start + 10])
    assert rows(archive) == exact_front(X, F)


def test_unique_objective_comparisons_keep_all_distinct_candidates(monkeypatch):
    import tidyhebo.archive as module

    original = module.NonDominatedSorting
    sizes = []

    class CheckedSorting(original):
        def do(self, F, **kwargs):
            assert len(F) == len(np.unique(F, axis=0))
            sizes.append(len(F))
            return super().do(F, **kwargs)

    monkeypatch.setattr(module, "NonDominatedSorting", CheckedSorting)
    archive = ParetoAccumulator()
    # Only eight tradeoff objective vectors, but 512 distinct candidates.
    group = np.tile(np.arange(8), 32)
    F = np.column_stack((group, 7 - group, group % 2))
    X = np.arange(256, dtype=float)[:, None]
    archive.add(X, F)
    archive.add(X + 256, F)
    assert sizes == [8, 8]
    np.testing.assert_array_equal(archive.X, np.concatenate((X, X + 256)))
    np.testing.assert_array_equal(archive.F, np.concatenate((F, F)))
    # One new vector dominates every stored group; all their rows disappear.
    archive.add([[512]], [[-1, -1, -1]])
    assert archive.take_front()[0].tolist() == [[512]]


def test_unique_objectives_restore_masks_after_dominated_groups_are_removed():
    archive = ParetoAccumulator()
    archive.add([[0], [1], [2]], [[3, 1], [3, 1], [1, 3]])
    X = np.arange(3, 9, dtype=float)[:, None]
    F = np.array([[4, 4], [0, 4], [2, 0], [0, 4], [4, 4], [2, 0]])
    archive.add(X, F)
    assert archive.X.tolist() == [[2], [4], [5], [6], [8]]
    assert rows(archive) == exact_front(
        np.concatenate((np.array([[0], [1], [2]]), X)),
        np.concatenate((np.array([[3, 1], [3, 1], [1, 3]]), F)),
    )


def test_cross_front_masks():
    old_F = np.array([[0, 5], [2, 3], [4, 1], [6, -1]], dtype=float)
    new_F = np.array([[1, 6], [2, 2], [4, 1], [7, -2]], dtype=float)
    old_copy, new_copy = old_F.copy(), new_F.copy()
    keep_old, keep_new = ParetoAccumulator._cross_front_masks(old_F, new_F)
    # Each front dominates one row of the other; equal and tradeoff rows survive.
    np.testing.assert_array_equal(keep_old, [True, False, True, True])
    np.testing.assert_array_equal(keep_new, [False, True, True, True])
    np.testing.assert_array_equal(old_F, old_copy)
    np.testing.assert_array_equal(new_F, new_copy)

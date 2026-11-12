"""
Тесты для функциональности оптимизатора.
"""


from dataclasses import replace

import numpy as np
import torch
import pandas as pd
import pytest
from pytest import approx

from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.mixed_params import CategoricalP, IntegerP, BoolP, RealP, DiscreteRealP
from tidyhebo.optimizer import tidyHEBO


# optimizer io
def test_optimizer_creation():
    cp = CategoricalP(0, ['cat', 'mouse', 'dog'])
    ds = MixedDesignSpace([cp,])
    df: pd.DataFrame = pd.Series(["cat", "cat", "dog", "mouse"]).to_frame(name="x0")
    y = np.array([[0], [1], [1], [0]])

    # empty y
    with pytest.raises(ValueError):
        tidyHEBO(parameters=[cp,], train_x=df, train_y=y[:0])

    # empty x
    with pytest.raises(ValueError):
        tidyHEBO(parameters=[cp,], train_x=df.iloc[:0, :], train_y=y)

    # number of samples mismatch
    with pytest.raises(ValueError):
        tidyHEBO(parameters=[cp,], train_x=df, train_y=y[:-2])

    # Public inputs require a DataFrame.
    with pytest.raises(TypeError):
        tidyHEBO(parameters=[cp,], train_x=torch.zeros(4, 1), train_y=y)
    
    # wrong shape of y
    with pytest.raises(ValueError):
        tidyHEBO(parameters=[cp,], train_x=df, train_y=y.reshape(-1, 1, 1))

    # missmatch column names
    cp2 = CategoricalP(1, ['cat', 'mouse', 'dog'])
    with pytest.raises(ValueError):
        tidyHEBO(parameters=[cp2,], train_x=df, train_y=y)


def test_optimizer_observe():
    cp = CategoricalP(0, ['cat', 'mouse', 'dog'])
    ds = MixedDesignSpace([cp,])
    df: pd.DataFrame = pd.Series(["cat", "cat", "dog", "mouse"]).to_frame(name="x0")
    y = np.array([[0], [1], [1], [0]])

    optimizer = tidyHEBO(parameters=[cp,], train_x=df, train_y=y)
    # empty y
    with pytest.raises(ValueError):
        optimizer.observe(df, y[:0])

    # empty x
    with pytest.raises(ValueError):
        optimizer.observe(df.iloc[:0, :], y)

    # number of samples mismatch
    with pytest.raises(ValueError):
        optimizer.observe(df, y[:-2])

    # Public inputs require a DataFrame.
    with pytest.raises(TypeError):
        optimizer.observe(torch.zeros(4, 1), y)
    
    # wrong shape of y
    with pytest.raises(ValueError):
        optimizer.observe(df, y.reshape(-1, 1, 1))

    # missmatch column names
    df1 = df.copy()
    df1["x1"] = df1.loc[:, "x0"] + 'a'
    with pytest.raises(ValueError):
        optimizer.observe(df1, y)

    optimizer.observe(df.loc[:0, :], y[:1])
    assert optimizer.train_x.shape[0] == optimizer.train_y.shape[0]


def test_order_params():
    p1 = CategoricalP(0, ['cat', 'mouse', 'dog'])
    p2 = IntegerP(1, 0, 1)
    p3 = BoolP(2)
    p4 = RealP(3, 0, 1)
    p5 = DiscreteRealP(4, 2, 5, 4)  # 2, 3, 4, 5

    params = [p1, p2, p3, p4, p5]
    ds = MixedDesignSpace([p1, p2, p3, p4, p5])
    df = pd.DataFrame.from_dict(
        data={"x0": ["cat", "cat", "dog", "mouse"],
               "x1": [0, 0, 1, 1],
               "x2": [True, True, False, False],
               "x3": [0.1, 0.2, 0.3, 0.1],
               "x4": [2., 3, 4, 5.],
            }, 
        orient="columns"
    )
    train_y = np.array([0, 0, 1, 1]).reshape(-1, 1)

    reordered_df = df.copy()[["x4", "x3", "x2", "x1", "x0"]] 
    optimizer = tidyHEBO(params, train_x=reordered_df, train_y=train_y)

    # test if parameters are ordered correctly
    df_recovered = optimizer.design_space.from_GA_to_DS(optimizer.train_x, return_df=True)
    pd.testing.assert_frame_equal(
        df_recovered, df
    )

    # test if parameters are ordered correctly in observe
    new_x = reordered_df.iloc[0, :].to_frame().T
    new_y = train_y[:1, :]
    optimizer.observe(new_x, new_y)

    assert (optimizer.train_x[-1, :] == optimizer.train_x[0, :]).all()


def test_suggest():
    params = [BoolP(0), 
              RealP(1, 0., 1.),
            ]
    train_x = pd.DataFrame.from_dict(
        data={"x0": [True, True, False, False],
               "x1": [0.1, 0.2, 0.3, 0.1],
            }, 
        orient="columns"
    )
    train_y = np.array([0., 0., 1., 1.]).reshape(-1, 1)
    optimizer = tidyHEBO(params, train_x=train_x, train_y=train_y)
    suggestion: pd.DataFrame = optimizer.suggest()
    
    expected_dtypes = pd.Series({
    'x0': bool, 
    'x1': np.float64,
    })
    assert suggestion.dtypes.equals(expected_dtypes)

# ----------------------------------------------------------------------
# Tests for init generator (n_init) functionality
# ----------------------------------------------------------------------

def test_init_generator_single():
    """Точки берутся по одной; после исчерпания генератор обнуляется."""
    param = RealP(0, 0.0, 1.0)
    n_init = 5
    optimizer = tidyHEBO(parameters=[param], n_init=n_init)

    for _ in range(n_init):
        suggestion = optimizer.suggest()
        assert suggestion.shape == (1, 1)
        assert suggestion.columns[0] == param.name
        assert 0.0 <= suggestion.iloc[0, 0] <= 1.0

    assert optimizer._init_design_generator is not None


def test_init_generator_all():
    """Все точки забираются сразу; генератор обнуляется."""
    param = RealP(0, 0.0, 1.0)
    n_init = 5
    optimizer = tidyHEBO(parameters=[param], n_init=n_init)

    optimizer.config = replace(optimizer.config, return_all=True)
    all_points = optimizer.suggest()
    assert all_points.shape == (n_init, 1)
    assert all_points.columns[0] == param.name
    assert all(0.0 <= v <= 1.0 for v in all_points.iloc[:, 0])
    assert optimizer._init_design_generator is None


def test_init_generator_mixed():
    """Смешанный режим: пара точек по одной, остаток — через return_all."""
    param = RealP(0, 0.0, 1.0)
    n_init = 5
    optimizer = tidyHEBO(parameters=[param], n_init=n_init)

    singles = [optimizer.suggest() for _ in range(2)]
    optimizer.config = replace(optimizer.config, return_all=True)
    rest = optimizer.suggest()

    assert rest.shape[0] == n_init - 2
    values = [pt.iloc[0, 0] for pt in singles] + rest.iloc[:, 0].tolist()
    assert len(set(values)) == n_init
    assert optimizer._init_design_generator is None


@pytest.mark.parametrize("n_init", [None, 0, -1, True, 1.5])
def test_initialization_requires_a_positive_integer(n_init):
    with pytest.raises(ValueError, match="positive integer n_init"):
        tidyHEBO([RealP(0, 0, 1)], n_init=n_init)


@pytest.mark.parametrize("missing", ["X", "y"])
def test_observations_must_be_supplied_together(missing):
    X = None if missing == "X" else pd.DataFrame({"x0": [.2]})
    y = None if missing == "y" else [[1.]]
    with pytest.raises(ValueError, match="both X and y"):
        tidyHEBO([RealP(0, 0, 1)], train_x=X, train_y=y, n_init=4)


@pytest.mark.parametrize("y", [[1., 2.], [[1., 2.]], 1.])
def test_observation_targets_require_a_column(y):
    X = pd.DataFrame({"x0": [.2, .7]})
    with pytest.raises(ValueError, match="shape"):
        tidyHEBO([RealP(0, 0, 1)], train_x=X, train_y=y)


def test_duplicate_input_columns_are_rejected():
    X = pd.DataFrame([[.2, .3]], columns=["x0", "x0"])
    with pytest.raises(ValueError, match="unique"):
        tidyHEBO([RealP(0, 0, 1)], train_x=X, train_y=[[1.]])


@pytest.mark.parametrize("n_init", [4, 0, -1])
def test_observations_take_priority_over_initialization(monkeypatch, n_init):
    from tidyhebo.candidate_search import CandidateSearch
    from types import SimpleNamespace

    X = pd.DataFrame({"x1": [True, False], "x0": [.2, .7]}, index=[3, 8])
    original = X.copy(deep=True)
    y = [[1.], [2.]]
    optimizer = tidyHEBO([RealP(0, 0, 1), BoolP(1)], train_x=X, train_y=y, n_init=n_init)
    assert optimizer._init_design_generator is None
    search = SimpleNamespace(search=lambda *args: torch.tensor([[.4, 2.]], dtype=torch.double))
    monkeypatch.setattr(CandidateSearch, "from_config", lambda *args, **kwargs: search)
    monkeypatch.setattr(optimizer, "_train_and_get_acqf", lambda: (None, None, None))
    result = optimizer.suggest()
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["x0", "x1"]
    assert result.iloc[0, 0] == approx(.4)
    optimizer.observe(X.iloc[:1], [[3.]])
    torch.testing.assert_close(optimizer.train_x[-1], optimizer.train_x[0])
    assert optimizer.train_y.dtype == torch.double
    pd.testing.assert_frame_equal(X, original)
    assert y == [[1.], [2.]]
    before_x, before_y = optimizer.train_x.clone(), optimizer.train_y.clone()
    with pytest.raises(ValueError, match="shape"):
        optimizer.observe(X, [1., 2.])
    torch.testing.assert_close(optimizer.train_x, before_x)
    torch.testing.assert_close(optimizer.train_y, before_y)

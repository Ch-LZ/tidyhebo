"""
Тесты для функциональности оптимизатора.
"""


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
    y = np.array([0, 1, 1, 0])

    # empty y
    with pytest.raises(AssertionError):
        tidyHEBO(parameters=[cp,], train_x=df, train_y=y[:0])

    # empty x
    with pytest.raises(AssertionError):
        tidyHEBO(parameters=[cp,], train_x=df.iloc[:0, :], train_y=y)

    # number of samples mismatch
    with pytest.raises(AssertionError):
        tidyHEBO(parameters=[cp,], train_x=df, train_y=y[:-2])

    # wrong type of y
    with pytest.raises(AssertionError):
        tidyHEBO(parameters=[cp,], train_x=df, train_y=torch.from_numpy(y))
    
    # wrong shape of y
    with pytest.raises(AssertionError):
        tidyHEBO(parameters=[cp,], train_x=df, train_y=y.reshape(-1, 1, 1))

    # missmatch column names
    cp2 = CategoricalP(1, ['cat', 'mouse', 'dog'])
    with pytest.raises(AssertionError):
        tidyHEBO(parameters=[cp2,], train_x=df, train_y=y)


def test_optimizer_observe():
    cp = CategoricalP(0, ['cat', 'mouse', 'dog'])
    ds = MixedDesignSpace([cp,])
    df: pd.DataFrame = pd.Series(["cat", "cat", "dog", "mouse"]).to_frame(name="x0")
    y = np.array([0, 1, 1, 0])

    optimizer = tidyHEBO(parameters=[cp,], train_x=df, train_y=y)
    # empty y
    with pytest.raises(AssertionError):
        optimizer.observe(df, y[:0])

    # empty x
    with pytest.raises(AssertionError):
        optimizer.observe(df.iloc[:0, :], y)

    # number of samples mismatch
    with pytest.raises(AssertionError):
        optimizer.observe(df, y[:-2])

    # wrong type of y
    with pytest.raises(AssertionError):
        optimizer.observe(df, torch.from_numpy(y))
    
    # wrong shape of y
    with pytest.raises(AssertionError):
        optimizer.observe(df, y.reshape(-1, 1, 1))

    # missmatch column names
    df1 = df.copy()
    df1["x1"] = df1.loc[:, "x0"] + 'a'
    with pytest.raises(AssertionError):
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
    new_y = train_y[0, :]
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
        suggestion = optimizer.suggest(return_all=False)
        assert suggestion.shape == (1, 1)
        assert suggestion.columns[0] == param.name
        assert 0.0 <= suggestion.iloc[0, 0] <= 1.0

    assert optimizer._init_design_generator is not None


def test_init_generator_all():
    """Все точки забираются сразу; генератор обнуляется."""
    param = RealP(0, 0.0, 1.0)
    n_init = 5
    optimizer = tidyHEBO(parameters=[param], n_init=n_init)

    all_points = optimizer.suggest(return_all=True)
    assert all_points.shape == (n_init, 1)
    assert all_points.columns[0] == param.name
    assert all(0.0 <= v <= 1.0 for v in all_points.iloc[:, 0])
    assert optimizer._init_design_generator is None


def test_init_generator_mixed():
    """Смешанный режим: пара точек по одной, остаток — через return_all."""
    param = RealP(0, 0.0, 1.0)
    n_init = 5
    optimizer = tidyHEBO(parameters=[param], n_init=n_init)

    singles = [optimizer.suggest(return_all=False) for _ in range(2)]
    rest = optimizer.suggest(return_all=True)

    assert rest.shape[0] == n_init - 2
    values = [pt.iloc[0, 0] for pt in singles] + rest.iloc[:, 0].tolist()
    assert len(set(values)) == n_init
    assert optimizer._init_design_generator is None

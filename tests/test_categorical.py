"""
Тесты для функциональности - передачи данных в API через dataframe
"""


import numpy as np
import torch
import pandas as pd
import pytest
from pytest import approx

from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.mixed_params import CategoricalP, IntegerP, BoolP, RealP, DiscreteRealP

# mixed params test
def test_simple_convertion():
    cp = CategoricalP(0, ['cat', 'mouse', 'dog'])

    # convertion by map
    df_row = pd.Series(["cat", "cat", "dog", "mouse"])

    ds_row = df_row.map(cp.cat_to_index)
    assert (ds_row > 0).all(), "[1, n]"

    ds_row_ref = pd.Series([1., 1., 3., 2.])
    assert (ds_row_ref.map(cp.index_to_cat) == df_row).all()

# design space
def test_design_space_convertion():
    cp = CategoricalP(0, ['cat', 'mouse', 'dog'])
    ds = MixedDesignSpace([cp,])
    df: pd.DataFrame = pd.Series(["cat", "cat", "dog", "mouse"]).to_frame(name="x0")

    # DS -> GA
    ref_ga = torch.tensor([1., 1., 3., 2.]).reshape(-1, 1)
    assert (ds.from_DS_to_GA(df.copy()) == ref_ga).all()

    # GA -> DS
    ## to tensor
    # не надо хотеть тензор, если подразумевается небоходимость df
    # assert (ds.from_GA_to_DS(ref_ga, return_df=False) == ref_ga).all()
    ## to dataframe
    assert ds.from_GA_to_DS(ref_ga, return_df=True).equals(df)


def test_ds_convertion2():
    cp1 = CategoricalP(0, ['cat', 'mouse', 'dog'])
    cp2 = CategoricalP(1, ['duck', 'bear', 'bean'])

    ds = MixedDesignSpace([cp1, cp2])
    df = pd.DataFrame.from_dict(
        data={"x0": ["cat", "cat", "dog", "mouse"],
               "x1": ["bear", "duck", "bean", "bean"]},
        orient="columns"
    )
    ref_ga = torch.tensor(
        [[1., 1., 3., 2.],
        [2, 1, 3, 3]]).T
    # DS -> GA
    assert (ds.from_DS_to_GA(df.copy()) == ref_ga).all()

    # GA -> DS
    ## to tensor
    # assert (ds.from_GA_to_DS(ref_ga, return_df=False) == ref_ga).all()
    ## to dataframe
    assert ds.from_GA_to_DS(ref_ga, return_df=True).equals(df)


def test_ds_convertion_type_convertion():
    p1 = CategoricalP(0, ['cat', 'mouse', 'dog'])
    p2 = IntegerP(1, 0, 1)
    p3 = BoolP(2)
    p4 = RealP(3, 0, 1)
    p5 = DiscreteRealP(4, 2, 5, 4)  # 2, 3, 4, 5

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
    ref_ga = torch.tensor(
        [[1., 1., 3., 2.],  # x0
        [1, 1, 2, 2],  # x1
        [2, 2, 1, 1],  # x2
        [0.1, 0.2, 0.3, 0.1],  # x3
        [1., 2, 3, 4.],  # x4
        ], dtype=torch.double).T  # todo: проставить всюду, где создается новый тензор приведение типов
    # DS -> GA
    assert ds.from_DS_to_GA(df.copy()) == approx(ref_ga)

    # GA -> DS
    ## to tensor
    # ref_ds_tensor = torch.tensor(
    #     [[1., 1., 3., 2.],
    #      [0, 0, 1, 1],
    #      [1, 1, 0, 0,],
    #      [0.1, 0.2, 0.3, 0.1],
    #      [2., 3, 4, 5.],
    #      ], dtype=torch.double).T
    # assert ds.from_GA_to_DS(ref_ga, return_df=False) == approx(ref_ds_tensor)
    ## to dataframe
    # assert ds.from_GA_to_DS(ref_ga, return_df=True).equals(df)
    pd.testing.assert_frame_equal(
        ds.from_GA_to_DS(ref_ga, return_df=True),
        df,
        atol=1e-6
    )

from tidyhebo.design_space import MixedDesignSpace
from tidyhebo.mixed_params import (RealP, BoolP, DiscreteRealP, 
                                     CategoricalP, IntegerP, AbstractParameter)
from tidyhebo.common import dtype

import torch
from pytest import approx


def test_transformations_simple():
    params = [RealP(0, 0, 1), RealP(1, 0, 1), RealP(2, 0, 1)]

    ms = MixedDesignSpace(params)
    x_ds = torch.tensor([[0, 0.3, 1],
                         [0, 0.3, 1],
                         [0, 0.3, 1]])
    # x_ga = x_ds + 1
    x_ga = x_ds
    x_gp = x_ds.clone()
    assert ms.from_DS_to_GA(x_ds) == approx(x_ga)
    assert ms.from_GA_to_DS(x_ga, return_df=False) == approx(x_ds)

    assert ms.from_GA_to_GP(x_ga) == approx(x_gp)
    assert ms.from_GP_to_GA(x_gp) == approx(x_ga)

    assert ms.from_DS_to_GP(x_ds) == approx(x_gp)
    assert ms.from_GP_to_DS(x_gp, return_df=False) == approx(x_ds)

    params = [RealP(0, 1, 3), RealP(1, 2, 4), RealP(2, 3, 5)]
    ms = MixedDesignSpace(params)

    x_ds = torch.tensor([
        [1, 2, 3],
        [3, 4, 5]
    ], dtype=dtype)
    x_gp = torch.tensor([
        [0, 0, 0,],
        [1, 1, 1]
    ])
    assert ms.from_DS_to_GP(x_ds) == approx(x_gp)


def test_transformations_single_simple():
    params = [RealP(0, 0, 1)]

    ms = MixedDesignSpace(params)
    x_ds = torch.tensor([[0,],
                         [0.3,],
                         [1,]])
    # x_ga = x_ds + 1
    x_ga = x_ds
    x_gp = x_ds.clone()
    assert ms.from_DS_to_GA(x_ds) == approx(x_ga)
    assert ms.from_GA_to_DS(x_ga, return_df=False) == approx(x_ds)

    assert ms.from_GA_to_GP(x_ga) == approx(x_ds)
    assert ms.from_GP_to_GA(x_gp) == approx(x_ga)

    assert ms.from_DS_to_GP(x_ds) == approx(x_ds)
    assert ms.from_GP_to_DS(x_gp, return_df=False) == approx(x_ds)    


def test_sampling():
    params = [RealP(0, 0, 1)]
    ms = MixedDesignSpace(params)
    assert ms._large_GA_sample == approx(torch.linspace(0, 1, 100).reshape(-1, 1))

    params = [BoolP(0)]
    ms = MixedDesignSpace(params)
    assert ms._large_GA_sample == approx(torch.tensor([1., 2]).reshape(-1, 1))

    params = [BoolP(0), RealP(1, 0, 1)]
    ms = MixedDesignSpace(params)
    assert ms._large_GA_sample == approx(
        torch.cartesian_prod(
            torch.tensor([1., 2]), torch.linspace(0, 1, 100)
            )
        )
    

def test_new_vals():
    mds = MixedDesignSpace([BoolP(0)])
    cand = torch.tensor([[0,],
                         [1.,]], dtype=dtype)
    X_old = torch.tensor([[0.,]], dtype=dtype)
    assert (mds.are_values_new(cand, X_old) == torch.tensor([False, True])).all()

    # граничные условия
    X_old = torch.empty(0, cand.shape[-1])
    assert mds.are_values_new(cand, X_old).all()
    assert mds.are_values_new(X_old, cand).all()

    mds = MixedDesignSpace(
        [BoolP(0),
        IntegerP(1, 1, 4),
        RealP(2, 2, 3),])
    cand = torch.tensor([
        [0., 1, 2],
        [0., 1, 3]
    ])
    X_old = torch.tensor([
        [0., 1, 2],
        [0., 4, 2],
    ])
    assert (mds.are_values_new(cand, X_old) == torch.tensor([False, True])).all()


def test_tolerance():
    mds = MixedDesignSpace(
        [BoolP(0), RealP(1, 1, 2., ds_tolerance=0.5)]
    )
    cand = torch.tensor([
        [0., 1,],
        [0., 1.2,],
    ])
    X_old = torch.tensor([
        [0., 1,],
    ])
    print(mds.tolerances)
    assert (mds.are_values_new(cand, X_old) == torch.tensor([False, False])).all()


def test_repr():
    mds = MixedDesignSpace([BoolP(0), BoolP(1)])
    assert repr(mds) == "[BoolP[x0](), BoolP[x1]()]"
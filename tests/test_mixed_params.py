import torch
import pytest
from pytest import approx
import pymoo.core.variable as pmv

from tidyhebo.mixed_params import (RealP, BoolP, DiscreteRealP, 
                                CategoricalP, IntegerP, RealPowP,
                                AbstractParameter)


def test_realp():
    rp = RealP(0, 1., 3.)

    assert rp.index == 0
    assert rp.name == "x0"

    assert rp.ga_lb == 0.
    assert rp.ga_ub == 1.

    assert isinstance(rp.pymoo_var, pmv.Real)

    x = torch.tensor([2., 3.,])
    x_ga = (x - 1) / 2
    assert torch.all(rp.from_DS_to_GA(x) == x_ga)
    assert torch.all(rp.from_GA_to_DS(x_ga) == x)

    assert torch.all(rp.from_GA_to_GP(x_ga) == torch.tensor([0.5, 1.]))
    assert torch.all(rp.from_GP_to_GA(torch.tensor([0.5, 1.])) == x_ga)

    rp2 = RealP(0, 1., 3., name="DEADBEEF")
    assert rp2.name == "DEADBEEF"
    # representations test
    assert repr(rp2) == "RealP[DEADBEEF](lb=1.0, ub=3.0)"
    rp3 = RealP(1, 1., 3., ds_tolerance=1e-5)
    assert repr(rp3) == "RealP[x1](lb=1.0, ub=3.0, tol=1e-05)"


def test_boolp():
    bp = BoolP(0)

    assert bp.index == 0
    assert bp.name == "x0"
    assert bp.ga_lb == 1
    assert bp.ga_ub == 2
    assert isinstance(bp.pymoo_var, pmv.Integer)

    x_ds = torch.tensor([True, False])
    x_ga = torch.tensor([2., 1.])
    x_gp = torch.tensor([0.75, 0.25])

    assert (bp.from_DS_to_GA(x_ds) == x_ga).all()
    assert (bp.from_GA_to_DS(x_ga) == x_ds).all()
    assert bp.from_GA_to_DS(x_ga).dtype is torch.bool
    
    assert bp.from_GA_to_GP(x_ga) == approx(x_gp, abs=1e-5)
    assert (bp.from_GP_to_GA(x_gp) == x_ga).all()

    bp2 = BoolP(0, name="DEADBEEF")
    assert bp2.name == "DEADBEEF"
    assert repr(bp2) == "BoolP[DEADBEEF]()"


def test_integerp():
    ip = IntegerP(0, 1, 3)

    assert ip.index == 0
    assert ip.name == "x0"
    assert ip.ga_lb == 1
    assert ip.ga_ub == 3

    assert isinstance(ip.pymoo_var, pmv.Integer)
    assert ip.pymoo_var.bounds[0] == 1
    assert ip.pymoo_var.bounds[1] == 3

    x_ds = torch.tensor([2., 3.])
    x_ga = torch.tensor([2., 3.])
    x_gp = (x_ga - 0.5) / (3.5 - 0.5)
    assert ip.from_DS_to_GA(x_ds) == approx(x_ga, abs=1e-5)
    assert ip.from_GA_to_DS(x_ga) == approx(x_ds, abs=1e-5)

    assert ip.from_GA_to_GP(x_ga) == approx(x_gp, abs=1e-5)
    assert ip.from_GP_to_GA(x_gp) == approx(x_ga, abs=1e-5)

    ip2 = IntegerP(0, 1, 3, name="DEADBEEF")
    assert ip2.name == "DEADBEEF"
    assert repr(ip2) == "IntegerP[DEADBEEF](lb=1, ub=3)"

#todo: прописываем все контракты и делаем сильные тесты
def test_categoricalp():
    cp = CategoricalP(0, [-3, -2, -1])

    assert cp.index == 0
    assert cp.name == "x0"
    assert cp.ga_lb == 1
    assert cp.ga_ub == 3
    assert cp.categories_are_numeric == True

    assert isinstance(cp.pymoo_var, pmv.Choice)
    assert frozenset(cp.pymoo_var.options) == frozenset((1, 2, 3))

    x_ds = torch.tensor([-3, -2, -1])
    x_ga = torch.tensor([1.,2, 3])
    x_gp = (x_ga - 0.5) / (3.5 - 0.5)
    assert cp.from_DS_to_GA(x_ds) == approx(x_ga)
    assert cp.from_GA_to_DS(x_ga) == approx(x_ds)

    assert cp.from_GA_to_GP(x_ga) == approx(x_gp, abs=1e-5)
    assert cp.from_GP_to_GA(x_gp) == approx(x_ga, abs=1e-5)

    cp2 = CategoricalP(0, [-3, -2, -1], name="DEADBEEF")
    assert cp2.name == "DEADBEEF"
    assert repr(cp2) == "CategoricalP[DEADBEEF](categories=[-3, -2, -1])"


def test_discreterealp():
    drp = DiscreteRealP(0, 2, 5, 4)

    assert drp.index == 0
    assert drp.name == "x0"
    assert drp.ga_lb == 1
    assert drp.ga_ub == 4

    assert isinstance(drp.pymoo_var, pmv.Integer)
    assert drp.design_med_dist == approx(1.)

    x_ds = torch.tensor([2, 5, 3, 4])
    x_ga = torch.tensor([1, 4, 2, 3])
    x_gp = (x_ga - 0.5) / (4.5 - 0.5)
    assert drp.from_DS_to_GA(x_ds) == approx(x_ga)
    assert drp.from_GA_to_DS(x_ga) == approx(x_ds)

    assert drp.from_GA_to_GP(x_ga) == approx(x_gp, abs=1e-5)
    assert drp.from_GP_to_GA(x_gp) == approx(x_ga, abs=1e-5)

    cp2 = DiscreteRealP(0, 2, 5, 4, name="DEADBEEF")
    assert cp2.name == "DEADBEEF"

    assert repr(cp2) == "DiscreteRealP[DEADBEEF](lb=2, ub=5, n=4)"


def test_tolerance():
    rp = RealP(0, 1., 3.)
    # assert rp.ga_tol == (3 - 1) / 100
    assert rp.ga_tol == 1e-6


def test_realpowp():
    rpp = RealPowP(0, 1, 1000, 10)

    assert rpp.index == 0
    assert rpp.name == "x0"
    assert rpp.ga_lb == 0.0
    assert rpp.ga_ub == 1.0
    assert rpp._llb == approx(0.0)
    assert rpp._lub == approx(3.0)

    assert isinstance(rpp.pymoo_var, pmv.Real)

    x_ds = torch.tensor([1.0, 10., 100.])
    x_ga = torch.tensor([0.0, 1., 2.]) / 3.
    x_gp = x_ga.clone()

    assert torch.all(rpp.from_DS_to_GA(x_ds) == x_ga)
    assert rpp.from_GA_to_DS(x_ga) == approx(x_ds)

    assert torch.all(rpp.from_GA_to_GP(x_ga) == x_gp)
    assert torch.all(rpp.from_GP_to_GA(x_gp) == x_ga)

    assert repr(rpp) == "RealPowP[x0](lb=1, ub=1000, base=10)"

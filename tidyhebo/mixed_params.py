import numbers
import pandas as pd
import torch
import math

from abc import ABC, abstractmethod
from functools import cached_property
import pymoo.core.variable as pmv
from typing import List, Optional, Dict, Any

from .common import GA_Tensor, GP_Tensor, DS_Tensor


class AbstractParameter(ABC):
    def __init__(self, index: int, name: Optional[str]=None):

        self.name: str
        if name is not None:
            self.name = str(name)
        elif index is not None:
            self.name = f"x{index}"
        else:
            raise RuntimeError("Nigher name, nor index was provided")

        self.index: int = index
        self.ga_lb = None # to be filled
        self.ga_ub = None # to be filled
        self.ga_tol = 1e-6

    #* GA <-> GP
    @abstractmethod
    def from_GA_to_GP(self, X: GA_Tensor) -> GP_Tensor:
        pass

    @abstractmethod
    def from_GP_to_GA(self, X: GP_Tensor) -> GA_Tensor:
        pass

    #* DS <-> GA
    @abstractmethod
    def from_DS_to_GA(self, X: DS_Tensor) -> GA_Tensor:
        pass

    @abstractmethod
    def from_GA_to_DS(self, X: GA_Tensor) -> DS_Tensor:
        pass

    @cached_property
    @abstractmethod
    def pymoo_var(self) -> pmv.BoundedVariable:
        # borders in GA space
        pass

    # @abstractmethod
    # def __repr__(self) -> str:
    #     pass


class RealParam(AbstractParameter):  # abstract
    def __init__(self, index: int, ub: float, lb: float, name: Optional[str]=None) -> None:
        super().__init__(index=index, name=name)
        self.ga_lb = lb  # in GA space
        self.ga_ub = ub  # in GA space

    #* GA <-> GP
    def from_GA_to_GP(self, X: GA_Tensor) -> GP_Tensor:
        return X

    def from_GP_to_GA(self, X: GP_Tensor) -> GA_Tensor:
        return X


class RealP(RealParam):
    def __init__(self, index: int, lb: float, ub: float, *, ds_tolerance: Optional[float]=None, name: Optional[str]=None) -> None:
        self.ds_lb = lb
        self.ds_ub = ub
        self.ds_tolerance = ds_tolerance

        ga_tolerance = ds_tolerance / (ub - lb) if ds_tolerance else 1e-6
        super().__init__(index=index, lb=0, ub=1, name=name)
        self.ga_tol = ga_tolerance or abs(self.ga_ub - self.ga_lb) * 1e-8

    #* DS <-> GA
    def from_DS_to_GA(self, X: DS_Tensor) -> GA_Tensor:
        return (X - self.ds_lb) / (self.ds_ub - self.ds_lb)

    def from_GA_to_DS(self, X: GA_Tensor) -> DS_Tensor:
        return X * (self.ds_ub - self.ds_lb) + self.ds_lb

    @cached_property
    def pymoo_var(self) -> pmv.BoundedVariable:
        # borders in GA space
        return pmv.Real(bounds=(self.ga_lb, self.ga_ub))

    def __repr__(self) -> str:
        _cls: str = self.__class__.__name__
        if self.ds_tolerance is not None:
            repr = f"{_cls}[{self.name}](lb={self.ds_lb}, ub={self.ds_ub}, tol={self.ds_tolerance})" 
        else:
            repr = f"{_cls}[{self.name}](lb={self.ds_lb}, ub={self.ds_ub})" 
        return repr


class RealPowP(RealParam):
    def __init__(self, index: int, lb: float, ub: float, base: float = 10, *, name: Optional[str]=None) -> None:
        assert lb > 0
        self.ds_lb = lb
        self.ds_ub = ub
        self.base = base

        self._lub = math.log(ub, base)
        self._llb = math.log(lb, base)
        self._lbase = math.log(self.base)

        super().__init__(index=index, lb=0, ub=1,name=name)

    #* DS <-> GA
    def from_DS_to_GA(self, X: DS_Tensor) -> GA_Tensor:
        X_tr = X.log() / self._lbase
        return (X_tr - self._llb) / (self._lub - self._llb)

    def from_GA_to_DS(self, X: GA_Tensor) -> DS_Tensor:
        X_tr = X * (self._lub - self._llb) + self._llb
        return (X_tr * self._lbase).exp()

    @cached_property
    def pymoo_var(self) -> pmv.BoundedVariable:
        # borders in GA space
        return pmv.Real(bounds=(self.ga_lb, self.ga_ub))

    def __repr__(self) -> str:
        _cls: str = self.__class__.__name__
        rv = f"{_cls}[{self.name}](lb={self.ds_lb}, ub={self.ds_ub}, base={self.base})" 
        return rv


class DiscreteParam(AbstractParameter):  # abstract
    def __init__(self, index: int, lb: int, ub: int, n: int, *, name: Optional[str]=None) -> None:
        super().__init__(index=index, name=name)
        self.ga_lb = lb  # in GA space
        self.ga_ub = ub  # in GA space

        assert lb == 1, f"{lb} != 1"
        assert ub == n, "should be in GA space"

        self._aug_lb = lb - 0.499999  # ~0.5
        self._aug_ub = ub + 0.499999  # ~0.5

    #* GA <-> GP
    """
    Особенность: при  GP -> GA переходе инты округляются до ближайшего инта
    Поэтому при переходе GA -> GP границы расширяются на 0.5 с обоих сторон.
    """
    def from_GA_to_GP(self, X: GA_Tensor) -> GP_Tensor:
        return (X - self._aug_lb) / (self._aug_ub - self._aug_lb)

    def from_GP_to_GA(self, X: GP_Tensor) -> GA_Tensor:
        return (X * (self._aug_ub - self._aug_lb) + self._aug_lb).round()


class BoolP(DiscreteParam):
    def __init__(self, index, *, name: Optional[str]=None):
        super().__init__(index, lb=1, ub=2, n=2, name=name)  # lb = 1 possibly due to rounding issues during converting
    
    #* DS <-> GA
    def from_DS_to_GA(self, X: DS_Tensor) -> GA_Tensor:
        return (X + 1).to(dtype=torch.double)

    def from_GA_to_DS(self, X: GA_Tensor) -> DS_Tensor:
        return (X - 1).round().to(torch.bool)

    @cached_property
    def pymoo_var(self) -> pmv.BoundedVariable:
        # borders in GA space
        return pmv.Integer(bounds=(self.ga_lb, self.ga_ub))

    def __repr__(self) -> str:
        _cls: str = self.__class__.__name__
        rv = f"{_cls}[{self.name}]()"
        return rv


class IntegerP(DiscreteParam):
    def __init__(self, index, lb, ub, *, name: Optional[str]=None):
        self._shift = lb
        super().__init__(index, lb - self._shift + 1, ub - self._shift + 1, n=ub - lb + 1, name=name)

    #* DS <-> GA
    def from_DS_to_GA(self, X: DS_Tensor) -> GA_Tensor:
        return X.to(dtype=torch.double) - self._shift + 1

    def from_GA_to_DS(self, X: GA_Tensor) -> DS_Tensor:
        return X - 1 + self._shift

    @cached_property
    def pymoo_var(self) -> pmv.BoundedVariable:
        # borders in GA space
        return pmv.Integer(bounds=(self.ga_lb, self.ga_ub))
    
    def __repr__(self) -> str:
        _cls: str = self.__class__.__name__
        ds_lb = self._shift
        ds_ub = self.ga_ub + self._shift - 1

        rv = f"{_cls}[{self.name}](lb={ds_lb}, ub={ds_ub})"
        return rv


class CategoricalP(DiscreteParam):
    """
    Категории могут быть числами или строками.
    """
    def __init__(self, index, categories: List, *, name: Optional[str]=None):
        # self.index_to_cat = categories
        self.cat_to_index: Dict[Any, int] = {cat: ind for ind, cat in enumerate(categories, start=1)}
        self.index_to_cat: Dict[int, Any] = {ind: cat for ind, cat in enumerate(categories, start=1)}
        self.__category_type = type(categories[0])
        self.categories_are_numeric: bool = issubclass(self.__category_type, numbers.Number)
        self._convert_here: bool = self.categories_are_numeric
        self.n = len(categories)
        super().__init__(index=index, lb=1, ub=self.n, n=self.n, name=name)


    #* DS <-> GA
    def from_DS_to_GA(self, X: DS_Tensor) -> GA_Tensor:
        if not self._convert_here:
            """
            Если категории не числа - принято решение обновлять их на уровне выше - в design_space
            """
            return X
        else:  # numeric categories
            X_ga = torch.tensor(
                [self.cat_to_index[elem.item()] for elem in X], dtype=torch.double)
            return X_ga


    def from_GA_to_DS(self, X: GA_Tensor) -> DS_Tensor:
        if not self._convert_here:
            """
            Если категории не числа - принято решение обновлять их на уровне выше - в design_space
            """
            return X
        else:  # numeric categories
            X_ds = torch.tensor(
                [self.index_to_cat[elem.item()] for elem in X.round().to(dtype=int)], dtype=torch.double)
            return X_ds

    @cached_property
    def pymoo_var(self) -> pmv.BoundedVariable:
        # borders in GA space
        return pmv.Choice(options=list(range(self.ga_lb, self.ga_ub + 1)))

    def __repr__(self) -> str:
        _cls: str = self.__class__.__name__
        sorted_keys_asc = sorted(self.cat_to_index, key=self.cat_to_index.get)
        rv = f"{_cls}[{self.name}](categories={sorted_keys_asc})"
        return rv


class DiscreteRealP(DiscreteParam):
    def __init__(self, index, lb, ub, n, *, name: Optional[str]=None):
        super().__init__(index, lb=1, ub=n, n=n, name=name)

        self.design_med_dist = (ub - lb) / (n - 1)

        self.design_lb = lb
        self.design_ub = ub
    
    #* DS <-> GA
    def from_DS_to_GA(self, X: DS_Tensor) -> GA_Tensor:
        return (X - self.design_lb) / self.design_med_dist + 1

    def from_GA_to_DS(self, X: GA_Tensor) -> DS_Tensor:
        return (X - 1) * self.design_med_dist + self.design_lb

    @cached_property
    def pymoo_var(self) -> pmv.BoundedVariable:
        # borders in GA space
        return pmv.Integer(bounds=(self.ga_lb, self.ga_ub))

    def __repr__(self) -> str:
        _cls: str = self.__class__.__name__
        repr = f"{_cls}[{self.name}](lb={self.design_lb}, ub={self.design_ub}, n={self.ga_ub})"
        return repr

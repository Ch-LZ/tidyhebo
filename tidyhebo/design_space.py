import numpy as np
import pandas as pd
import torch

from functools import cached_property
from typing import List, Iterator

from .common import dtype, device, DS_df, DS_Tensor, GA_Tensor, GP_Tensor
from .mixed_params import AbstractParameter, DiscreteParam, RealParam, CategoricalP, IntegerP, BoolP, RealP, DiscreteRealP, RealPowP


# region MixedDesign
class MixedDesignSpace:
    __convertion_methods = ["from_DS_to_GA", "from_GA_to_DS", "from_GA_to_GP", "from_GP_to_GA"]

    def __init__(self, parameters: List[AbstractParameter]):
        # проверка, что параметры идут в порядке индексов.
        assert (np.array([p.index for p in parameters]) == np.arange(len(parameters))).all(), "Params should be provided in sorted index order"
        # проверка что все параметры называются по-разному
        assert len(set([p.name for p in parameters])) == len(parameters), "All parameters should be provided with unique names."
        self.params: List[AbstractParameter] = parameters

        self.ordered_params: List[str] = [p.name for p in parameters]
        self.tolerances: GA_Tensor = torch.tensor([p.ga_tol for p in parameters]).unsqueeze(-1)
        self._categorical_params: List[CategoricalP] = [p for p in parameters if isinstance(p, CategoricalP)]
        self._map_categories_here = None
        self._set_map_cat_here(False)

    def from_DS_to_GA(self, x: DS_df | DS_Tensor) -> GA_Tensor:
        # если тип - DS_df, то преобразуем его в тензор
        if isinstance(x, DS_df(pd.DataFrame)):
            """
            #* pd.DataFrame <-> DS_Tensor
            Суть проблемы: хотим работать с категориальными переменными выраженными строками, 
            torch.Tensor не поддерживает строки, а pd.DataFrame поддерживает
            """
            self._set_map_cat_here(True)
            x = x.copy()
            p: CategoricalP
            for p in self._categorical_params:
                x[p.name] = x.loc[:, p.name].map(p.cat_to_index).astype(float)
            # меняем тип на Tensor
            x: DS_Tensor = torch.from_numpy(x.values.astype(float))
        else:
            self._set_map_cat_here(False)

        return self.__apply_along_columns("from_DS_to_GA", x)

    def from_GA_to_DS(self, x: GA_Tensor, *, return_df: bool) -> DS_df | DS_Tensor:
        """
        Не до конца понятно как обозначать тип, который нужно вернуть. 
        DS_Tensor(torch.Tensor) удобен для тестирования и был изначально для внутреннего использования. 
        DS_df(pd.DataFrame) понадобился для внешнего использования.
        """
        self._set_map_cat_here(return_df)
        x: DS_Tensor = self.__apply_along_columns("from_GA_to_DS", x)
        if return_df:
            # требуется привести типы к исходным
            x = pd.DataFrame(x.detach().numpy(), columns=self.ordered_params)
            p: AbstractParameter
            for p in self.params:
                if isinstance(p, CategoricalP):
                    x[p.name] = x.loc[:, p.name].map(p.index_to_cat)
                elif isinstance(p, IntegerP):
                    x[p.name] = x[p.name].astype(int)
                elif isinstance(p, BoolP):
                    x[p.name] = x[p.name].astype(bool)
                else:
                    assert isinstance(p, (RealP, DiscreteRealP, RealPowP))

            x: DS_df 
        return x


    def from_GA_to_GP(self, x: GA_Tensor) -> GP_Tensor:
        return self.__apply_along_columns("from_GA_to_GP", x)

    def from_GP_to_GA(self, x: GP_Tensor) -> GA_Tensor:
        return self.__apply_along_columns("from_GP_to_GA", x)

    def from_DS_to_GP(self, x: DS_Tensor) -> GP_Tensor:
        return self.from_GA_to_GP(
            self.from_DS_to_GA(x)
        )

    def from_GP_to_DS(self, x: GP_Tensor, *, return_df: bool) -> DS_Tensor:
        return self.from_GA_to_DS(
            self.from_GP_to_GA(x), return_df=return_df
        )

    def __apply_along_columns(self, method: str, x: torch.Tensor) -> torch.Tensor:
        assert method in self.__convertion_methods, "invalid convertion method"
        # if q-batch
        original_shape = x.shape
        # Если тензор 3D (b, q, d), преобразуем в 2D (b * q, d)
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        
        result = torch.stack(
            [getattr(param, method)(column) 
             for column, param in zip(torch.unbind(x, dim=1), self.params)], dim=1)
        
        # Восстанавливаем исходную размерность, если нужно
        if len(original_shape) == 3:
            result = result.reshape(*original_shape)
        return result


    def are_values_new(self, cand: GA_Tensor, X_old: GA_Tensor) -> torch.Tensor:
        if (cand.numel() == 0) != (X_old.numel() == 0):
            return torch.full((cand.shape[0],), True)
        diff = (X_old.unsqueeze(-1) - cand.T).abs()
        return (diff > self.tolerances).any(dim=1).all(dim=0)           


    def _set_map_cat_here(self, value: bool):
        # if nothing changes - do nothing
        # print(f"setting map cat here to: {value}")
        if self._map_categories_here == value:
            return
        else:
            for p in self._categorical_params:
                p._convert_here = not value
            self._map_categories_here = value


    @cached_property
    def _large_GA_sample(self) -> GA_Tensor:
        """
        Большой и достаточно разнообразный семпл из GA-space, 
        заведомо достаточный чтобы получить новую точку в случае фейла сложной оптимизации (очень редкое событие)
        Потому что алгоритм планируется использовать на величинах порядка <~10 осей и кардинальность каждой из дискретных будет небольшой
        и количество непрерывных осей будет маленьким как правило.
        """
        # просто каждую непрерывную ось на 100 частей попилить и всё.
        variants = self.__full_space_variants()
        res = torch.cartesian_prod(*variants)
        if res.dim() == 1:
            res = res.unsqueeze(-1)
        return res


    def _large_GA_sample_iter(self) -> Iterator:
        from itertools import product
        variants = self.__full_space_variants()
        res = product(*variants)
        return res


    def __full_space_variants(self):
        variants = []
        for p in self.params:
            if isinstance(p, DiscreteParam):
                variants.append(torch.arange(p.ga_lb, p.ga_ub + 1).to(dtype=dtype))
            elif isinstance(p, RealParam):
                variants.append(torch.linspace(p.ga_lb, p.ga_ub, steps=100).to(dtype=dtype))
            else:
                assert False, "Unsupported parameter type"
        return variants
# endregion

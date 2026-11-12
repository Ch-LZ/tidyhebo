import numpy as np
import pandas as pd
import torch

from typing import List

from .common import DS_df, DS_Tensor, GA_Tensor, GP_Tensor
from .mixed_params import AbstractParameter, CategoricalP, IntegerP, BoolP, RealP, DiscreteRealP, RealPowP


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


    def sample_unobserved(self, train_x: GA_Tensor, q: int = 1, *, max_draws: int = 4096) -> GA_Tensor:
        """Draw uniformly in [0, 1)^d and apply the existing GA novelty predicate.

        At most max_draws proposals are generated in chunks of 256. Selected points
        are also excluded, so a q-batch is mutually novel. Failure to find q points
        within the budget does not prove that the domain is exhausted.
        """
        if q < 1 or max_draws < 1:
            raise ValueError("q and max_draws must be positive")
        selected = []
        for start in range(0, max_draws, 256):
            gp = torch.rand((min(256, max_draws - start), len(self.params)),
                            dtype=train_x.dtype, device=train_x.device)
            pool = self.from_GP_to_GA(gp)
            old = torch.cat((train_x, *selected)) if selected else train_x
            pool = pool[self.are_values_new(pool, old)]
            while len(pool):
                point = pool[:1]
                selected.append(point)
                if len(selected) == q:
                    return torch.cat(selected)
                pool = pool[self.are_values_new(pool, point)]
        raise RuntimeError(
            f"Could not construct q={q} novel fallback points after {max_draws} random draws; "
            "the space may be exhausted or the remaining region too small. "
            "Increase GASearchConfig.fallback_samples to allow more attempts."
        )


    def __repr__(self) -> str:
        return repr(self.params)
# endregion

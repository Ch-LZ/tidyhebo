"""
Design space (DS) - то, как переменные видит вызывающий код. К примеру, категориальные в виде названий категорий.
Данные в DS могут приходить как в виде тензора,так и в виде датафрейма. Это связано с категориаьлными переменными: torch.Tensor не поддерживает тип строк (str).
Genetic Algorithm space (GA) (=optimization space) - Параметры на входе в генетический алгоритм. Все дискретные переменные в диапазоне [1, n], действительные - в диапазоне [0, 1].
Gaussian Process space (GP) - переменные на входе в гауссовский процесс. Все нормальзиванные в [0, 1]. Дискретные переменные тоже в [0, 1].

DS <-> GA <-> GP
"""

import torch
import pandas as pd
from typing import NewType

dtype = torch.double
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  #* попытки заставить работать на gpu оставлены на потом.


GA_Tensor = NewType("GA_Tensor", torch.Tensor)
GP_Tensor = NewType("GP_Tensor", torch.Tensor)
DS_Tensor = NewType("DS_Tensor", torch.Tensor)
DS_df = NewType("DS_df", pd.DataFrame)

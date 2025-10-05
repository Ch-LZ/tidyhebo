# how to install
pip install -e .

При этом из версий библиотек важен именно botorch. Остальные можно взять любые совместимые.

# How to run
Это как с нуля запустить
```python
## optimizer imports
import numpy as np
import pandas as pd

from tidyhebo import RealOptimizer
from tidyhebo import RealP, CategoricalP, IntegerP, DiscreteRealP

## define parameter space
params = [
    IntegerP(0, 2, 6, name="log2_batch_size"),
    RealP(1, -5, -3, ds_tolerance=0.1, name="log10_lr"),
    DiscreteRealP(2, 1e-2, 1.5e-1, n=5, name="weight_decay"),
    DiscreteRealP(3, 0.1, 0.5, n=5, name="dropout"),
    CategoricalP(4, ["max", "mean"], name="pooling")
]

## generate initialization sample
# from tidyhebo import MixedDesignSpace
# from torch.quasirandom import SobolEngine

# n_init = 8
# design_space = MixedDesignSpace(params)
# x_init: GP_Tensor = SobolEngine(len(params)).draw(n_init, dtype=dtype)
# x_init: DS_df = design_space.from_GP_to_DS(x_init, return_df=True)

#* X requred to be pandas datafrme with columns in a right order.
## columns order and names are specified by `params' (see above) by their index and name respectively.
x_known = pd.read_csv("init_for_dbp.csv", index_col=0)
# y required to be a 2d numpy array
y_known = np.random.rand(8, 1)

# create optimizer
optimizer = RealOptimizer(parameters=params,
                          train_x=x_known,
                          train_y=y_known)
optimizer.use_brute_force = True

# make a new suggestion about next point
x_cand = optimizer.suggest()  # took ~14 seconds for 6 params

print(x_cand)

# convert a suggestion into a dict if needed for convinient config usages
x_cand_as_dict = x_cand.to_dict(orient="records")[0]

print(x_cand_as_dict)
```
Это как добавить новые наблюдения
```python
# required to be a dataframe as `x_known`
x_new = x_known.iloc[0].to_frame().T
# required to be a 2d numpy array
y_new = y_known[0, :]
print(x_new)
print(y_new)


optimizer.observe(X=x_new, y=y_new)

optimizer.suggest()
```

## В конце можно проверить по сетке параметров предсказываемый мксимум и проверить был ли он посещён в истории
"""
Пронаблюдал один раз, что Brute Force подход может не посетить максимум. Опционально можно ослабить сетку параметров и посмотреть не предполагается ли моделью максимум между узлами сетки.
"""

params = [
    IntegerP(0, 2, 6, name="batch_size"),
    RealP(1, -5, -3, ds_tolerance=0.1, name="lr"),
    DiscreteRealP(2, 1e-2, 1.5e-1, n=5, name="weight_decay"),
    DiscreteRealP(3, 0.1, 0.5, n=5, name="dropout"),
    CategoricalP(4, ["max", "avg"], name="pooling")
]


## generate space 
from tidyhebo import MixedDesignSpace
from torch.quasirandom import SobolEngine

n_all = 2*17  # x4 избыток
design_space = MixedDesignSpace(params)
x_all = SobolEngine(len(params)).draw(n_all, dtype=dtype)
x_all = design_space.from_GP_to_DS(x_all, return_df=True)


## обучить GP внутри модели
x_init = SobolEngine(len(params)).draw(8, dtype=dtype)
x_init = design_space.from_GP_to_DS(x_init, return_df=True)
y_known = np.random.rand(8, 1)


optimizer = RealOptimizer(parameters=params,
                          train_x=x_init,  # x_known
                          train_y=y_known) # y_known
# optimizer.use_brute_force = True
optimizer.suggest()

## найти максимум среди сетки
x_all_inference = design_space.from_DS_to_GP(x_all)

posterior = optimizer.model.model.posterior(x_all_inference)

x_all.iloc[posterior.mean.argmax().item()]  # новые параметры

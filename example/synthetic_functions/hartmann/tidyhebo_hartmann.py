import numpy as np
import pandas as pd
import torch
from tidyhebo import RealOptimizer, MixedDesignSpace
from tidyhebo import RealP, CategoricalP, IntegerP, DiscreteRealP
from tqdm import tqdm
from botorch.test_functions.synthetic import Hartmann
import torch


# define a problem
problem = Hartmann(dim=6, negate=True)

def test_optimizer(n_total, n_init):
    # dfeine a parameter space
    params = [
        RealP(i, lb, ub) for (i, (lb, ub)) in enumerate(problem.bounds.T.detach().numpy())
    ]
    ds = MixedDesignSpace(params)

    # sample initial points
    n_init = 8
    x0 = torch.rand(n_init, len(params)).double()
    x0 = torch.from_numpy(ds.from_GP_to_DS(x0, return_df=True).values)

    y0 = problem(x0).reshape(-1, 1)

    # initialize optimizer
    optimizer = RealOptimizer(
        parameters=params, train_x=x0, train_y=y0)

    for _ in tqdm(range(n_init, n_total), total=n_total, initial=n_init):
        x_new = optimizer.suggest()
        y_new = problem(x_new)
        optimizer.observe(x_new, y_new)
    return optimizer.train_y.detach().numpy()


if __name__ == '__main__':
    from datetime import datetime
    
    n_repeats = 30
    n_trials = 100

    for i in range(n_repeats):
        print()
        print(f"iter: {i} time: {datetime.now()}")
        current_res = test_optimizer(n_total=n_trials, n_init=4).T
        with open("data/tidyhebo_H6_out.txt", "a") as f:
            np.savetxt(f, current_res)

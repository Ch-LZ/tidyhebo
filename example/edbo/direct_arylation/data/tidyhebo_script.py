from tidyhebo.acquisition import HedgedPareto
from tidyhebo.base_model import SimpleModel
from botorch.utils.multi_objective.pareto import is_non_dominated
import torch
import numpy as np
import tqdm
import pandas as pd


def test_edbo(n_total):
    n_init = 8
    assert n_total > n_init
    ## initialize
    df = pd.read_csv("../direct_arylation_embeddings.csv", index_col=0)

    xy0 = df.sample(n_init)
    x0 = xy0.drop(columns="yield")
    y0 = xy0.loc[:, "yield"].to_numpy()

    x_historical = x0
    x_all_new = pd.concat([df.drop(columns="yield"), x_historical]).drop_duplicates(keep=False)
    y_historical = y0.reshape(-1, 1)

    _state_dict = None
    for _ in tqdm.tqdm(range(n_total-n_init), total=n_total, initial=n_init):
        # get model
        model = SimpleModel(train_x=torch.from_numpy(x_historical.to_numpy()),
                            train_y=torch.from_numpy(y_historical).reshape(-1, 1),
                            kernel="Matern52"
                            )

        model.train_model(state_dict=_state_dict)
        _state_dict = model.model.state_dict()

        # calculate acquisition
        acqf = HedgedPareto(model=model.model,
                            best_f=torch.from_numpy(y_historical).max().item(),
                            beta=4,
                            train_x=torch.from_numpy(x_historical.to_numpy()),
                            )

        x_all_new_torch = torch.from_numpy(x_all_new.to_numpy())
        with torch.no_grad():
            yb = acqf(x_all_new_torch.unsqueeze(1))

        # select paretto-frontier
        x_cand = x_all_new[is_non_dominated(yb, deduplicate=False).detach().numpy()]
        x_new = x_cand.sample(1)

        # "evaluate" objective
        y_new = df.loc[x_new.index, "yield"].to_numpy().reshape(-1, 1)


        x_historical = pd.concat((x_historical, x_new))
        y_historical = np.concatenate((y_historical, y_new))
        x_all_new = pd.concat([df.drop(columns="yield"), x_historical]).drop_duplicates(keep=False)

    return y_historical


if __name__ == '__main__':
    from datetime import datetime

    # import tqdm 
    all_edbo_res = []
    n_repeats = 50
    n_trials = 100
    for i in range(n_repeats):
        print()
        print(f"iter: {i} time: {datetime.now()}")
        all_edbo_res.append(test_edbo(n_trials).T)

    tidyhebo_edbo_res = np.vstack(all_edbo_res)

    np.savetxt("edbo_features_matern52_dimscaled.txt", tidyhebo_edbo_res)

from edbo.bro import BO


import pandas as pd
df = pd.read_csv("suzuki_embedded.csv", index_col=0)
from edbo import Data


def get_simulation_results(n_steps):
    bo = BO(
        exindex=df,
        domain=df.drop('yield', axis=1),
        batch_size=1,
        acquisition_function='EI',
        fast_comp=True
    )
    bo.simulate(iterations=n_steps)
    outputs = bo.obj.results_input()["yield"].to_numpy().ravel()
    return outputs


from datetime import datetime
import numpy as np


all_edbo_res = []
n_repeats = 50
n_trials = 100
for i in range(n_repeats):
    print()
    print(f"iter: {i} time: {datetime.now()}")
    all_edbo_res.append(get_simulation_results(n_trials))


tidyhebo_edbo_res = np.vstack(all_edbo_res)

np.savetxt("edbo_suzuki_edbo.txt", tidyhebo_edbo_res)

# model here
import os
import numpy as np
import sys

conda_prefix = os.path.basename(os.getenv('CONDA_PREFIX'))
print(f"model is running in: {conda_prefix} with {sys.executable}")

print(sys.argv[1:])
pipe1, pipe2, inp_file, output_file, number_of_iterations, dataset = sys.argv[1:]
number_of_iterations = int(number_of_iterations)

#### initialize optimizer
# from tidyhebo import MixedDesignSpace, RealP
import torch
import optuna


study = optuna.create_study(direction="maximize")


for i in range(number_of_iterations):
    trial = study.ask()
    x = [
        trial.suggest_float("reaction_time", 3., 31),
        trial.suggest_float("sultine_conc", 1.6, 6.0),
        trial.suggest_float("temperature", 100, 150),
    ]
    next_point = np.array(x).reshape(1, -1)

    np.savetxt(inp_file, next_point)

    ## sync
    with open(pipe1, "w") as f:
        f.write("1")
        f.flush()

    ## sync
    with open(pipe2, "r") as f:
        f.read()

    # загрузить данные
    rv = np.loadtxt(output_file)
    study.tell(trial.number, rv.ravel()[0])


# postprocessing
with open("optuna_values.txt", "a") as f:
    np.savetxt(f, 
               np.array(study.trials_dataframe()["value"]).reshape(1, -1)
               )
# how to run:
# for i in {1..30}; do echo -e "\n run $i $(date)";. script.sh hplc 100; done
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
        trial.suggest_float("q_agno3", 4.53, 42.80981595),
        trial.suggest_float("q_pva", 9.999518096000001, 40.00101474),
        trial.suggest_float("q_tsc", 0.5, 30.5),
        trial.suggest_float("q_seed", 0.498851653, 19.5),
        trial.suggest_float("q_tot", 200.0, 983.0),
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
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
from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace
from tqdm import tqdm


params = [
    {'name': 'reaction_time', 'type': 'num', 'lb': 3., 'ub': 31.},
    {'name': 'sultine_conc', 'type': 'num', 'lb': 1.6, 'ub': 6.0},
    {'name': 'temperature', 'type': 'num', 'lb': 100., 'ub': 150.},
]

hebo_design_space = DesignSpace().parse(params)
optimizer = HEBO(hebo_design_space)  #* minimization task only

results = []
for i in tqdm(range(number_of_iterations), total=number_of_iterations):
    cand = optimizer.suggest(n_suggestions=1)
    next_point = cand.values.reshape(1, -1)
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
    results.append(rv)
    optimizer.observe(cand, -1 * rv.reshape(-1))

# postprocessing
with open("hebo_values.txt", "a") as f:
    np.savetxt(f, 
               np.array(results).reshape(1, -1)
            )
# how to run:
# for i in {1..30}; do echo -e "\n run $i $(date)";. script.sh hplc 100; done
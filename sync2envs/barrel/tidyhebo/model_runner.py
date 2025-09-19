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
from tidyhebo import MixedDesignSpace, RealOptimizer, RealP
import torch
from tqdm import tqdm


params = [
    RealP(0, 6.0, 12.0, name="n"),
    RealP(1, 0.0, 200.0, name="theta"),
    RealP(2, 1.5, 2.5, name="r"),
    RealP(3, 0.7, 1.4, name="t"),
]

ds = MixedDesignSpace(params)
x0 = torch.rand(4, len(params)).double()
x0 = ds.from_GP_to_DS(x0, return_df=True).values

##### init batch забираем именно через for
y0 = []
for cand in x0:
    cand = cand.reshape(1, -1)
    np.savetxt(inp_file, cand)
    ## sync
    with open(pipe1, "w") as f:
        f.write("1")
        f.flush()
    ## sync
    with open(pipe2, "r") as f:
        f.read()

    y0.append(np.loadtxt(output_file).ravel()[0])
# загрузить данные
y0 = torch.tensor(y0).reshape(-1, 1)
x0 = torch.from_numpy(x0)
optimizer = RealOptimizer(parameters=params, train_x=x0, train_y=y0)
#### end init
# endregion
results = y0.ravel().tolist()
for i in tqdm(range(x0.shape[0], number_of_iterations),
              total=number_of_iterations - x0.shape[0]):
    cand = optimizer.suggest()
    np.savetxt(inp_file, cand.detach().numpy().reshape(1, -1))

    ## sync
    with open(pipe1, "w") as f:
        f.write("1")
        f.flush()

    ## sync
    with open(pipe2, "r") as f:
        f.read()

    # загрузить данные
    y_new = np.loadtxt(output_file)
    results.append(y_new.ravel()[0])
    rv = torch.from_numpy(y_new).reshape(-1, 1)
    optimizer.observe(cand, rv)

# postprocessing
with open("tidyhebo_values.txt", "a") as f:
    np.savetxt(f, 
               np.array(results).reshape(1, -1)
               )
# how to run:
# for i in {1..30}; do echo -e "\n run $i $(date)";. script.sh hplc 100; done
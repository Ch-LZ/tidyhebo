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
from tidyhebo import MixedDesignSpace, RealP
import torch

params = [
    RealP(0, 6.0, 12.0, name="n"),
    RealP(1, 0.0, 200.0, name="theta"),
    RealP(2, 1.5, 2.5, name="r"),
    RealP(3, 0.7, 1.4, name="t"),
]

ds = MixedDesignSpace(params)
n_random_samples = 100 * 100
x0 = torch.rand(n_random_samples, len(params))
x0 = ds.from_GP_to_DS(x0, return_df=True).values


next_point = x0
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

# postprocessing
np.savetxt("random_out.txt", rv)

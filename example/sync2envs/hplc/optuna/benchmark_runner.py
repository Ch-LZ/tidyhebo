# benchmark here
import os
import numpy as np
import sys

conda_prefix = os.path.basename(os.getenv('CONDA_PREFIX'))
print(f"benchmark is running in: {conda_prefix} with {sys.executable}")

pipe1, pipe2, inp_file, output_file, number_of_iterations, dataset = sys.argv[1:]
number_of_iterations = int(number_of_iterations)

# initialize emulator
from olympus import Emulator

dataset_name = dataset
em = Emulator(dataset_name, "BayesNeuralNet")

for i in range(number_of_iterations):
    # подождать появления данных
    with open(pipe1, "r") as f:
        f.read()

    # загрузить данные
    next_point = np.loadtxt(inp_file)
    # проделать вычисления
    rv = em.run(next_point)
    rv = np.array(rv)

    # положить их в файл
    np.savetxt(output_file, rv)

    # посигналить, что данные доступны
    with open(pipe2, "w") as f:
        f.write("1")
        f.flush()

# benchmark here

import os
import sys
conda_prefix = os.path.basename(os.getenv('CONDA_PREFIX'))
print(f"foo is running in: {conda_prefix} with {sys.executable}")

from olympus import Emulator

import numpy as np


pipe1, pipe2, inp_file, output_file, number_of_iterations, dataset = sys.argv[1:]

number_of_iterations = int(number_of_iterations)
# initialize emulator
dataset_name = dataset
em = Emulator(dataset_name, "BayesNeuralNet")

for i in range(number_of_iterations):
    # ждем пока не отправят икс
    # print(f"FOO: wait for x  at {pipe1}", flush=True)
    with open(pipe1, "r") as f:
        f.read()

    # загрузить данные
    # print(f"FOO: load data from {inp_file}", flush=True)
    next_point = np.loadtxt(inp_file)
    # проделать вычисления
    rv = em.run(next_point)
    rv = np.array(rv)
    print(rv)
    # положить их в файл
    # print(f"FOO: save data to {output_file}", flush=True)
    np.savetxt(output_file, rv)
    # посигналить, что данные доступны
    # print(f"FOO: signal to {pipe2}", flush=True)
    with open(pipe2, "w") as f:
        f.write("1")
        f.flush()

# model here
import os
import numpy as np
import sys

conda_prefix = os.path.basename(os.getenv('CONDA_PREFIX'))
print(f"bar is running in: {conda_prefix} with {sys.executable}")

print(sys.argv[1:])
pipe1, pipe2, inp_file, output_file, number_of_iterations, dataset = sys.argv[1:]
number_of_iterations = int(number_of_iterations)
# initialize optimizer

#todo: write function to conduct one test and whole experiment
for i in range(number_of_iterations):
    # положить данные в файл
    #todo: suggest new point and turn it into numpy

    next_point = [[0.01, 0.02, 0.5, 1.0, 100., 7.]]
    next_point = np.array(next_point) + i * 0.001
    print(next_point)
    # print(f"BAR: save x to file {inp_file}", flush=True)
    np.savetxt(inp_file, next_point)
    # почитать из пайпа 1
    # print(f"BAR: signal to pipe1", flush=True)
    with open(pipe1, "w") as f:
        f.write("1")
        f.flush()

    # ждем пока приготовят ответ
    # print(f"BAR: wait at pipe2", flush=True)
    with open(pipe2, "r") as f:
        f.read()

    # загрузить данные
    # print(f"BAR: load data from {output_file}", flush=True)
    rv = np.loadtxt(output_file)
    #todo: after eash loading make sure to append results to a file
    # print(f"received_rv: {rv}", flush=True)

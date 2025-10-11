#!/bin/bash

# execute via `source` or `.`
# don't execute with `./script.sh`, it won't work


# run as source srcipt.sh ...
# echo "First argument is" "$1"

dataset="$1"
number_of_iterations="$2"
echo "dataset '$1' with number of iterations '$2'"

echo "self pid is:" $$
# make shure directory is clean
rm from_M_to_B 2> /dev/null
rm from_B_to_M 2> /dev/null
rm $inp_file 2> /dev/null
rm $out_file 2> /dev/null

# define file names
mkfifo from_M_to_B  # make pipe1
mkfifo from_B_to_M  # make pipe2
inp_file="t_input.txt"
out_file="t_output.txt"
# create new empty files
touch $inp_file
touch $out_file

# first part
benchmark(){
    mamba deactivate
    mamba activate olympus

    python benchmark_runner.py from_M_to_B from_B_to_M $inp_file $out_file $number_of_iterations $dataset
}
benchmark >&2 &
benchmarkpid=$!
echo "benchmark pid is" $benchmarkpid

# second part
model(){
    mamba deactivate
    mamba activate hebo

    python model_runner.py from_M_to_B from_B_to_M $inp_file $out_file $number_of_iterations $dataset
}

model >&2 &
modelpid=$!  # pid of last background task

echo "model pid is:" $modelpid
wait $benchmarkpid
wait $modelpid

rm from_M_to_B
rm from_B_to_M
rm $inp_file
rm $out_file

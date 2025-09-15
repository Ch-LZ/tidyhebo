
# run as source srcipt.sh ...

number_of_iterations=2
dataset=hplc

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
model(){
    mamba deactivate
    mamba activate olympus

    python foo.py from_M_to_B from_B_to_M $inp_file $out_file $number_of_iterations $dataset
}
model >&2 &
modelpid=$!
echo "model pid is" $modelpid

# second part
benchmark(){
    mamba deactivate
    mamba activate tidyhebo

    python bar.py from_M_to_B from_B_to_M $inp_file $out_file $number_of_iterations $dataset
}

benchmark >&2 &
benchmarkpid=$!  # pid of last background task

echo "benchmark pid is:" $benchmarkpid
wait $benchmarkpid
wait $modelpid

rm from_M_to_B
rm from_B_to_M
rm $inp_file
rm $out_file

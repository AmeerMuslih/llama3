#!/bin/bash

start_dir=$1
num_dirs=$2

# Get a list of all directories and limit the range
dirs=($(ls -d ~/Ameer/llama3/Matrices/*/))

# Iterate over the specified range of directories
for ((i=$start_dir; i<$((start_dir + num_dirs)); i++)); do
    if [ $i -ge ${#dirs[@]} ]; then
        echo "Reached end of directories"
        break
    fi

    dir=${dirs[$i]}
    dirname=$(basename "$dir")
    echo "Processing $dir"
    nohup srun python3 ./Matmul_SA.py "$dir" > ./Outputs/output_$dirname & disown
done

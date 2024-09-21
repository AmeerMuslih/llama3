#!/bin/bash

for dir in ~/Ameer/llama3/Matrices/*/; do
    nohup srun python3 ./Matmul_SA.py $dir > ./Outputs/output_$dir & disown
done
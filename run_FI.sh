#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <bitFlips_number>"
    exit 1
fi

bitFlips_number=$1
bitFlips_number=$((bitFlips_number/72))

nohup srun --gres=gpu:12 -w plotinus1 torchrun --nproc_per_node 1 sentiment_analysis_imdb.py --ckpt_dir Meta-Llama-3-8B-Instruct --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 1 --max_num_pred 72 --batch_start_idx 0 --group_id 0 --bitFlips_number $bitFlips_number > output_0 & disown
nohup srun --gres=gpu:12 -w plotinus2 torchrun --nproc_per_node 1 sentiment_analysis_imdb.py --ckpt_dir Meta-Llama-3-8B-Instruct --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 1 --max_num_pred 72 --batch_start_idx 0 --group_id 1 --bitFlips_number $bitFlips_number > output_1 & disown




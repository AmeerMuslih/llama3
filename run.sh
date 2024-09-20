#!/bin/bash

for ((i=0; i<6; i+=2))
do
    nohup srun --gres=gpu:12 -w plotinus1 torchrun --nproc_per_node 1 sentiment_analysis_imdb.py --ckpt_dir Meta-Llama-3-8B-Instruct --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 1 --max_num_pred 12 --batch_start_idx $((i*12)) --group_id $i > output_$i & disown
    nohup srun --gres=gpu:12 -w plotinus2 torchrun --nproc_per_node 1 sentiment_analysis_imdb.py --ckpt_dir Meta-Llama-3-8B-Instruct --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 2048 --max_batch_size 1 --max_num_pred 12 --batch_start_idx $(((i+1)*12)) --group_id $((i+1)) > output_$((i+1)) & disown
done



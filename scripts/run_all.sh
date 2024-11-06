#!/bin/bash

tasks=("cola" "sst2" "qnli" "mrpc" "rte" "stsb")

for task in "${tasks[@]}"
do
    python3 run_trglue.py \ 
    --model_name_or_path dbmdz/bert-base-turkish-cased \ 
    --task_name "$task" \ 
    --max_seq_length 128 \ 
    --output_dir berturca_"$task" \ 
    --num_train_epochs 3 \ 
    --learning_rate 2e-5 \ 
    --per_device_train_batch_size 128 \ 
    --per_device_eval_batch_size 128 \ 
    --do_train \ 
    --do_eval \ 
    --save_strategy no
done


#!/bin/bash
#SBATCH -c 4 # Number of cores
#SBATCH -N 1 # 1 node requested
#SBATCH -n 1 # 1 task requested
#SBATCH --mem=180000 # Memory - Use 32G
#SBATCH --time=0 # No time limit
#SBATCH -o /bos/tmp16/luyug/outputs/multibert/logs/train-10-anchor.out  # send stdout to outfile
#SBATCH --gres=gpu:1 # Use 1 GPUs
#SBATCH -p gpu

marco_pair_dir=/bos/tmp16/luyug/data/marco/pt-pair-2M
save_dir=/bos/tmp16/luyug/outputs/multibert/models/10-anchor

set -e
python run_retrieval.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name marco \
  --do_train \
  --do_lower_case \
  --data_dir $marco_pair_dir \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --output_dir $save_dir \
  --save_steps 1000 \
  --logging_steps 1000 \
  --evaluate_during_training \
  --warmup_steps 1000 \
  --n_anchors 10
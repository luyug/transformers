#!/bin/bash
#SBATCH -c 4 # Number of cores
#SBATCH -N 1 # 1 node requested
#SBATCH -n 1 # 1 task requested
#SBATCH --mem=96000 # Memory - Use 32G
#SBATCH --time=6:00:00 # No time limit
#SBATCH -o score-ret.slurm
#SBATCH --gres=gpu:4 # Use 1 GPUs
#SBATCH -p gpu
test_pair_dir=/bos/tmp16/luyug/data/marco/test_pair/6980-no-filter
output_dir=/bos/tmp16/luyug/outputs/tmp/model
ranking_dir=/bos/tmp16/luyug/outputs/tmp/ranking

mkdir $ranking_dir

set -e
python run_reduced_bert.py \
  --model_type bert \
  --model_name_or_path $output_dir \
  --task_name fmarco \
  --do_score \
  --do_lower_case \
  --data_dir $test_pair_dir \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size 128 \
  --output_dir $output_dir

score_file_id=6980.all

cp $output_dir/scores.txt $ranking_dir/scores.${score_file_id}.txt
python build_tein.py \
  --score $ranking_dir/scores.${score_file_id}.txt \
  --pair $test_pair_dir/eval_qid_pid.tsv \
  --output $ranking_dir/${score_file_id}.teIn

bash helpers/tein_to_marco_eval.sh $ranking_dir/${score_file_id}.teIn
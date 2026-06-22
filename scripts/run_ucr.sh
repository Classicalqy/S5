#!/bin/bash
#SBATCH --job-name=ucr_s5_small
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate s5

UCR_SPLIT_MODE=${UCR_SPLIT_MODE:-standard}

for dataset in ucr-ecg5000-classification ucr-forda-classification ucr-wafer-classification; do
  for seed in 0 1 2 3 4; do
    for p in original original_no_D real_decay resonant_2x2 energy_shaped_2x2; do
      echo "Running dataset=$dataset split_mode=$UCR_SPLIT_MODE ssm_param=$p seed=$seed"

      python run_train.py \
        dataset=$dataset \
        dir_name=../data2 \
        ucr_split_mode=$UCR_SPLIT_MODE \
        model.ssm_param=$p \
        n_layers=2 \
        d_model=16 \
        ssm_size_base=64 \
        blocks=1 \
        activation_fn=relu \
        batchnorm=False \
        layernorm=False \
        p_dropout=0.0 \
        bsz=64 \
        epochs=20 \
        seed=$seed
    done
  done
done

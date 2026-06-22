#!/bin/bash
#SBATCH --job-name=ucr_tdi_small
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate s5

for seed in 0 1 2 3 4; do
  for p in original original_no_D real_decay resonant_2x2 energy_shaped_2x2; do
    echo "Running ssm_param=$p seed=$seed"

    python run_train.py \
      dataset=synthetic_frequency-classification \
      model.ssm_param=$p \
      n_layers=2 \
      d_model=8 \
      ssm_size_base=16 \
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

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

for p in original original_no_D real_decay resonant_2x2 energy_shaped_2x2; do
  python run_train.py \
    dataset=synthetic_frequency-classification \
    model.ssm_param=$p \
    n_layers=2 \
    d_model=64 \
    ssm_size_base=128 \
    blocks=1 \
    batchnorm=False \
    p_dropout=0.1 \
    bsz=64 \
    epochs=50 \
    seed=0
done

#!/bin/bash
#SBATCH --job-name=mnist_variation_sweep
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate s5

python -m spice.digital_variation_test \
  --params "checkpoints/mnist_resonant_2x2_seed*_params_calibrated.msgpack" \
           "checkpoints/mnist_resonant_2x2_seed*_params_variation_aware.msgpack" \
  --out-dir out/mnist_variation_sweep \
  --dataset mnist-classification \
  --ssm-param resonant_2x2 \
  --sample-rate 160000 \
  --n-layers 2 \
  --d-model 16 \
  --ssm-size-base 64 \
  --blocks 1 \
  --mode last \
  --use-residual False \
  --batchnorm False \
  --activation-fn relu \
  --layernorm False \
  --p-dropout 0.0 \
  --bsz 64 \
  --c-min 1e-12 \
  --c-max 1e-9 \
  --variation-sigma 0 0.005 0.01 0.02 \
  --variation-seed 0 1 2 3 4

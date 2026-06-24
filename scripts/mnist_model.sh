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

for seed in 0; do
  for p in resonant_2x2; do
    echo "Running ssm_param=$p seed=$seed"

    python run_train.py \
      dataset=mnist-classification \
      model.ssm_param=$p \
      n_layers=2 \
      d_model=16 \
      ssm_size_base=64 \
      blocks=1 \
      mode=last \
      use_residual=False \
      batchnorm=False \
      activation_fn=relu \
      layernorm=False \
      p_dropout=0.0 \
      bsz=64 \
      epochs=20 \
      seed=$seed \
      save_params=True \
      params_out=checkpoints/mnist_${p}_seed${seed}_params.msgpack
  done
done

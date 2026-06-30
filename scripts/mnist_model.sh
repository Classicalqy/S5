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
  p=resonant_2x2
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
    hw_calibrate_readout=True \
    hw_calibrate_mode=analog \
    hw_calibrate_epochs=5 \
    hw_calibrate_lr=1e-4 \
    hw_c_max=1e-8 \
    hw_quant_bits=0 \
    params_out=checkpoints/mnist_${p}_seed${seed}_params.msgpack
done

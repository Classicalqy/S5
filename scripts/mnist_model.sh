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

p=resonant_2x2
for seed in 0 1 2 3 4; do
  base=checkpoints/mnist_${p}_seed${seed}_params
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
    hw_sample_rate=160000 \
    hw_c_min=1e-12 \
    hw_c_max=1e-9 \
    hw_variation_sigma=0.0 \
    hw_variation_aware_epochs=8 \
    hw_variation_aware_sigma=0.075 \
    hw_variation_aware_sigma_schedule=0.03,0.05,0.05,0.075,0.075,0.075,0.075,0.075 \
    hw_variation_aware_seed=$seed \
    hw_variation_aware_train_samples=8 \
    hw_variation_aware_eval_samples=8 \
    hw_variation_aware_select_sigma=0.05 \
    hw_variation_aware_nominal_fraction=0.1 \
    hw_variation_aware_select_metric=mean_std_strong \
    hw_variation_aware_loss=physical_noise_cvar \
    hw_variation_aware_consistency_weight=0.5 \
    hw_variation_aware_cvar_fraction=0.5 \
    hw_train_noise_sigma=0.05 \
    hw_train_noise_sigma_schedule=0,0,0,0,0,0.01,0.01,0.02,0.02,0.03,0.03,0.04,0.04,0.05,0.05,0.05,0.05,0.05,0.05,0.05 \
    hw_train_noise_samples=4 \
    hw_train_noise_consistency_weight=0.5 \
    hw_train_noise_cvar_fraction=0.5 \
    params_out=${base}.msgpack \
    hw_calibrated_params_out=${base}_calibrated.msgpack \
    hw_variation_aware_params_out=${base}_variation_aware.msgpack
done

echo "Running MNIST variation sweep"
python -m spice.digital_variation_test \
  --params "checkpoints/mnist_${p}_seed*_params_calibrated.msgpack" \
           "checkpoints/mnist_${p}_seed*_params_variation_aware.msgpack" \
  --out-dir out/mnist_variation_sweep \
  --dataset mnist-classification \
  --ssm-param "$p" \
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
  --variation-sigma 0 0.005 0.01 0.02 0.05 0.075 0.10 \
  --variation-seed 0 1 2 3 4

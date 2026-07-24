#!/usr/bin/env bash
set -euo pipefail

# Override with, for example:
#   SSM_PARAM=real_decay SEED=1 ./scripts/run_speech10.sh
SSM_PARAM="${SSM_PARAM:-resonant_2x2}"
SEED="${SEED:-0}"

python run_train.py \
  --dataset=speech10-classification \
  --ssm_param="${SSM_PARAM}" \
  --use_D=False \
  --C_init=lecun_normal \
  --batchnorm=True \
  --bidirectional=True \
  --blocks=16 \
  --bsz=16 \
  --d_model=96 \
  --epochs=40 \
  --jax_seed="${SEED}" \
  --lr_factor=4 \
  --n_layers=6 \
  --opt_config=noBCdecay \
  --p_dropout=0.1 \
  --ssm_lr_base=0.002 \
  --ssm_size_base=128 \
  --warmup_end=1 \
  --weight_decay=0.04

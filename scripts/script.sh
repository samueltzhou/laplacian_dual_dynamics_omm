#!/usr/bin/env bash
# run_gridroom.sh

for run in {4..10}; do
  CUDA_VISIBLE_DEVICES=0 \
  python train_laprepr.py \
    eff_omm_seq_FROBENIUS_FIXED_7_27_25_dim_50_lr_2e-3_n_samples_5000000_GridRoom-16_run_${run} \
    --config_file omm.yaml \
    --use_wandb \
    --env_name GridRoom-16
done
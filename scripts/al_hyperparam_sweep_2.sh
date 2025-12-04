#!/usr/bin/env bash
NAME=allo_hyperparam_sweep_7_28_25_dim_50_n_samples_5000000
ENV_NAME="GridRoom-16"

echo "DEBUG: Running hyperparameter sweep for ALLO on ${ENV_NAME}"

barrier_initial_vals=(1.0 2.0)
lr_barrier_coefs=(0.05 0.1 0.2)

for barrier_initial_val in "${barrier_initial_vals[@]}"; do
  for lr_barrier_coef in "${lr_barrier_coefs[@]}"; do
    run_name="${NAME}_${ENV_NAME}_barrier_${barrier_initial_val}_barrier_lr_${lr_barrier_coef}"
    CUDA_VISIBLE_DEVICES=1 python train_laprepr.py "${run_name}" \
      --config_file al.yaml \
      --use_wandb \
      --env_name "${ENV_NAME}" \
      --barrier_initial_val "${barrier_initial_val}" \
      --lr_barrier_coefs "${lr_barrier_coef}"
  done
done


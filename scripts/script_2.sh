NAME=eff_omm_seq_FROBENIUS_FIXED_7_28_25_dim_11_lr_2e-3_n_samples_5000000_layernorm_enabled

for env in GridMaze-26 GridMaze-32 GridRoom-32 GridRoom-64; do
  CUDA_VISIBLE_DEVICES=0 \
  python train_laprepr.py \
    ${NAME}_${env} \
    --config_file omm.yaml \
    --use_wandb \
    --env_name ${env} \
    --lr 0.002
done
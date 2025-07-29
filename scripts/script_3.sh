NAME=eff_omm_seq_FROBENIUS_FIXED_7_28_25_dim_50_lr_2e-3_n_samples_5000000_layernorm_off
RUN_NUMBER=1

envs=(GridMaze-17 GridMaze-19 GridMaze-26 GridMaze-32 GridRoom-1 GridRoom-4 GridRoom-16 GridRoom-32 GridRoom-64 GridRoomSym-4)

for env in "${envs[@]}"; do
  CUDA_VISIBLE_DEVICES=0 \
  python train_laprepr.py \
    ${NAME}_${env}_run_${RUN_NUMBER} \
    --config_file omm.yaml \
    --use_wandb \
    --env_name ${env} \
    --lr 0.002
done
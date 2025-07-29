NAME=allo_7_29_25_dim_50_n_samples_5000000_default_barrier_hyperparams
RUN_NUMBER=2
envs=(GridMaze-17 GridMaze-19 GridMaze-26 GridMaze-32 GridRoom-1 GridRoom-4 GridRoom-16 GridRoom-32 GridRoom-64 GridRoomSym-4)

for env in "${envs[@]}"; do
  CUDA_VISIBLE_DEVICES=1 \
  python train_laprepr.py \
    ${NAME}_${env}_run_${RUN_NUMBER} \
    --config_file al.yaml \
    --use_wandb \
    --env_name ${env} \
    --barrier_initial_val 2.0 \
    --lr_barrier_coefs 1.0
done

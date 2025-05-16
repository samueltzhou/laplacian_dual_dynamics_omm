NAME=joint_omm_1.0_shift_5_15_25

echo "DEBUG: NAME is '${NAME}'"

python train_laprepr.py gridroom_32_${NAME}_run_1 --config_file omm_joint.yaml --use_wandb --env_name GridRoom-32
python train_laprepr.py gridroom_32_${NAME}_run_2 --config_file omm_joint.yaml --use_wandb --env_name GridRoom-32
python train_laprepr.py gridroom_32_${NAME}_run_3 --config_file omm_joint.yaml --use_wandb --env_name GridRoom-32
python train_laprepr.py gridroom_32_${NAME}_run_4 --config_file omm_joint.yaml --use_wandb --env_name GridRoom-32
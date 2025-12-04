NAME=joint_omm_0.5_shift_5_14_25

echo "DEBUG: NAME is '${NAME}'"
echo "DEBUG: Constructing exp_label as grid_x_${NAME}_run_{num}"

# first block of runs
python train_laprepr.py gridmaze_7_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridMaze-7
python train_laprepr.py gridmaze_9_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridMaze-9
python train_laprepr.py gridmaze_17_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridMaze-17
python train_laprepr.py gridmaze_19_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridMaze-19
python train_laprepr.py gridmaze_26_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridMaze-26
python train_laprepr.py gridmaze_32_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridMaze-32
python train_laprepr.py gridroom_1_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridRoom-1
python train_laprepr.py gridroom_4_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridRoom-4
python train_laprepr.py gridroom_16_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridRoom-16
python train_laprepr.py gridroom_32_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridRoom-32
python train_laprepr.py gridroom_64_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridRoom-64
python train_laprepr.py gridroomsym_4_${NAME}_run_1 --config_file omm.yaml --use_wandb --env_name GridRoomSym-4

# second block of runs
python train_laprepr.py gridmaze_7_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridMaze-7
python train_laprepr.py gridmaze_9_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridMaze-9
python train_laprepr.py gridmaze_17_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridMaze-17
python train_laprepr.py gridmaze_19_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridMaze-19
python train_laprepr.py gridmaze_26_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridMaze-26
python train_laprepr.py gridmaze_32_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridMaze-32
python train_laprepr.py gridroom_1_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridRoom-1
python train_laprepr.py gridroom_4_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridRoom-4
python train_laprepr.py gridroom_16_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridRoom-16
python train_laprepr.py gridroom_32_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridRoom-32
python train_laprepr.py gridroom_64_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridRoom-64
python train_laprepr.py gridroomsym_4_${NAME}_run_2 --config_file omm.yaml --use_wandb --env_name GridRoomSym-4

# third block of runs
python train_laprepr.py gridmaze_7_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridMaze-7
python train_laprepr.py gridmaze_9_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridMaze-9
python train_laprepr.py gridmaze_17_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridMaze-17
python train_laprepr.py gridmaze_19_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridMaze-19
python train_laprepr.py gridmaze_26_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridMaze-26
python train_laprepr.py gridmaze_32_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridMaze-32
python train_laprepr.py gridroom_1_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridRoom-1
python train_laprepr.py gridroom_4_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridRoom-4
python train_laprepr.py gridroom_16_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridRoom-16
python train_laprepr.py gridroom_32_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridRoom-32
python train_laprepr.py gridroom_64_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridRoom-64
python train_laprepr.py gridroomsym_4_${NAME}_run_3 --config_file omm.yaml --use_wandb --env_name GridRoomSym-4

# fourth block of runs
python train_laprepr.py gridmaze_7_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridMaze-7
python train_laprepr.py gridmaze_9_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridMaze-9
python train_laprepr.py gridmaze_17_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridMaze-17
python train_laprepr.py gridmaze_19_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridMaze-19
python train_laprepr.py gridmaze_26_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridMaze-26
python train_laprepr.py gridmaze_32_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridMaze-32
python train_laprepr.py gridroom_1_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridRoom-1
python train_laprepr.py gridroom_4_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridRoom-4
python train_laprepr.py gridroom_16_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridRoom-16
python train_laprepr.py gridroom_32_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridRoom-32
python train_laprepr.py gridroom_64_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridRoom-64
python train_laprepr.py gridroomsym_4_${NAME}_run_4 --config_file omm.yaml --use_wandb --env_name GridRoomSym-4

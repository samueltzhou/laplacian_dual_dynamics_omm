NAME=allo_5_9_25
NUMBER=1

echo "DEBUG: NAME is '${NAME}'"
echo "DEBUG: NUMBER is '${NUMBER}'"
echo "DEBUG: Constructing exp_label as gridmaze_7_${NAME}_run_${NUMBER}"

nohup python train_laprepr.py gridmaze_7_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridMaze-7
nohup python train_laprepr.py gridmaze_9_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridMaze-9
nohup python train_laprepr.py gridmaze_17_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridMaze-17
nohup python train_laprepr.py gridmaze_19_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridMaze-19
nohup python train_laprepr.py gridmaze_26_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridMaze-26
nohup python train_laprepr.py gridmaze_32_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridMaze-32
nohup python train_laprepr.py gridroom_1_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridRoom-1
nohup python train_laprepr.py gridroom_4_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridRoom-4
nohup python train_laprepr.py gridroom_16_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridRoom-16
nohup python train_laprepr.py gridroom_32_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridRoom-32
nohup python train_laprepr.py gridroom_64_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridRoom-64
nohup python train_laprepr.py gridroomsym_4_${NAME}_run_${NUMBER} --config_file al.yaml --use_wandb --env_name GridRoomSym-4



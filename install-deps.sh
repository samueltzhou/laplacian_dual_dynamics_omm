python -m venv venv
source venv/bin/activate
export WANDB_API_KEY=5bafe94a7c3be779b0cdb6ea86a28283984fe95a

pip install jax
pip install dm-haiku
pip install numpy
pip install pickle
pip install collections
pip install gymnasium
pip install typing
pip install pathlib
pip install equinox
pip install optax
pip install scipy
pip install pyyaml
pip install matplotlib
pip install wandb
pip install mpmath
pip install pygame

pip freeze > requirements.txt
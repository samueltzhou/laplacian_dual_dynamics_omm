from collections import namedtuple
from typing import Tuple
from itertools import product
from typing_extensions import override
import numpy as np
import jax
import jax.numpy as jnp
import math

import equinox as eqx

from src.trainer.laplacian_encoder import LaplacianEncoderTrainer

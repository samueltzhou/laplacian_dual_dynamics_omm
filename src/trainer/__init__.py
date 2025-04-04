from src.trainer.trainer import Trainer
from src.trainer.laplacian_encoder import LaplacianEncoderTrainer 
from src.trainer.generalized_gdo import GeneralizedGraphDrawingObjectiveTrainer
from src.trainer.generalized_augmented import GeneralizedAugmentedLagrangianTrainer
from src.trainer.al import AugmentedLagrangianTrainer
from src.trainer.quadratic_penalty_ggdo import QuadraticPenaltyGGDOTrainer
from src.trainer.sqp import StopGradientQuadraticPenaltyTrainer as SQPTrainer
from src.trainer.cqp import CoefficientSymmetryBreakingQuadraticPenaltyTrainer as CQPTrainer
from src.trainer.low_rank import LowRankObjectiveTrainer as LoRATrainer
from functools import partial
from src.trainer.low_rank import LowRankObjectiveTrainer

# Create a specialized version of LowRankObjectiveTrainer with orbital_enabled=True for OMM
OMMTrainer = partial(LowRankObjectiveTrainer, orbital_enabled=True)
from src.trainer.trainer import Trainer
from src.trainer.laplacian_encoder import LaplacianEncoderTrainer
from src.trainer.generalized_gdo import GeneralizedGraphDrawingObjectiveTrainer
from src.trainer.generalized_augmented import GeneralizedAugmentedLagrangianTrainer
from src.trainer.al import AugmentedLagrangianTrainer
from src.trainer.quadratic_penalty_ggdo import QuadraticPenaltyGGDOTrainer
from src.trainer.sqp import StopGradientQuadraticPenaltyTrainer as SQPTrainer
from src.trainer.cqp import (
    CoefficientSymmetryBreakingQuadraticPenaltyTrainer as CQPTrainer,
)
from src.trainer.low_rank import JointLowRankObjectiveTrainer as JointLoRATrainer
from src.trainer.seq_low_rank import (
    SequentialLowRankObjectiveTrainer as SequentialLoRATrainer,
)
from src.trainer.eff_seq_low_rank import (
    EfficientSequentialLowRankObjectiveTrainer as EfficientSequentialLoRATrainer,
)
from src.trainer.lora_and_omm import (
    CombinedLowRankObjectiveTrainer as CombinedLoRATrainer,
)
from src.trainer.high_order_omm import (
    HighOrderSequentialOMMLossTrainer as HighOrderSequentialOMMTrainer,
)
from src.trainer.mixed_order_seq_omm import (
    MixedOrderSequentialOMMLossTrainer as MixedOrderSequentialOMMTrainer,
)
from functools import partial

# Create a specialized version of LowRankObjectiveTrainer with orbital_enabled=True for OMM
JointOMMTrainer = partial(JointLoRATrainer, orbital_enabled=True)
SequentialOMMTrainer = partial(SequentialLoRATrainer, orbital_enabled=True)
EfficientSequentialOMMTrainer = partial(
    EfficientSequentialLoRATrainer, orbital_enabled=True
)

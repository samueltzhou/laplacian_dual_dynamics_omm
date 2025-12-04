from collections import namedtuple
from typing import Tuple
from itertools import product
from typing_extensions import override
import numpy as np
import jax
import jax.numpy as jnp

import equinox as eqx

from src.trainer.laplacian_encoder import LaplacianEncoderTrainer
from src.trainer.constants import *

MC_sample_expanded = namedtuple(
    "MC_sample_expanded",
    "state future_state uncorrelated_state_1 uncorrelated_state_2 state2 future_state2",
)


class JointSquaredOMMObjectiveTrainer(LaplacianEncoderTrainer):
    """
    Frobenius norm gain is disabled for this. Gain is pegged at 1, and bias also behaves a bit differently:
    bias is applied as A --> (A + bias)^2 to shift eigvals to > 1.

    -2 (AV)^T (AV) - 4b V^T AV + V^TV (AV)^T (AV) + 2b V^TV V^T AV + b^2 (V^TV - I)_2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def _get_train_batch(self):
        # Sample two pairs of states for contrastive learning and two sets of uncorrelated states.
        states_to_sample = [
            lambda: self.replay_buffer.sample_pairs(
                batch_size=self.batch_size, discount=self.discount
            ),
            lambda: self.replay_buffer.sample_pairs(
                batch_size=self.batch_size, discount=self.discount
            ),
            lambda: (self.replay_buffer.sample_steps(self.batch_size),),
            lambda: (self.replay_buffer.sample_steps(self.batch_size),),
        ]

        # Unpack samples into a flat list
        samples = [item for sample_fn in states_to_sample for item in sample_fn()]

        # Process all states with _get_obs_batch
        processed_states = map(self._get_obs_batch, samples)

        return MC_sample_expanded(*processed_states)

    @override
    def encode_states(
        self,
        params_encoder,
        train_batch: MC_sample_expanded,
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray]:

        # The order of states here is important and corresponds to the structure of MC_sample_expanded
        states_to_encode = [
            train_batch.state,
            train_batch.future_state,
            train_batch.uncorrelated_state_1,
            train_batch.uncorrelated_state_2,
            train_batch.state2,
            train_batch.future_state2,
        ]

        representations = [
            self.encoder_fn.apply(params_encoder, state) for state in states_to_encode
        ]

        permuted_representations = [
            self.permute_representations(rep) for rep in representations
        ]

        return tuple(permuted_representations)

    def compute_frobenius_norm_loss(self, representation, matrix_mask):
        """
        Computes the Frobenius norm loss between the representation and the identity matrix.

        This is ||rep.T @ rep - I||_2. Bias factor removed, bias is to be multiplied in during the
        actual loss computation.
        """
        return jnp.sum(
            matrix_mask
            * (
                representation.T @ representation / representation.shape[0]
                - jnp.eye(representation.shape[1])
            )
            ** 2
        )

    def loss_function(self, params, train_batch, **kwargs) -> Tuple[jnp.ndarray]:
        # start_representation and end_representation are correlated (sampled from an edge)
        # constraint_start_representation and constraint_end_representation are uncorrelated (iid sampled from states)

        # Get representations
        (
            start_representation,
            end_representation,
            constraint_start_representation,
            constraint_end_representation,
            start_representation_2,
            end_representation_2,
        ) = self.encode_states(params["encoder"], train_batch)

        print("Shapes of representations in batch: ")
        print(f"start_representation: {start_representation.shape}")
        print(f"end_representation: {end_representation.shape}")
        print(
            f"constraint_start_representation: {constraint_start_representation.shape}"
        )
        print(f"constraint_end_representation: {constraint_end_representation.shape}")
        print(f"start_representation_2: {start_representation_2.shape}")
        print(f"end_representation_2: {end_representation_2.shape}")

        # Create the mask in a vectorized way
        coeff_vector_mask = jnp.arange(self.d, 0, -1)
        coeff_matrix_mask = jnp.minimum(
            jnp.expand_dims(coeff_vector_mask, 1), jnp.expand_dims(coeff_vector_mask, 0)
        )  # Shape: (d, d)

        # A^2 loss adjustment
        # -2 (AV)^T (AV) - 4b V^T AV + V^TV (AV)^T (AV) + 2b V^TV V^T AV + b^2 (V^TV - I)_2
        # -2 trace((AV)^T (AV)) - 4b trace(V^T AV) + trace(V^TV (AV)^T (AV)) + 2b trace(V^TV V^TAV) + b^2 || V^TV - I ||_2
        cov_matrix_start = (
            start_representation.T
            @ start_representation
            / start_representation.shape[0]
        )  # [d, d]
        cov_matrix_end = (
            end_representation.T @ end_representation / end_representation.shape[0]
        )  # [d, d]

        corr_matrix = (
            start_representation.T @ end_representation / start_representation.shape[0]
        )  # [d, d]
        corr_matrix_2 = (
            start_representation_2.T
            @ end_representation_2
            / start_representation_2.shape[0]
        )  # [d, d]

        frobenius_norm_loss = self.compute_frobenius_norm_loss(
            constraint_start_representation,
            coeff_matrix_mask,
        )

        loss = (
            -2 * jnp.trace(coeff_matrix_mask * cov_matrix_end)
            - 4 * FROBENIUS_NORM_BIAS * jnp.trace(corr_matrix)
            + jnp.sum(cov_matrix_start * cov_matrix_end)
            + 2 * FROBENIUS_NORM_BIAS * jnp.sum(cov_matrix_start * corr_matrix_2)
            + FROBENIUS_NORM_BIAS**2
            * frobenius_norm_loss
        )

        metrics_dict = {
            "train_loss": loss,
        }
        metrics = (
            loss,
            0.0,
            0.0,
            frobenius_norm_loss,
            metrics_dict,
        )
        aux = (metrics, None)

        return loss, aux

    # note: the rest are from generalized_gdo.py
    # no difference between non-permuted and permuted loss functions; TODO: check this
    def loss_function_non_permuted(self, *args, **kwargs):
        return self.loss_function(*args, **kwargs)

    def loss_function_permuted(self, *args, **kwargs):
        return self.loss_function(*args, **kwargs)

    def update_training_state(self, params, *args, **kwargs):
        """Leave params unchanged"""

        return params

    def additional_update_step(self, step, params, *args, **kwargs):
        """Leave params unchanged"""

        return params

    def init_additional_params(self, *args, **kwargs):
        additional_params = {}
        return additional_params


class JointOMMObjectiveTrainerV2(LaplacianEncoderTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def _get_train_batch(self):
        state, future_state = self.replay_buffer.sample_pairs(
            batch_size=self.batch_size,
            discount=self.discount,
        )
        state2, future_state2 = self.replay_buffer.sample_pairs(
            batch_size=self.batch_size,
            discount=self.discount,
        )  # orbital
        uncorrelated_state_1 = self.replay_buffer.sample_steps(self.batch_size)
        uncorrelated_state_2 = self.replay_buffer.sample_steps(self.batch_size)
        (
            state,
            future_state,
            uncorrelated_state_1,
            uncorrelated_state_2,
            state2,
            future_state2,
        ) = map(
            self._get_obs_batch,
            [
                state,
                future_state,
                uncorrelated_state_1,
                uncorrelated_state_2,
                state2,
                future_state2,
            ],
        )
        batch = MC_sample_expanded(
            state,
            future_state,
            uncorrelated_state_1,
            uncorrelated_state_2,
            state2,
            future_state2,
        )
        return batch

    @override
    def encode_states(
        self,
        params_encoder,
        train_batch: MC_sample_expanded,
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray]:
        # Compute start representations
        start_representation = self.encoder_fn.apply(params_encoder, train_batch.state)
        constraint_representation_1 = self.encoder_fn.apply(
            params_encoder, train_batch.uncorrelated_state_1
        )

        # Compute end representations
        end_representation = self.encoder_fn.apply(
            params_encoder, train_batch.future_state
        )
        constraint_representation_2 = self.encoder_fn.apply(
            params_encoder, train_batch.uncorrelated_state_2
        )

        # compute extra pair of start and end representations
        start_representation_2 = self.encoder_fn.apply(
            params_encoder, train_batch.state
        )
        end_representation_2 = self.encoder_fn.apply(
            params_encoder, train_batch.future_state
        )

        # Permute representations
        start_representation = self.permute_representations(start_representation)
        end_representation = self.permute_representations(end_representation)
        constraint_representation_1 = self.permute_representations(
            constraint_representation_1
        )
        constraint_representation_2 = self.permute_representations(
            constraint_representation_2
        )
        # for orbital
        start_representation_2 = self.permute_representations(start_representation_2)
        end_representation_2 = self.permute_representations(end_representation_2)

        return (
            start_representation,
            end_representation,
            constraint_representation_1,
            constraint_representation_2,
            start_representation_2,  # for orbitak
            end_representation_2,  # for orbital
        )

    def compute_frobenius_norm_loss(self, representation, matrix_mask):
        """
        Computes the Frobenius norm loss between the representation and the identity matrix.

        This is ||rep.T @ rep - I||_2. Bias factor removed, bias is to be multiplied in during the
        actual loss computation.
        """
        return jnp.sum(
            matrix_mask
            * (
                representation.T @ representation / representation.shape[0]
                - jnp.eye(representation.shape[1])
            )
            ** 2
        )

    def loss_function(self, params, train_batch, **kwargs) -> Tuple[jnp.ndarray]:
        # start_representation and end_representation are correlated (sampled from an edge)
        # constraint_start_representation and constraint_end_representation are uncorrelated (iid sampled from states)

        # Get representations
        (
            start_representation,
            end_representation,
            constraint_start_representation,
            constraint_end_representation,
            start_representation_2,
            end_representation_2,
        ) = self.encode_states(params["encoder"], train_batch)

        print("Shapes of representations in batch: ")
        print(f"start_representation: {start_representation.shape}")
        print(f"end_representation: {end_representation.shape}")
        print(
            f"constraint_start_representation: {constraint_start_representation.shape}"
        )
        print(f"constraint_end_representation: {constraint_end_representation.shape}")
        print(f"start_representation_2: {start_representation_2.shape}")
        print(f"end_representation_2: {end_representation_2.shape}")

        # Create the mask in a vectorized way
        coeff_vector_mask = jnp.arange(self.d, 0, -1)
        coeff_matrix_mask = jnp.minimum(
            jnp.expand_dims(coeff_vector_mask, 1), jnp.expand_dims(coeff_vector_mask, 0)
        )  # Shape: (d, d)

        # V^T V
        cov_matrix = (
            constraint_start_representation.T
            @ constraint_start_representation
            / constraint_start_representation.shape[0]
        )  # [d, d]

        # V^T AV
        corr_matrix = (
            start_representation.T @ end_representation / start_representation.shape[0]
        )  # [d, d]
        corr_matrix_2 = (
            start_representation_2.T
            @ end_representation_2
            / start_representation_2.shape[0]
        )  # [d, d]
        scaled_corr_matrix = coeff_matrix_mask * corr_matrix
        scaled_corr_matrix_2 = coeff_matrix_mask * corr_matrix_2

        # loss: trace(coeff_matrix_mask * (-2 corr_matrix + cov_matrix @ corr_matrix))
        raw_loss = -2 * jnp.trace(scaled_corr_matrix) + jnp.sum(
            cov_matrix * scaled_corr_matrix_2
        )

        # shift term (decorrelate this too)
        # huh do we just apply matrix mask naively to frobenius norm loss?
        frobenius_norm_loss = self.compute_frobenius_norm_loss(
            constraint_end_representation,
            coeff_matrix_mask,
        )

        # Compute total loss. NOTE: WITH NON-DEFAULT BIAS AND GAIN, THIS HAS TO BE WITH ORBITALS
        loss = (
            FROBENIUS_NORM_GAIN * raw_loss + FROBENIUS_NORM_BIAS * frobenius_norm_loss
        )

        metrics_dict = {
            "train_loss": loss,
            "raw_loss": raw_loss,
            "frobenius_norm_loss": frobenius_norm_loss,
        }
        metrics = (
            loss,
            raw_loss,
            0.0,
            frobenius_norm_loss,
            metrics_dict,
        )
        aux = (metrics, None)

        return loss, aux

    # note: the rest are from generalized_gdo.py
    # no difference between non-permuted and permuted loss functions; TODO: check this
    def loss_function_non_permuted(self, *args, **kwargs):
        return self.loss_function(*args, **kwargs)

    def loss_function_permuted(self, *args, **kwargs):
        return self.loss_function(*args, **kwargs)

    def update_training_state(self, params, *args, **kwargs):
        """Leave params unchanged"""

        return params

    def additional_update_step(self, step, params, *args, **kwargs):
        """Leave params unchanged"""

        return params

    def init_additional_params(self, *args, **kwargs):
        additional_params = {}
        return additional_params

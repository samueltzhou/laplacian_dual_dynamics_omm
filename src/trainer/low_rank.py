from collections import namedtuple
from typing import Tuple
from itertools import product
from typing_extensions import override
import numpy as np
import jax
import jax.numpy as jnp

import equinox as eqx

from src.trainer.laplacian_encoder import LaplacianEncoderTrainer

MC_sample_expanded = namedtuple(
    "MC_sample_expanded",
    "state future_state uncorrelated_state_1 uncorrelated_state_2 state2 future_state2",
)


class JointLowRankObjectiveTrainer(LaplacianEncoderTrainer):
    def __init__(self, orbital_enabled=False, *args, **kwargs):
        self.orbital_enabled = orbital_enabled
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

    def compute_approximation_error_loss(
        self, start_representation, end_representation
    ):
        """
        Compute the approximation error loss between the start and end representations.
        This is the -2 Sigma <g_l, Tf_l> term, but as we are working with EVD, it is just <f_l, Tf_l>.
        Recall from the derivation that this is equivalent to E[(f_l - Tf_l)^2], where we sample over
        transitions and take an expectation over p(x, x').

        Args:
            start_representation: The representation of the start state.
            end_representation: The representation of the end state.

        Returns:
            The approximation error loss.
        """
        print(
            f"for approximation error loss: start_representation: {start_representation.shape}"
        )
        print(
            f"for approximation error loss: end_representation: {end_representation.shape}"
        )
        print(f"we wish to learn the top {self.d} eigenvectors")
        # Compute the loss term
        # loss = -2 * ((start_representation - end_representation)**2).mean() # need to recheck this...
        loss = 0
        coeff_mask = jnp.arange(self.d, 0, -1)  # for joint LoRA
        coeff_mask = coeff_mask**2  # this is just a test, comment this out later
        if len(start_representation.shape) == 1 and len(end_representation.shape) == 1:
            print("one dimensional approximation error loss")

            # computing loss via squares
            squared_diff = (start_representation - end_representation) ** 2
            joint_squared_diff = squared_diff * coeff_mask
            loss = -(jnp.sum(joint_squared_diff))

            # computing loss via straight inner product
            inner_prod = jnp.dot(start_representation, end_representation)
            loss = -2 * inner_prod
        else:
            print("batched approximation error loss")
            # shapes (1024, 11) and (1024, 11)

            # computing loss via squares
            # squared_diff = (start_representation - end_representation) ** 2
            # joint_squared_diff = squared_diff * coeff_mask
            # loss = - jnp.mean(jnp.sum(joint_squared_diff, axis=1))

            # computing loss via straight inner product
            # inner_prod = jnp.einsum('bi, bi, i -> b', start_representation, end_representation, coeff_mask)
            # loss = -2 * jnp.mean(inner_prod)

            averaged_inner_prod = (
                jnp.einsum("bi, bi -> i", start_representation, end_representation)
                / start_representation.shape[0]
            )
            loss = -2 * jnp.sum(averaged_inner_prod * coeff_mask)

        # Normalize loss
        # if self.coefficient_normalization:
        #     loss = loss / (self.d * (self.d + 1) / 2)

        return loss

    def compute_orthogonality_loss(
        self, representation_1, representation_2, representation_2_end=None
    ):
        """
        Compute the orthogonality loss between the two representations.

        If we are working with LoRA:
            This is the Sigma_l Sigma_l' <f_l | f_l'> <g_l | g_l'> term.
            Since we are working with EVD, it is just Sigma_l <f_l | f_l'>^2.
        If we are working with OMM:
            This is Sigma_l Sigma_l' <f_l | Tf_l'> <f_l | f_l'> term.

        Args:
            representation_1: The first representation.
            representation_2: The second representation.

        Returns:
            The orthogonality loss.
        """
        loss = 0

        print(
            "we are using joint OMM"
            if self.orbital_enabled
            else "we are using joint LoRA"
        )
        print(f"for orthogonality loss: representation_1: {representation_1.shape}")
        print(f"for orthogonality loss: representation_2: {representation_2.shape}")
        print(
            f"for orthogonality loss: representation_2_end: {representation_2_end.shape}"
            if self.orbital_enabled
            else None
        )
        print(f"representation size: {representation_1.shape}")

        # Create the mask in a vectorized way
        coeff_vector_mask = jnp.arange(self.d, 0, -1)
        coeff_vector_mask = (
            coeff_vector_mask**2
        )  # this is just a test, comment this out later
        coeff_vector_mask_col = jnp.expand_dims(coeff_vector_mask, 1)  # Shape: (d, 1)
        coeff_vector_mask_row = jnp.expand_dims(coeff_vector_mask, 0)  # Shape: (1, d)
        coeff_matrix_mask = jnp.minimum(
            coeff_vector_mask_col, coeff_vector_mask_row
        )  # Shape: (d, d)

        if self.orbital_enabled:
            # OMM loss case: Sigma_l Sigma_l' <f_l | Tf_l'> <f_l | f_l'>
            try:
                if len(representation_1.shape) == 1:
                    print("one dimensional orthogonality loss")
                    product_1 = jnp.einsum(
                        "j, k -> jk", representation_2, representation_2_end
                    )
                    product_2 = jnp.einsum(
                        "j, k -> jk", representation_1, representation_1
                    )
                    loss += jnp.sum(coeff_matrix_mask * product_1 * product_2)
                else:
                    print("batched orthogonality loss")
                    product_1 = jnp.einsum(
                        "bj, bk -> bjk", representation_2, representation_2_end
                    )
                    averaged_product_1 = jnp.mean(product_1, axis=0)
                    product_2 = jnp.einsum(
                        "bj, bk -> bjk", representation_1, representation_1
                    )
                    averaged_product_2 = jnp.mean(product_2, axis=0)
                    loss += jnp.sum(
                        coeff_matrix_mask * averaged_product_1 * averaged_product_2
                    )
            except:
                print(f"Shape of representation_1: {representation_1.shape}")
                print(f"Shape of representation_2: {representation_2.shape}")
                raise
        else:
            # LoRA loss case: Sigma_l Sigma_l' <f_l | f_l'>^2
            # CHECK TO MAKE SURE WE DO A STRAIGHT UP DOT PRODUCT AND THAT WE DON'T NEED TO APPROXIMATE.
            try:
                # check dimensions to determine einsum or not
                if len(representation_1.shape) == 1:
                    print("one dimensional orthogonality loss")
                    pairwise_product = jnp.einsum(
                        "i, j -> ij", representation_1, representation_1
                    )
                    loss += jnp.sum(coeff_matrix_mask * (pairwise_product**2))
                else:
                    print("batched orthogonality loss")
                    pairwise_product_tensor = jnp.einsum(
                        "ij, ik -> ijk", representation_1, representation_1
                    )
                    averaged_over_batch = jnp.mean(pairwise_product_tensor, axis=0)
                    loss += jnp.sum(coeff_matrix_mask * (averaged_over_batch**2))
            except:
                print(f"Shape of representation_1: {representation_1.shape}")
                raise

        # # Normalize loss
        # if self.coefficient_normalization:
        #     loss = loss / (self.d * (self.d + 1) / 2)

        return loss

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

        # Compute graph loss and regularization
        approximation_error_loss = self.compute_approximation_error_loss(
            start_representation, end_representation
        )

        # NOTE: the start/end representations used by orthogonality loss should not be the same as
        # the ones used by the graph loss, may need to rework batching
        # NOTE: let's try doing it without vmap: don't think its necessary here
        if self.orbital_enabled:
            orthogonality_loss = self.compute_orthogonality_loss(
                constraint_start_representation,
                start_representation_2,
                representation_2_end=end_representation_2,
            )
        else:
            orthogonality_loss = self.compute_orthogonality_loss(
                constraint_start_representation,
                constraint_end_representation,
                representation_2_end=None,
            )

        # Compute total loss
        loss = approximation_error_loss + orthogonality_loss

        metrics_dict = {
            "train_loss": loss,
            "approximation_error_loss": approximation_error_loss,
            "orthogonality_loss": orthogonality_loss,
        }
        metrics = (
            loss,
            approximation_error_loss,
            0.0,
            orthogonality_loss,
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

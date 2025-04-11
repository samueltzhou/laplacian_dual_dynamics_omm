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


class SequentialLowRankObjectiveTrainer(LaplacianEncoderTrainer):
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
        )
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
        start_representation_2 = self.permute_representations(start_representation_2)
        end_representation_2 = self.permute_representations(end_representation_2)

        return (
            start_representation,
            end_representation,
            constraint_representation_1,
            constraint_representation_2,
            start_representation_2,
            end_representation_2,
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
        if len(start_representation.shape) == 1 and len(end_representation.shape) == 1:
            print("one dimensional approximation error loss")

            # computing loss via squares
            squared_diff = (start_representation - end_representation) ** 2
            joint_squared_diff = squared_diff
            loss = -(jnp.sum(joint_squared_diff))

            # computing loss via straight inner product
            inner_prod = jnp.dot(start_representation, end_representation)
            loss = -2 * inner_prod
        else:
            print("batched approximation error loss")
            # shapes (1024, 11) and (1024, 11)

            averaged_inner_prod = (
                jnp.einsum("bi, bi -> i", start_representation, end_representation)
                / start_representation.shape[0]
            )
            loss = -2 * jnp.sum(averaged_inner_prod)

        # Normalize loss
        if self.coefficient_normalization:
            loss = loss / (self.d * (self.d + 1) / 2)

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
            "we are using sequential OMM"
            if self.orbital_enabled
            else "we are using sequential LoRA"
        )
        print(f"for orthogonality loss: representation_1: {representation_1.shape}")
        print(f"for orthogonality loss: representation_2: {representation_2.shape}")
        print(
            f"for orthogonality loss: representation_2_end: {representation_2_end.shape}"
            if self.orbital_enabled
            else None
        )
        print(f"representation size: {representation_1.shape}")

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
                    loss += jnp.sum(product_1 * product_2)
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
                    loss += jnp.sum(averaged_product_1 * averaged_product_2)
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
                    loss += jnp.sum(pairwise_product**2)
                else:
                    print("batched orthogonality loss")
                    pairwise_product_tensor = jnp.einsum(
                        "ij, ik -> ijk", representation_1, representation_1
                    )
                    averaged_over_batch = jnp.mean(pairwise_product_tensor, axis=0)
                    loss += jnp.sum(averaged_over_batch**2)
            except:
                print(f"Shape of representation_1: {representation_1.shape}")
                raise

        # Normalize loss
        if self.coefficient_normalization:
            loss = loss / (self.d * (self.d + 1) / 2)

        return loss

    def _build_stop_grad_encoding(
        self, encoding: jnp.ndarray, top_i: int, **kwargs
    ) -> jnp.ndarray:
        """
        Builds a top_i state encoding where everything but the top_ith eigenfunction is frozen.

        Args:
            encodings: The state encoding.
            top_i: How many of the top eigenfunctions to use for the loss function.

        Returns:
            The top_i state encoding.
        """
        # encodings are either of shape (d,) or (b, d)
        if len(encoding.shape) == 1:
            mask_function = lambda encoding: jnp.concatenate(
                [
                    jax.lax.stop_gradient(encoding[: top_i - 1]),
                    encoding[top_i - 1 : top_i],
                ],
                axis=0,
            )
        else:
            mask_function = lambda encoding: jnp.concatenate(
                [
                    jax.lax.stop_gradient(encoding[:, : top_i - 1]),
                    encoding[:, top_i - 1 : top_i],
                ],
                axis=1,
            )

        return mask_function(encoding)

    def loss_function(self, params, train_batch, **kwargs) -> Tuple[jnp.ndarray]:
        # Get representations
        (
            start_representation,
            end_representation,
            constraint_start_representation,
            constraint_end_representation,
            start_representation_2,
            end_representation_2,
        ) = self.encode_states(params["encoder"], train_batch)
        # start_representation and end_representation are correlated (sampled from an edge)
        # so are start_representation_2 and end_representation_2
        # constraint_start_representation and constraint_end_representation are uncorrelated (iid sampled from states)

        def _compute_loss_function_component(top_i, **kwargs) -> Tuple[jnp.ndarray]:
            """
            Computes a single component of the loss function (the top_i loss).

            Args:
                params: The parameters of the model.
                state_encoding: The state encoding.
                top_i: How many of the top eigenfunctions to use for the loss function.

            Returns:
                Tuple of (loss, approximation_error_loss, orthogonality_loss)
            """

            approximation_error_loss = self.compute_approximation_error_loss(
                self._build_stop_grad_encoding(start_representation, top_i),
                self._build_stop_grad_encoding(end_representation, top_i),
            )
            orthogonality_loss = self.compute_orthogonality_loss(
                self._build_stop_grad_encoding(constraint_start_representation, top_i),
                (
                    self._build_stop_grad_encoding(start_representation_2, top_i)
                    if self.orbital_enabled
                    else self._build_stop_grad_encoding(
                        constraint_end_representation, top_i
                    )
                ),
                representation_2_end=(
                    self._build_stop_grad_encoding(end_representation_2, top_i)
                    if self.orbital_enabled
                    else None
                ),
            )
            loss = approximation_error_loss + orthogonality_loss

            return loss, approximation_error_loss, orthogonality_loss

        print("Shapes of representations in batch: ")
        print(f"start_representation: {start_representation.shape}")
        print(f"end_representation: {end_representation.shape}")
        print(
            f"constraint_start_representation: {constraint_start_representation.shape}"
        )
        print(f"constraint_end_representation: {constraint_end_representation.shape}")
        print(f"start_representation_2: {start_representation_2.shape}")
        print(f"end_representation_2: {end_representation_2.shape}")

        total_loss = 0
        total_approximation_error_loss = 0
        total_orthogonality_loss = 0
        for top_i in range(1, self.d + 1):
            curr_coef = self.d - top_i + 1
            curr_loss, curr_approximation_error_loss, curr_orthogonality_loss = (
                _compute_loss_function_component(top_i)
            )
            total_loss += curr_coef * curr_loss
            total_approximation_error_loss += curr_coef * curr_approximation_error_loss
            total_orthogonality_loss += curr_coef * curr_orthogonality_loss

        metrics_dict = {
            "train_loss": total_loss,
            "approximation_error_loss": total_approximation_error_loss,
            "orthogonality_loss": total_orthogonality_loss,
        }
        metrics = (
            total_loss,
            total_approximation_error_loss,
            0.0,
            total_orthogonality_loss,
            metrics_dict,
        )
        aux = (metrics, None)

        return total_loss, aux

    def update_training_state(self, params, *args, **kwargs):
        """Leave params unchanged"""

        return params

    def additional_update_step(self, step, params, *args, **kwargs):
        """Leave params unchanged"""

        return params

    def init_additional_params(self, *args, **kwargs):
        additional_params = {}
        return additional_params

    def loss_function_non_permuted(self, *args, **kwargs):
        return self.loss_function(*args, **kwargs)

    def loss_function_permuted(self, *args, **kwargs):
        return self.loss_function(*args, **kwargs)

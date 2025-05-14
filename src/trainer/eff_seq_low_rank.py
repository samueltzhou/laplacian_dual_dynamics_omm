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


class EfficientSequentialLowRankObjectiveTrainer(LaplacianEncoderTrainer):
    def __init__(self, orbital_enabled=False, *args, **kwargs):
        self.orbital_enabled = orbital_enabled
        super().__init__(*args, **kwargs)

        # tmp = self._generate_static_masks()
        # self.static_vector_mask = tmp[0]
        # self.static_matrix_mask = tmp[1]
        # self.static_upper_mask = tmp[2]
        # self.static_lower_mask = tmp[3]
        # self.static_diag_mask = tmp[4]

    def _generate_static_masks(self) -> Tuple[jnp.ndarray]:
        """
        Generates cached vector mask, matrix mask, upper triangular mask, lower triangular mask, and diagonal mask
        """
        vector_mask, matrix_mask = self._generate_masks(self.d)
        upper_mask = jnp.tril(jnp.ones((self.d, self.d)), k=-1)
        lower_mask = upper_mask.T
        diag_mask = jnp.eye(self.d)

        return vector_mask, matrix_mask, upper_mask, lower_mask, diag_mask

    def _generate_masks(self, curr_d) -> Tuple[jnp.ndarray]:
        """
        Generates a monotonically decreasing mask in dimension index - both vector and matrix
        masks. Namely, they look like:
        [d, d - 1, ..., 2, 1]
        [[3, 2, 1]
         [2, 2, 1]
         [1, 1, 1]] (for the d = 3 case)

        Returns:
            Tuple of the form (vector_mask, matrix_mask)
        """
        vector_mask = jnp.arange(curr_d, 0, -1)
        matrix_mask = jnp.minimum(
            jnp.expand_dims(vector_mask, 1), jnp.expand_dims(vector_mask, 0)
        )

        return vector_mask, matrix_mask

    def _compute_indexwise_products(self, f, g) -> jnp.ndarray:
        assert f.shape[0] == g.shape[0]
        return jnp.einsum("bi, bj -> ij", f, g) / f.shape[0]

        # # use matmul instead of einsum for speed
        # return f.T @ g / f.shape[0]

    def loss_function(self, params, train_batch, **kwargs) -> Tuple[jnp.ndarray]:
        """
        Computes the low rank loss (either LoRA or OMM) for the sequential formulation.
        Implemented more efficiently with smarter stop grads and triangular masks.
        Equivalent to iterating the non-joint loss function from top_i in range(1, self.d + 1)

        Args:
            params: The parameters of the model.
            state_encoding: The state encoding.
            top_i: How many of the top eigenfunctions to use for the loss function.

        Returns:
            Tuple of (loss, approximation_error_loss, orthogonality_loss)
        """
        # Get representations
        (
            start_representation,
            end_representation,
            constraint_start_representation,
            constraint_end_representation,
        ) = self.encode_states(params["encoder"], train_batch)
        # start_representation and end_representation are correlated (sampled from an edge)
        # so are start_representation_2 and end_representation_2
        # constraint_start_representation and constraint_end_representation are uncorrelated (iid sampled from states)

        assert self.d == start_representation.shape[1]

        # shave off dimensions depending on what the current global_step is
        # curr_d linearly increases from 1 to self.d at the halfway point of training, then stays at self. d
        # curr_d = min(
        #     self.d,
        #     max(1, math.ceil(2 * self.d * self._global_step / self.total_train_steps)),
        # )
        # curr_d = 7
        curr_d = self.d

        (
            start_representation,
            end_representation,
            constraint_start_representation,
            constraint_end_representation,
        ) = (
            start_representation[:, :curr_d],
            end_representation[:, :curr_d],
            constraint_start_representation[:, :curr_d],
            constraint_end_representation[:, :curr_d],
        )

        start_representation_sg = jax.lax.stop_gradient(start_representation)
        end_representation_sg = jax.lax.stop_gradient(end_representation)
        constraint_start_representation_sg = jax.lax.stop_gradient(
            constraint_start_representation
        )
        # constraint_end_representation_sg = jax.lax.stop_gradient(
        #     constraint_end_representation
        # )

        print("Shapes of representations in batch: ")
        print(f"start_representation: {start_representation.shape}")
        print(f"end_representation: {end_representation.shape}")
        print(
            f"constraint_start_representation: {constraint_start_representation.shape}"
        )
        print(f"constraint_end_representation: {constraint_end_representation.shape}")

        lower_mask = jnp.tril(jnp.ones((curr_d, curr_d)), k=-1)
        upper_mask = lower_mask.T
        diag_mask = jnp.eye(curr_d)

        vector_mask, matrix_mask = self._generate_masks(curr_d)

        joint_self_inner_products = (
            jnp.einsum("bi, bi -> i", start_representation, end_representation)
            / start_representation.shape[0]
        )
        joint_self_inner_products_matrix = jnp.diag(joint_self_inner_products)

        # optional off diagonal penalty term
        off_diag_penalty_mat = (
            jnp.einsum("bi, bj -> ij", start_representation, end_representation)
            / start_representation.shape[0]
        )
        off_diag_penalty = (
            self.d
            / 2
            * jnp.sum(
                (off_diag_penalty_mat - jnp.diag(jnp.diag(off_diag_penalty_mat))) ** 2
            )
        )
        print(f"off_diag_penalty: {off_diag_penalty}")

        # common matrix computations for both LoRA and OMM
        # we're dropping start_representation_2 and end_representation_2, as the approximation that
        # results from not using them is negligible (slight increase in correlation between terms)
        joint_indexwise_products = (
            self._compute_indexwise_products(
                start_representation_sg, end_representation
            )
            * upper_mask
            + self._compute_indexwise_products(
                start_representation, end_representation_sg
            )
            * lower_mask
            + joint_self_inner_products_matrix
        )
        cov_indexwise_products = (
            self._compute_indexwise_products(
                constraint_start_representation_sg, constraint_start_representation
            )
            * upper_mask
            + self._compute_indexwise_products(
                constraint_start_representation, constraint_start_representation_sg
            )
            * lower_mask
            + self._compute_indexwise_products(
                constraint_start_representation, constraint_start_representation
            )
            * diag_mask
        )

        if self.orbital_enabled:
            approximation_error_loss = -2 * jnp.sum(
                vector_mask * joint_self_inner_products
            )
            orthogonality_loss = jnp.sum(
                matrix_mask * joint_indexwise_products * cov_indexwise_products
            )
        else:
            approximation_error_loss = -2 * jnp.sum(
                vector_mask * joint_self_inner_products
            )
            orthogonality_loss = jnp.sum(matrix_mask * cov_indexwise_products**2)
        train_loss = approximation_error_loss + orthogonality_loss + off_diag_penalty

        if self.coefficient_normalization:
            approximation_error_loss = approximation_error_loss / (
                curr_d * (curr_d + 1) / 2
            )
            orthogonality_loss = orthogonality_loss / (curr_d * (curr_d + 1) / 2)
            train_loss = train_loss / (curr_d * (curr_d + 1) / 2)

        metrics_dict = {
            "train_loss": train_loss,
            "approximation_error_loss": approximation_error_loss,
            "orthogonality_loss": orthogonality_loss,
        }
        metrics = (
            train_loss,
            approximation_error_loss,
            0.0,
            orthogonality_loss,
            metrics_dict,
        )
        aux = (metrics, None)

        return train_loss, aux

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

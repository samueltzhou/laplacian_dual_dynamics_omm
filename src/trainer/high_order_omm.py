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


class HighOrderSequentialOMMLossTrainer(LaplacianEncoderTrainer):
    def __init__(self, *args, **kwargs):
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
        # return jnp.einsum("bi, bj -> ij", f, g) / f.shape[0]

        # use matmul instead of einsum for speed
        return f.T @ g / f.shape[0]

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
        """
        Computes the orbital loss for the sequential formulation.
        Works with the p = 2 case instead of the p = 1 case.
        The full loss expansion is:
        trace(-4 <f|Tf> + 6 <f|f> <f|Tf> - 4<f|f>^2 <f|Tf> + <f|f>^3 <f|Tf>)

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

        def _compute_loss_function_component(top_i, **kwargs) -> jnp.ndarray:
            """
            Computes a single component of the loss function (the top_i loss).

            Args:
                params: The parameters of the model.
                state_encoding: The state encoding.
                top_i: How many of the top eigenfunctions to use for the loss function.

            Returns:
                Training loss for top_i eigenfunctions.
            """
            (
                curr_start_representation,
                curr_end_representation,
                curr_constraint_start_representation,
                # curr_constraint_end_representation,
            ) = (
                self._build_stop_grad_encoding(start_representation, top_i),
                self._build_stop_grad_encoding(end_representation, top_i),
                self._build_stop_grad_encoding(constraint_start_representation, top_i),
                # self._build_stop_grad_encoding(constraint_end_representation, top_i),
            )

            f_tf_inner_prod = self._compute_indexwise_products(
                curr_start_representation, curr_end_representation
            )
            f_f_inner_prod = self._compute_indexwise_products(
                curr_constraint_start_representation,
                curr_constraint_start_representation,
            )

            # full loss expansion
            # full loss is: trace(-4 <f|Tf> + 6 <f|f> <f|Tf> - 4<f|f>^2 <f|Tf> + <f|f>^3 <f|Tf>)

            f_f2 = f_f_inner_prod @ f_f_inner_prod
            f_f3 = f_f2 @ f_f_inner_prod

            matrix_poly = (
                -4 * jnp.eye(f_f_inner_prod.shape[0])
                + 6 * f_f_inner_prod
                - 4 * f_f2
                + 1 * f_f3
            )

            train_loss = jnp.sum(matrix_poly * f_tf_inner_prod)

            if self.coefficient_normalization:
                train_loss = train_loss / (top_i * (top_i + 1) / 2)

            return train_loss

        total_loss = 0
        for top_i in range(1, self.d + 1):
            curr_loss = _compute_loss_function_component(top_i)
            total_loss += curr_loss

        metrics_dict = {
            "train_loss": total_loss,
            "approximation_error_loss": total_loss,
            "orthogonality_loss": total_loss,
        }
        metrics = (
            total_loss,
            total_loss,
            0.0,
            total_loss,
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

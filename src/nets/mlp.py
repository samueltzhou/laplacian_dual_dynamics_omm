from typing import List
import haiku as hk
import jax
import numpy as np


def generate_fc_layers(
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "relu",
    use_layer_norm: bool = False,
) -> List[hk.Module]:
    """Generate layers for MLP.

    Args:
        output_dim: Size of the final linear layer.
        hidden_dims: List containing the widths of the hidden layers.
        activation: Activation function to use between layers ("relu" or "leaky_relu").
        use_layer_norm: If True, inserts a ``hk.LayerNorm`` after each linear layer. This
            can help stabilise training and mitigate issues such as dying neurons.
    """
    layers = []
    for dim in hidden_dims:
        layers.append(hk.Linear(dim))
        if use_layer_norm:
            # LayerNorm over the feature dimension (last axis).
            layers.append(hk.LayerNorm(axis=-1, create_scale=True, create_offset=True))

        if activation == "relu":
            layers.append(jax.nn.relu)
        elif activation == "leaky_relu":
            layers.append(jax.nn.leaky_relu)
        else:
            raise NotImplementedError
    layers.append(hk.Linear(output_dim))
    return layers


def generate_conv_layers(
    n_conv_layers: int = 2,
    activation: str = "relu",
    kernel_shape: int = 3,
) -> List[hk.Module]:
    """Generate layers for MLP."""
    layers = []
    for i in range(n_conv_layers - 1):
        layers.append(
            hk.Conv2D(
                output_channels=16,
                kernel_shape=kernel_shape,
                stride=2,
                padding=(1, 1),
                w_init=hk.initializers.VarianceScaling(
                    2.0, "fan_in", "truncated_normal"
                ),
            )
        )
        if activation == "relu":
            layers.append(jax.nn.relu)
        elif activation == "leaky_relu":
            layers.append(jax.nn.leaky_relu)
        else:
            raise NotImplementedError
    layers.append(
        hk.Conv2D(
            output_channels=16,
            kernel_shape=kernel_shape,
            stride=2,
            padding=(1, 1),
            w_init=hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal"),
        )
    )
    layers.append(jax.nn.relu)
    return layers


class MLP(hk.Module):
    """
    Standard multi-layer perceptron.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        use_layer_norm: bool = False,
        name: str = "MLP",
    ) -> None:
        super().__init__(name=name)
        print(
            f"MLP with the following parameters: hidden dims {hidden_dims}, activation {activation}, output dim {output_dim}, layer_norm {use_layer_norm}"
        )

        self.sequential = hk.Sequential(
            generate_fc_layers(output_dim, hidden_dims, activation, use_layer_norm)
        )

    def __call__(self, x: np.ndarray) -> jax.Array:
        """Forward pass through the layers."""
        return self.sequential(x)


class ConvNet(hk.Module):
    """
    Standard multi-layer perceptron.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        name: str = "ConvNet",
        n_conv_layers: int = 2,
        kernel_shape: int = 3,
    ) -> None:
        super().__init__(name=name)
        self.conv = hk.Sequential(
            generate_conv_layers(n_conv_layers, activation, kernel_shape)
        )
        self.flatten = hk.Flatten()
        self.linear = hk.Sequential(
            generate_fc_layers(output_dim, hidden_dims, activation)
        )

    def __call__(self, x: np.ndarray) -> jax.Array:
        """Forward pass through the layers."""
        x1 = self.conv(x)
        x2 = self.flatten(x1)
        x3 = self.linear(x2)
        x4 = x3
        return x4

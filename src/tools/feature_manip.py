import jax
import jax.numpy as jnp
from typing import Iterable, Iterator, List, Union, Any, Tuple


def slice_top_dimensions(
    arrays: Iterable[jnp.ndarray], top_d: int
) -> Iterator[jnp.ndarray]:
    """
    Slices the last dimension of each input array to keep only the top_d dimensions.

    Args:
        arrays: An iterable of JAX arrays, all with the same number of dimensions.
        top_d: Integer specifying how many of the top dimensions to keep in the -1 axis.

    Returns:
        An iterator yielding the sliced arrays with only the top_d dimensions in the last axis.

    Raises:
        AssertionError: If the input arrays have different numbers of dimensions.

    Example:
        >>> x = jnp.ones((3, 5))
        >>> y = jnp.zeros((4, 5))
        >>> sliced_arrays = list(slice_top_dimensions([x, y], 2))
        >>> sliced_arrays[0].shape  # (3, 2)
        >>> sliced_arrays[1].shape  # (4, 2)
    """
    arrays_list = list(arrays)
    if not arrays_list:
        return

    # Check that all arrays have the same number of dimensions
    ndim = arrays_list[0].ndim
    for i, arr in enumerate(arrays_list[1:], 1):
        assert (
            arr.ndim == ndim
        ), f"Array at index {i} has {arr.ndim} dimensions, but expected {ndim}"

    # Slice each array along the last dimension
    for arr in arrays_list:
        yield arr[..., :top_d]


def freeze_gradients(arrays: Iterable[jnp.ndarray]) -> Iterator[jnp.ndarray]:
    """
    Applies stop_gradient to each array in the input iterable to prevent gradient flow.

    Args:
        arrays: An iterable of JAX arrays, all with the same number of dimensions.

    Returns:
        An iterator yielding the input arrays with gradients frozen through jax.lax.stop_gradient.

    Raises:
        AssertionError: If the input arrays have different numbers of dimensions.

    Example:
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = jnp.array([[4.0, 5.0], [6.0, 7.0]])
        >>> # This would raise an AssertionError since x and y have different dimensions
        >>> # frozen_arrays = list(freeze_gradients([x, y]))
        >>>
        >>> # This would work:
        >>> x = jnp.ones((2, 3))
        >>> y = jnp.zeros((4, 3))
        >>> frozen_arrays = list(freeze_gradients([x, y]))
    """
    arrays_list = list(arrays)
    if not arrays_list:
        return

    # Check that all arrays have the same number of dimensions
    ndim = arrays_list[0].ndim
    for i, arr in enumerate(arrays_list[1:], 1):
        assert (
            arr.ndim == ndim
        ), f"Array at index {i} has {arr.ndim} dimensions, but expected {ndim}"

    # Apply stop_gradient to each array
    for arr in arrays_list:
        yield jax.lax.stop_gradient(arr)

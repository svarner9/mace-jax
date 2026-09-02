import jax
import jax.numpy as jnp
import numpy as np

from mace_jax.modules.utils import get_edge_vectors_and_lengths


def test_padded_zero_edge_has_finite_second_derivatives():
    edge_index = jnp.array([[0, 0], [1, 0]], dtype=jnp.int32)
    shifts = jnp.zeros((2, 3), dtype=jnp.float32)
    edge_mask = jnp.array([True, False])

    def summed_lengths(positions):
        _, lengths = get_edge_vectors_and_lengths(
            positions,
            edge_index,
            shifts,
            edge_mask=edge_mask,
            padded_edge_length=3.5,
        )
        return jnp.sum(lengths)

    positions = jnp.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=jnp.float32)
    _, lengths = get_edge_vectors_and_lengths(
        positions,
        edge_index,
        shifts,
        edge_mask=edge_mask,
        padded_edge_length=3.5,
    )
    hessian = jax.hessian(summed_lengths)(positions)

    np.testing.assert_allclose(lengths[:, 0], np.array([np.hypot(1.2, 0.1), 3.5]))
    assert np.all(np.isfinite(hessian))

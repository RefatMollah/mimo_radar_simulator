from __future__ import annotations

import numpy as np
import pytest

from mimo.geometry.quat import axis_angle_delta_quat, quat_multiply, quat_normalise, quat_rotate


def test_quaternion_operations_preserve_numpy_namespace() -> None:
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    v = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    assert isinstance(quat_multiply(q, q), np.ndarray)
    np.testing.assert_allclose(np.asarray(quat_normalise(q)), q)
    np.testing.assert_allclose(np.asarray(quat_rotate(q, v)), v)
    np.testing.assert_allclose(
        np.asarray(
            axis_angle_delta_quat(
                np.zeros((1, 3), dtype=np.float32), 
                np.array([0.1], dtype=np.float32))
            ),
        q,
    )


def test_mixed_numpy_and_jax_arrays_are_rejected() -> None:
    jnp = pytest.importorskip("jax.numpy")
    q_numpy = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    q_jax = jnp.asarray(q_numpy)

    with pytest.raises(TypeError, match="Multiple namespaces"):
        quat_multiply(q_numpy, q_jax)


def test_quaternion_operations_preserve_jax_namespace() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    q = jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    v = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)

    result = quat_rotate(q, v)

    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(np.asarray(result), np.asarray(v))

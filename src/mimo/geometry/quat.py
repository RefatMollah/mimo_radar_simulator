"""Quaternion operations that preserve the caller's array namespace."""

from __future__ import annotations

import numpy as np

from .._array import Array, ArrayInput, ArrayLike, DTypeLike, array_namespace


def identity_quats(n: int, *, like: ArrayLike | None = None, dtype: DTypeLike = np.float32) -> Array:
    """Return identity quaternions in the namespace of ``like``."""
    xp = np if like is None else array_namespace(like)
    base = xp.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    return xp.broadcast_to(base, (n, 4))


def quat_multiply(q1: ArrayLike, q2: ArrayLike, *, dtype: DTypeLike = np.float32) -> Array:
    """Return the Hamilton product of batches of ``[w, x, y, z]`` quaternions."""
    xp = array_namespace(q1, q2)
    w1, x1, y1, z1 = xp.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = xp.split(q2, 4, axis=-1)
    result = xp.concat((
        w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
        w2 * x1 + x2 * w1 - y2 * z1 + z2 * y1,
        w2 * y1 + x2 * z1 + y2 * w1 - z2 * x1,
        w2 * z1 - x2 * y1 + y2 * x1 + z2 * w1,
    ), axis=-1)
    return xp.astype(result, dtype)


def quat_normalise(quaternions: ArrayLike) -> Array:
    """Normalise quaternions, returning zero rows unchanged."""
    xp = array_namespace(quaternions)
    magnitude = xp.linalg.vector_norm(quaternions, axis=-1, keepdims=True)
    safe_magnitude = xp.where(magnitude > 0, magnitude, 1.0)
    return xp.where(magnitude > 0, quaternions / safe_magnitude, xp.zeros_like(quaternions))


def axis_angle_delta_quat(omega: ArrayLike, dt: ArrayLike, *, dtype: DTypeLike = np.float32) -> Array:
    """Convert batched angular velocities and time deltas into quaternions."""
    xp = array_namespace(omega, dt)
    omega_norm = xp.linalg.vector_norm(omega, axis=-1, keepdims=True)
    theta = omega_norm * dt[:, None]
    safe_norm = xp.where(omega_norm > 1e-12, omega_norm, 1.0)
    axis = omega / safe_norm
    half_theta = theta / 2.0
    delta = xp.concat((xp.cos(half_theta), axis * xp.sin(half_theta)), axis=-1)
    identity = identity_quats(omega.shape[0], like=omega, dtype=dtype)
    return xp.where(omega_norm > 1e-12, delta, identity)


def quat_rotate(q: ArrayLike, vectors: ArrayLike, *, dtype: DTypeLike = np.float32) -> Array:
    """Rotate vector rows by unit ``[w, x, y, z]`` quaternion rows."""
    xp = array_namespace(q, vectors)
    w, x, y, z = xp.split(q, 4, axis=-1)
    vector_part = xp.concat((x, y, z), axis=-1)
    r_cross_v = xp.cross(vector_part, vectors, axis=-1)
    rotated = vectors + 2.0 * xp.cross(vector_part, r_cross_v + w * vectors, axis=-1)
    return xp.astype(rotated, dtype)

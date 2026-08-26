###############################################################################
# Quaternion Utility Functions
###############################################################################
import numpy as np
from numpy.typing import NDArray, DTypeLike
from typing import TypeAlias, Any, TYPE_CHECKING, Union
from types import ModuleType

Backend: TypeAlias = Any

if TYPE_CHECKING:
    from jax import Array as JaxArray
else:
    JaxArray = Any
    
ArrayLike: TypeAlias = Union[NDArray, JaxArray]


"""
Quaternion convention:
[w,x,y,z]
represents body->world convention.
"""

def identity_quats(n: int, xp: Backend, dtype: DTypeLike) -> NDArray:
    base = xp.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    return xp.broadcast_to(base, (n, 4))

def quat_multiply(q1: ArrayLike, 
                  q2: ArrayLike,
                  *,
                  xp: Backend = np,
                  dtype: DTypeLike = np.float32
    ) -> ArrayLike:
    """Hamilton porduct of two arrays of quaternions."""
    w1, x1, y1, z1 = xp.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = xp.split(q2, 4, axis=-1)
    
    return xp.concatenate((
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 - y2*z1 + z2*y1,
        w2*y1 + x2*z1 + y2*w1 - z2*x1,
        w2*z1 - x2*y1 + y2*x1 + z2*w1
    ), axis=-1, dtype=dtype)
    
def quat_normalise(v: ArrayLike, *,xp: Backend = np) -> ArrayLike:
    mag = xp.linalg.norm(v, axis=-1, keepdims=True)
    return xp.divide(v, mag, out=xp.zeros_like(v), where= mag>0)

def axis_angle_delta_quat(omega: ArrayLike, 
                          dt: ArrayLike,
                          *,
                          xp: Backend = np,
                          dtype: DTypeLike = np.float32,
    ) -> ArrayLike:
    n = omega.shape[0]
    omega_norm = xp.linalg.norm(omega, axis=-1, keepdims=True)
    theta = omega_norm * dt[:, None]

    safe_norm = xp.where(omega_norm > 1e-12, omega_norm, 1.0)
    axis = omega/ safe_norm
    half = theta / 2.0
    
    w = xp.cos(half)
    xyz = axis * xp.sin(half)
    
    dq = xp.concat([w, xyz], axis=-1)
    dq_identity = xp.tile(xp.astype(xp.array([1.0, 0.0, 0.0, 0.0]), dtype), (omega.shape[0], 1))
    
    mask = (omega_norm > 1e-12)
    return xp.where(mask, dq, dq_identity)

def quat_rotate(
    q: ArrayLike,
    v: ArrayLike,
    *,
    xp: Backend = np,
    dtype: DTypeLike = np.float32,
) -> ArrayLike:
    """Rotate 3D vector(s) `v` by unit quaternion(s) `q` using Rodrigues-like vector formula:
    
    v' = v + 2 * r x (r x v + w * v), where q = (w, r).
    """
    w, x, y, z = xp.split(q, 4, axis=-1)
    r = xp.concatenate((x, y, z), axis=-1, dtype=dtype)
    
    # Compute cross product r x v
    rxv = xp.cross(r, v, axis=-1)
    
    # Compute cross product r x (r x v + w * v)
    t = 2.0 * xp.cross(r, rxv + w * v, axis=-1)
    
    return xp.astype(v + t, dtype=dtype)    
    

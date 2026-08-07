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
    ), axis=-1, dtype=np.float32)
    
def quat_normalise(v: ArrayLike, *,xp: Backend = np) -> ArrayLike:
    mag = xp.linalg.norm(v, axis=-1, keepdims=True)
    return xp.divide(v, mag, out=xp.zeros_like(v), where= mag>0)

def axis_angle_delta_quat(omega: ArrayLike, 
                          dt: ArrayLike,
                          *,
                          xp: ModuleType = np,
                          dtype: DTypeLike = np.float32,
    ) -> ArrayLike:
    n = omega.shape[0]
    omega_norm = xp.linalg.norm(omega, axis=-1, keepdims=True)
    theta = omega_norm * dt[:, None]

    dq = xp.zeros((n, 4), dtype=dtype)
    dq[:, 0] = 1.0    # Identity default for ~zero angular rate
    
    mask = (omega_norm > 1e-12).squeeze(-1)
    if xp.any(mask):
        axis = omega[mask] / omega_norm[mask]
        half = theta[mask] / 2.0
        
        dq[mask, 0] = xp.cos(half).squeeze(-1)
        dq[mask, 1:] = axis * xp.sin(half)
    
    return dq


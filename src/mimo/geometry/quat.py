###############################################################################
# Quaternion Utility Functions
###############################################################################
import numpy as np


"""
Quaternion convention:
[w,x,y,z]
represents body->world convention.
"""

def normalise(q) -> np.ndarray:
    return q / np.linalg.norm(q)

def quat_conjugate(q) -> np.ndarray:
    w, x, y, z = q

    return np.array([w, -x, -y, -z])

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 - y2*z1 + z2*y1,
        w2*y1 + x2*z1 + y2*w1 - z2*x1,
        w2*z1 - x2*y1 + y2*x1 + z2*w1
    ])

def rotate_vector(q, v) -> np.ndarray:
    q = normalise(q)
    v_quat = np.array([0, *v])
    
    rotated = quat_multiply(
        quat_multiply(q, v_quat),
        quat_conjugate(q)
    )
    
    return rotated[1:]

def angle_to_quat(axis, angle):
    axis = axis / np.linalg.norm(axis)
    
    s = np.sin(angle/2)
    
    return np.array([
        np.cos(angle/2),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s
    ])

def identity_quat() -> np.ndarray:
    return np.array([1, 0, 0, 0])
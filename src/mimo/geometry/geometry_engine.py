from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class GeometrySnapshot:
    time: float
    entity_ids: tuple[str, ...]
    
    los_vectors: NDArray[np.float64] # (N, N, 3)
    distances: NDArray[np.float64] # (N, N)
    radial_velocities: NDArray[np.float64] # (N, N)
    azimuths: NDArray[np.float64] # (N, N)
    elevations: NDArray[np.float64] # (N, N)
    


def calculate_los_vectors(positions: NDArray)-> NDArray:
    n = len(positions)
    out_matrix = np.zeros((n,n,3), dtype=np.float64)
    
    for idx, pos in enumerate(positions):
        diff = positions - pos
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        out_matrix[:,idx] = diff / norms
    
    idx = np.arange(n)            
    out_matrix[idx, idx] = np.nan
    
    return out_matrix

def calculate_los_vectors2(positions: NDArray) -> NDArray:
    diff = positions[:, None, :] - positions[None, :, :]
    
    norms = np.linalg.norm(diff, axis=2, keepdims=True)
    norms[norms==0] = 1.0
    
    los = diff / norms

    idx = np.arange(len(positions))
    los[idx, idx] = np.nan

    return los
    
    
    
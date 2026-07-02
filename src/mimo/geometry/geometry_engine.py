from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class GeometrySnapshot:
    time: float
    entity_ids: tuple[str, ...]
    
    los_vectors: NDArray[np.float64] # np.array((N, N, 3))
    distances: NDArray[np.float64] # np.array((N, N))
    radial_velocities: NDArray[np.float64] # np.array(())
    azimuths: NDArray[np.float64]
    elevations: NDArray[np.float64]
    
    
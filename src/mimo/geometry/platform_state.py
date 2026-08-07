from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from typing import ClassVar
from numpy.typing import NDArray
from enum import Enum

@dataclass(frozen=True)
class PlatformState:
    position: NDArray[np.float32]
    velocity: NDArray[np.float32]
    orientation: NDArray[np.float32]   # Quaternion [w, x, y, z]
    angular_velocity: NDArray[np.float32]
    time: float
    
    def __post_init__(self) -> None:
        object.__setattr__(self, 'position', np.asarray(self.position, dtype=np.float32))
        object.__setattr__(self, 'velocity', np.asarray(self.velocity, dtype=np.float32))
        object.__setattr__(self, 'orientation', np.asarray(self.orientation, dtype=np.float32))
        object.__setattr__(self, 'angular_velocity', np.asarray(self.angular_velocity, dtype=np.float32))
        
        if self.position.shape != (3,):
            raise ValueError(f"Position must be shape (3,) got {self.position.shape}.")
        if self.velocity.shape != (3,):
            raise ValueError(f"Velocity must be shape (3,) got {self.velocity.shape}.")
        if self.orientation.shape != (4,):
            raise ValueError(f"Quaternion must be shape (4,) got {self.orientation.shape}.")
        if self.angular_velocity.shape != (3,):
            raise ValueError(f"Angular velocity must be shape (3,) got {self.angular_velocity}.")
        
        norm = np.linalg.norm(self.orientation)
        
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError("Quaternion must be normalised.")
    

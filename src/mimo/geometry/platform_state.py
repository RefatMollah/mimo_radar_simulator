from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass
from numpy.typing import NDArray
from . import quat

@dataclass(frozen=True)
class PlatformState:
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    orientation: NDArray[np.float64]   # Quaternion [w, x, y, z]
    angular_velocity: NDArray[np.float64]
    time: float
    
    def __post_init__(self) -> None:
        object.__setattr__(self, 'position', np.asarray(self.position, dtype=np.float64))
        object.__setattr__(self, 'velocity', np.asarray(self.velocity, dtype=np.float64))
        object.__setattr__(self, 'orientation', np.asarray(self.orientation, dtype=np.float64))
        object.__setattr__(self, 'angular_velocity', np.asarray(self.angular_velocity, dtype=np.float64))
        
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
    
    # World Frame Functions    
    def relative_position_to(self, other: PlatformState) -> np.ndarray:
        return other.position - self.position
    
    def distance_to(self, other: PlatformState) -> np.floating:
        return np.linalg.norm(other.position - self.position)
    
    def distance_to_squared(self, other: PlatformState) -> np.floating:
        r = other.position - self.position
        return np.dot(r, r)
    
    def relative_velocity_to(self, other: PlatformState) -> np.ndarray:
        return other.velocity - self.velocity

    def line_of_sight_to(self, other: PlatformState) -> np.ndarray:
        r = other.position - self.position
        norm = np.linalg.norm(r)
        
        if norm < 1e-12:
            raise ValueError("Line of sight undefined for coincident platforms.")
        
        return r / norm
    
    def radial_velocity_to(self, other: PlatformState) -> np.floating:
        relative_velocity = other.velocity - self.velocity
        los = self.line_of_sight_to(other)
        return np.dot(relative_velocity, los)
    
    def closing_velocity_to(self, other: PlatformState) -> np.floating:
        return -self.radial_velocity_to(other)
    
    # Coordinate Transformations
    def body_to_world(self, vector: np.ndarray) -> np.ndarray:
        return quat.rotate_vector(self.orientation, vector)
    
    def world_to_body(self, vector: np.ndarray) -> np.ndarray:
        return quat.rotate_vector(
            quat.quat_conjugate(self.orientation), 
            vector
        )
    
    # Body Frame Functions
    def relative_position_body_to(self, other: PlatformState):
        return self.world_to_body(other.position - self.position)
    
    def relative_velocity_body_to(self, other: PlatformState):
        return self.world_to_body(other.velocity - self.velocity)
    


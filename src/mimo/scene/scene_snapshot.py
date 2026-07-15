from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from ..geometry.platform_state import PlatformState

 
@dataclass(slots=True, frozen=True)
class SceneSnapshot:
    time: float
    entity_ids: tuple[str, ...]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    orientations: NDArray[np.float64]
    angular_velocities: NDArray[np.float64]
    is_static: NDArray[np.bool_]
    
        
    def get_entity_state(self, entity_slot: int) -> PlatformState:
        try:
            idx = entity_slot
            return PlatformState.fast_create(
                position=self.positions[idx],
                velocity=self.velocities[idx],
                orientation=self.orientations[idx],
                angular_velocity=self.angular_velocities[idx],
                time=self.time
            )
        except KeyError:
            raise KeyError(
                f"Entity with slot:'{entity_slot}' not found in snapshot at t={self.time:.6f} s."
            ) from None
    
    def __len__(self) -> int:
        return len(self.entity_ids)
    
    def __repr__(self) -> str:
        return (
            f"SceneSnapshot(time={self.time:.6f}, n_entities{len(self.entity_ids)})"
        )
    
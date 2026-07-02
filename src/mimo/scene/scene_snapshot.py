from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from ..geometry.platform_state import PlatformState

 
@dataclass(frozen=True)
class SceneSnapshot:
    time: float
    
    entity_ids: tuple[str, ...]
    
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    orientations: NDArray[np.float64]
    angular_velocities: NDArray[np.float64]
    
    is_static: NDArray[np.bool_]
    
    _lookup: dict[str, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_lookup",
            {eid: i for i, eid in enumerate(self.entity_ids)}
        )
        
    def _idx(self, entity_id: str) -> int:
        try:
            return self._lookup[entity_id]
        except KeyError:
            raise KeyError(
                f"Entity '{entity_id} not found in snapshot at t={self.time:.6f} s.'"
            ) from None
    
    def get_position(self, entity_id: str) -> NDArray[np.float64]:
        return self.positions[self._idx(entity_id)]
            
    def get_velocity(self, entity_id: str) -> NDArray[np.float64]:
        return self.velocities[self._idx(entity_id)]
                
    def get_orientation(self, entity_id: str) -> NDArray[np.float64]:
        return self.orientations[self._idx(entity_id)]
    
    def get_angular_velocity(self, entity_id: str) -> NDArray[np.float64]:
        return self.angular_velocities[self._idx(entity_id)]
            
    def get_is_static(self, entity_id: str) -> np.bool_:
        return self.is_static[self._idx(entity_id)]
        
        
    def get_entity_state(self, entity_id: str) -> PlatformState:
        try:
            idx = self._lookup[entity_id]
            return PlatformState.fast_create(
                position=self.positions[idx],
                velocity=self.velocities[idx],
                orientation=self.orientations[idx],
                angular_velocity=self.angular_velocities[idx],
                time=self.time
            )
        except KeyError:
            raise KeyError(
                f"Entity '{entity_id}' not found in snapshot at t={self.time:.6f} s."
            ) from None
    
    def __len__(self) -> int:
        return len(self.entity_ids)
    
    def __repr__(self) -> str:
        return (
            f"SceneSnapshot(time={self.time:.6f}, n_entities{len(self.entity_ids)})"
        )
    
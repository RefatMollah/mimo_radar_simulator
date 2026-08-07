from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from ..geometry.platform_state import PlatformState
from ..scene.scene import EntityNotFoundError

 
@dataclass(slots=True, frozen=True)
class SceneSnapshot:
    time: float
    entity_ids: tuple[str, ...]
    positions: NDArray[np.float32]
    velocities: NDArray[np.float32]
    accelerations: NDArray[np.float32]
    orientations: NDArray[np.float32]
    angular_velocities: NDArray[np.float32]
    is_static: NDArray[np.bool_]
        

    
    def __len__(self) -> int:
        return len(self.entity_ids)
    
    def __repr__(self) -> str:
        return (
            f"SceneSnapshot(time={self.time:.6f}, n_entities{len(self.entity_ids)})"
        )
    
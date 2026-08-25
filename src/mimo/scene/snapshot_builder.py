from __future__ import annotations

import numpy as np
from numpy.typing import NDArray, DTypeLike

from dataclasses import dataclass
import logging

from ..geometry.spatial_engine import State, check_causality, state_at
from .scene import Scene, CompiledScene


logger = logging.getLogger(__name__)
  
@dataclass(frozen=True, slots=True)
class SceneSnapshot:
    state: State
    entity_ids: tuple[str, ...]
    is_static: NDArray[np.bool_]
    is_active: NDArray[np.bool_]
    xp: str
    
    @property
    def time(self) -> float | None:
        return self.state.time
    
    @property
    def positions(self) -> NDArray[np.float32]:
        return self.state.positions
    
    @property
    def velocities(self) -> NDArray[np.float32]:
        return self.state.velocities
    
    @property
    def accelerations(self) -> NDArray[np.float32]:
        return self.state.accelerations
    
    @property
    def orientations(self) -> NDArray[np.float32]:
        return self.state.orientations
    
    @property
    def angular_rates(self) -> NDArray[np.float32]:
        return self.state.angular_rates
    

class SnapshotBuilder:
    def __init__(
        self,
        scene: Scene,
        start_time: float = 0.0,
    ) -> None:
        self._scene = scene
        self._time = start_time
        self._compiled = scene.compile()
    
    @property
    def current_time(self) -> float:
        return self._time
    
    @property
    def compiled_scene(self) -> CompiledScene:
        self._sync()
        return self._compiled

    def _sync(self) -> None:
        if self._compiled.version != self._scene.topology_version:
            self._compiled = self._scene.compile()
    
    def advance(self, dt: float) -> SceneSnapshot:
        if dt < 0:
            raise ValueError(f"dt must be non-negative; got {dt!r}.")
        self._time += dt
        return self.snapshot_at(self._time)
    
    def reset(self, time: float = 0.0) -> None:
        self._time = time
    
    def snapshot_at(self, time: float) -> SceneSnapshot:
        self._sync()
        if self._compiled.backend == "numpy":
            check_causality(self._compiled.motion_batches, time)
        
        state = state_at(self._compiled, time)
        
        return SceneSnapshot(
            state=state,
            entity_ids=self._compiled.entity_ids,
            is_static=self._compiled.is_static,
            is_active=self._compiled.is_active,
            xp=self._compiled.backend
        )
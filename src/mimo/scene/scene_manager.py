from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass
import logging
from typing import Iterable

from .scene import Scene
from .scene_snapshot import SceneSnapshot

from src.mimo.entity.entity import Entity

logger = logging.getLogger(__name__)
  
class SceneManager:
    _static_batch: _EntityBatch
    _static_cache_time: float
    
    def __init__(self, scene: Scene, start_time: float =0.0) -> None:
        self._scene = scene
        self._time = start_time
        self._static_batch = _EntityBatch.empty()
        self._static_cache_time = start_time
        self.rebuild_static_cache(start_time)

    @property
    def current_time(self) -> float:
        return self._time                     
    
    def advance(self, dt: float) -> SceneSnapshot:
        if dt < 0:
            raise ValueError(f"dt must be non-negative; got {dt!r}.")
        self._time += dt
        return self._build_snapshot(self._time)
        
    def reset(self, time: float = 0.0) -> None:
        self._time = time

    def snapshot_at(self, time: float) -> SceneSnapshot:
        return self._build_snapshot(time)

    def rebuild_static_cache(self, time: float | None = None) -> None:
        if time is None:
            time = self._time
        self._static_batch = _collect_entities(self._scene.iter_static(), time, label="Static")
        self._static_cache_time = time

    def append_static_entities(self, entities: Iterable[Entity], time:float | None=None) -> None:
        if time is None:
            time = self._static_cache_time
        new_batch = _collect_entities(entities, time, label="Static")
        self._static_batch = _combine(self._static_batch, new_batch)
        
    def _build_snapshot(self, time: float):
        dynamic_batch = _collect_entities(self._scene.iter_dynamic(), time, label="Dynamic")
        combined = _combine(self._static_batch, dynamic_batch)
        
        is_static = np.zeros(combined.n, dtype=np.bool_)
        is_static[: self._static_batch.n] = True

        return SceneSnapshot(
            time=time,
            entity_ids=combined.ids,
            positions=combined.positions,
            velocities=combined.velocities,
            orientations=combined.orientations,
            angular_velocities=combined.angular_velocities,
            is_static=is_static,
        )
        
@dataclass
class _EntityBatch:
    ids: tuple[str, ...]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    orientations: NDArray[np.float64]
    angular_velocities: NDArray[np.float64]
    
    @property
    def n(self) -> int:
        return len(self.ids)
    
    @staticmethod
    def empty() -> "_EntityBatch":
        return _EntityBatch(
            ids=(),
            positions=np.empty((0,3), dtype=np.float64),
            velocities=np.empty((0,3), dtype=np.float64),
            orientations=np.empty((0,4), dtype=np.float64),
            angular_velocities=np.empty((0,3), dtype=np.float64),
        )
    

def _stack_or_empty(rows: list[NDArray[np.float64]], width: int) -> NDArray[np.float64]:
    if rows:
        return np.vstack(rows)
    return np.empty((0, width), dtype=np.float64)

def _collect_entities(entities: Iterable[Entity], time: float, *, label: str) -> _EntityBatch:
    ids: list[str] = []
    positions: list[NDArray[np.float64]] = []
    velocities: list[NDArray[np.float64]] = []
    orientations: list[NDArray[np.float64]] = []
    angular_vels: list[NDArray[np.float64]] = []
    
    for entity in entities:
        try:
            state = entity.get_state(time)
        except Exception as exc:
            logger.warning("%s entity %s dropped from snapshot at t=%.6f: %s.", label, entity.id, time, exc)
            continue
        
        ids.append(entity.id)
        positions.append(state.position)
        velocities.append(state.velocity)
        orientations.append(state.orientation)
        angular_vels.append(state.angular_velocity)
    
    return _EntityBatch(
        ids=tuple(ids),
        positions=_stack_or_empty(positions, 3),
        velocities=_stack_or_empty(velocities, 3),
        orientations=_stack_or_empty(orientations, 4),
        angular_velocities=_stack_or_empty(angular_vels, 3)
    )

def _combine(a: _EntityBatch, b: _EntityBatch) -> _EntityBatch:
    if a.n == 0:
        return b
    if b.n == 0:
        return a
    
    return _EntityBatch(
        ids = a.ids + b.ids,
        positions=np.vstack((a.positions, b.positions)),
        velocities=np.vstack((a.velocities, b.velocities)),
        orientations=np.vstack((a.orientations, b.orientations)),
        angular_velocities=np.vstack((a.angular_velocities, b.angular_velocities))
    )
      
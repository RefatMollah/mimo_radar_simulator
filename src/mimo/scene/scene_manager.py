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
    _static_cache_version: int
    
    def __init__(self, scene: Scene, start_time: float =0.0) -> None:
        self._scene = scene
        self._time = start_time
        
        self._static_batch = _EntityBatch.empty()
        self._static_cache_time = start_time
        self._static_cache_version = -1
        
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
    
    def _ensure_static_cache(self, time: float):
        if self._static_cache_version != self._scene.topology_version:
            self.rebuild_static_cache(time)

    def rebuild_static_cache(self, time: float | None = None) -> None:
        if time is None:
            time = self._time
            
        self._static_batch = _collect_entities(self._scene.iter_static(), time, label="Static")
        self._static_cache_time = time
        self._static_cache_version = self._scene.topology_version

    def append_static_entities(self, entities: Iterable[Entity], time:float | None=None) -> None:
        if time is None:
            time = self._static_cache_time
        new_batch = _collect_entities(entities, time, label="Static")
        self._static_batch = _combine(self._static_batch, new_batch)
        
    def _build_snapshot(self, time: float):
        
        # Ensure the static entity cache is not stale
        self._ensure_static_cache(time)
        
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
    
    @property
    def static_cache_stale(self) -> bool:
        return self._static_cache_version != self._scene.topology_version
        
@dataclass(slots=True, frozen=True)
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
    
def _collect_entities(entities: Iterable[Entity], time: float, *, label: str) -> _EntityBatch:

    entities = tuple(entities)
    n = len(entities)
    if n == 0:
        return _EntityBatch.empty()
    
    ids = [""] * n
    positions = np.empty((n, 3), dtype=np.float64)
    velocities = np.empty((n, 3), dtype=np.float64)
    orienations = np.empty((n, 4), dtype=np.float64)
    angular_velocities = np.empty((n, 3), dtype=np.float64)
    
    write = 0
    for entity in entities:
        try:
            state = entity.get_state(time)
        except Exception as exc:
            logger.warning(
                "%s entity %s dropped from snapshot at t=%.6f: %s.",
                label,
                entity.id,
                time,
                exc,
            )
            continue

        ids[write] = entity.id
        positions[write] = state.position
        velocities[write] = state.velocity
        orienations[write] = state.orientation
        angular_velocities[write] = state.angular_velocity
        write += 1
        
    return _EntityBatch(
        ids=tuple(ids[:write]),
        positions=positions[:write],
        velocities=velocities[:write],
        orientations=orienations[:write],
        angular_velocities=angular_velocities[:write]
        )

def _concat(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty((len(a) + len(b), a.shape[1]), dtype=a.dtype)
    out[:len(a)] = a
    out[len(a):] = b
    return out
    
def _combine(a: _EntityBatch, b: _EntityBatch) -> _EntityBatch:
    if a.n == 0:
        return b
    if b.n == 0:
        return a
    
    return _EntityBatch(
        ids = a.ids + b.ids,
        positions=_concat(a.positions, b.positions),
        velocities=_concat(a.velocities, b.velocities),
        orientations=_concat(a.orientations, b.orientations),
        angular_velocities=_concat(a.angular_velocities, b.angular_velocities)
    )
      
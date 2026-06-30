from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import logging
from typing import TYPE_CHECKING

from .scene import Scene
from .scene_snapshot import SceneSnapshot

logger = logging.getLogger(__name__)


class SceneManager:
    
    def __init__(self, scene: Scene, start_time: float =0.0) -> None:
        self._scene = scene
        self._time = start_time
        
        # Caching Static Entities
        static_ids: list[str] = []
        static_positions: list[NDArray[np.float64]] = []
        static_velocities: list[NDArray[np.float64]] = []
        static_orientations: list[NDArray[np.float64]] = []
        static_angular_vels: list[NDArray[np.float64]] = []
        
        for entity in self._scene.iter_static():
            try:
                entity_state = entity.get_state(start_time)
                static_ids.append(entity.id)
                static_positions.append(entity_state.position)
                static_velocities.append(entity_state.velocity)
                static_orientations.append(entity_state.orientation)
                static_angular_vels.append(entity_state.angular_velocity)
            except Exception as exc:
                logger.warning("Static entity %s dropped from snapshot: %s.", entity.id, exc)
        
        self._static_ids = tuple(static_ids)
        self._n_static = len(self._static_ids)
        
        if self._n_static > 0:
            self._static_positions = np.vstack(static_positions)
            self._static_velocities = np.vstack(static_velocities)
            self._static_orientations = np.vstack(static_orientations)
            self._static_angular_vels = np.vstack(static_angular_vels)
        else:
            self._static_positions = np.empty((0,3), dtype=np.float64)
            self._static_velocities = np.empty((0,3), dtype=np.float64)
            self._static_orientations = np.empty((0,3), dtype=np.float64)
            self._static_angular_vels = np.empty((0,3), dtype=np.float64)
                     
    
    def advance(self, dt: float) -> SceneSnapshot:
        if dt < 0:
            raise ValueError(f"dt must be non-negative; got {dt!r}.")
        self._time += dt
        return self._build_snapshot(self._time)
    
    def snapshot_at(self, time: float) -> SceneSnapshot:
        return self._build_snapshot(time)
    
    def reset(self, time: float = 0.0) -> None:
        self._time = time

    @property
    def current_time(self) -> float:
        return self._time
    
    def _build_snapshot(self, time: float):
        dynamic_ids: list[str] = []
        dynamic_positions: list[NDArray[np.float64]] = []
        dynamic_velocities: list[NDArray[np.float64]] = []
        dynamic_orientations: list[NDArray[np.float64]] = []
        dynamic_angular_vels: list[NDArray[np.float64]] = []
                
        for entity in self._scene.iter_dynamic():
            try:
                state = entity.get_state(time)
                dynamic_ids.append(entity.id)
                dynamic_positions.append(state.position)
                dynamic_velocities.append(state.velocity)
                dynamic_orientations.append(state.orientation)
                dynamic_angular_vels.append(state.angular_velocity)
            except Exception as exc:
                logger.warning("Entity %s dropped from snapshot at t=%.6f: %s.", entity.id, time, exc)
        
        n_dynamic = len(dynamic_ids)
        n_total = self._n_static + n_dynamic
        
        if n_total == 0:
            return SceneSnapshot(
                time=time,
                entity_ids=(),
                positions=np.empty((0,3)),
                velocities=np.empty((0,3)),
                orientations=np.empty((0,4)),
                angular_velocities=np.empty((0,3)),
                is_static=np.zeros(0, dtype=np.bool_)
            )
        
        if n_dynamic > 0:
            positions = np.vstack((self._static_positions, dynamic_positions))
            velocities = np.vstack((self._static_velocities, dynamic_velocities))
            orientations = np.vstack((self._static_orientations, dynamic_orientations))
            angular_velocities = np.vstack((self._static_angular_vels, dynamic_angular_vels))
        else:
            positions = self._static_positions
            velocities = self._static_velocities
            orientations = self._static_orientations
            angular_velocities = self._static_angular_vels
        
        # Build boolean mask.    
        is_static = np.zeros(n_total, dtype=np.bool_)
        is_static[:self._n_static] = True
        
        return SceneSnapshot(
            time=time,
            entity_ids=self._static_ids + tuple(dynamic_ids),
            positions=positions,
            velocities=velocities,
            orientations=orientations,
            angular_velocities=angular_velocities,
            is_static=is_static
        )
        
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from collections import defaultdict
from types import ModuleType
import numpy as np

from typing import TYPE_CHECKING, Dict, Mapping
from numpy.typing import NDArray

from .._array import Array, ArrayInput, BackendName, DTypeLike
from ..geometry.motion_model import build_batch, Motion, MotionBatch

if TYPE_CHECKING:
    from ..entity.entity import Entity
    
logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class BackendContext:
    """Explicit backend selection for scene compilation and JAX execution."""

    name: BackendName
    dtype: DTypeLike = np.float32
    
    @property
    def is_jax(self) -> bool:
        return self.name == "jax"

    @property
    def module(self) -> ModuleType:
        return self.xp

    @property
    def xp(self) -> ModuleType:
        if self.name == "numpy":
            return np
        try:
            import jax.numpy as jnp
        except ImportError as error:
            raise ImportError(
                "The JAX backend requires the optional dependency. "
                "Install it with 'pip install mimo-radar[jax]'."
            ) from error
        return jnp

    def array(self, value: ArrayInput) -> Array:
        return self.xp.asarray(value, dtype=self.dtype)
    
    @classmethod
    def numpy(cls, dtype: DTypeLike = np.float32) -> BackendContext:
        return cls(
            name="numpy",
            dtype=dtype,
        )
    
    @classmethod
    def jax(cls, dtype: DTypeLike = np.float32) -> BackendContext:
        return cls(
            name="jax",
            dtype=dtype,
        )

@dataclass(frozen=True, slots=True)
class CompiledScene:
    version: int
    entity_ids: tuple[str, ...]
    is_static: NDArray[np.bool_]
    is_active: NDArray[np.bool_]
    slots_by_motion: Mapping[str, NDArray[np.intp]]
    motion_batches: Mapping[str, MotionBatch]
    dtype: DTypeLike
    backend: BackendName
    
    @property
    def n(self) -> int:
        return len(self.entity_ids)
    
    @property
    def xp(self) -> ModuleType:
        return BackendContext(self.backend, self.dtype).xp

_JAX_REGISTERED = False

def _register_jax_pytree() -> None:
    global _JAX_REGISTERED
    
    if _JAX_REGISTERED:
        return
    
    import jax

    jax.tree_util.register_dataclass(
        CompiledScene,
        data_fields=(
            "is_static",
            "is_active",
            "slots_by_motion",
            "motion_batches",
        ),
        meta_fields=(
            "version",
            "entity_ids",
            "dtype",
            "backend",            
        ),
    )
    
    _JAX_REGISTERED = True


class Scene:
    
    def __init__(self, *, backend: BackendContext = BackendContext.numpy()) -> None:
        self._entities: Dict[int, Entity] = {}
        self._slots_by_id: Dict[str, int] = {}
        
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._topology_version = 0
        
        self._backend = backend
    
    @property
    def topology_version(self) -> int:
        return self._topology_version
    
    @property
    def max_slots(self) -> int:
        return self._next_slot

    @property
    def entity_count(self) -> int:
        return len(self._entities) 
         
    def add_entity(self, entity: Entity) -> int:
        """Register `entity` and return its slot.
        """
        entity_id = entity.id
        if entity_id in self._slots_by_id:
            raise DuplicateEntityError(f"Entity ID: {entity_id!r} is already registered.")

        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            slot = self._next_slot
            self._next_slot += 1

        entity._assign_scene_index(slot)
        self._entities[slot] = entity
        self._slots_by_id[entity_id] = slot
        self._bump_topology()
        return slot
    
    def remove_entity(self, entity_id: str) -> None:
        slot = self._slots_by_id.pop(entity_id, None)
        if slot is None:
            raise EntityNotFoundError(entity_id)

        entity = self._entities.pop(slot)
        entity._assign_scene_index(-1)
        self._free_slots.append(slot)
        self._bump_topology()
        
    def get_entity(self, entity_id: str) -> Entity:
        slot = self._slots_by_id.get(entity_id)
        if slot is None:
            raise EntityNotFoundError(entity_id)
        return self._entities[slot]
    
    def _bump_topology(self) -> None:
        self._topology_version += 1
        
    @staticmethod
    def _set_entity_slot(entity: Entity, slot: int) -> None:
        entity._assign_scene_index(slot)
    
    def iter_static(self) -> Iterator[Entity]:
        return (e for e in self._entities.values() if e.is_static)

    def iter_dynamic(self) -> Iterator[Entity]:
        return (e for e in self._entities.values() if not e.is_static)

    def iter_all(self) -> Iterator[Entity]:
        return iter(self._entities.values())
            
    def compile(self) -> CompiledScene:
        """Snapshot the current topology and per-entity motion parameters
        into an immutable `CompiledSimulation`.

        """
        backend = self._backend
        xp = backend.xp
        dtype = backend.dtype
        
        if backend.name == "jax":
            _register_jax_pytree()
        
        n = self._next_slot
        entity_ids: list[str] = [""] * n
        is_static = xp.zeros(n, dtype=bool)
        is_active = xp.zeros(n, dtype=bool)
        buckets: dict[type[Motion], list[Entity]] = defaultdict(list)

        for slot, entity in self._entities.items():
            entity_ids[slot] = entity.id
            if backend.is_jax:
                is_static = is_static.at[slot].set(entity.is_static)
                is_active = is_active.at[slot].set(True)
            else:
                is_static[slot] = entity.is_static
                is_active[slot] = True
            buckets[type(entity.motion)].append(entity)

        slots_by_motion: dict[str, NDArray[np.intp]] = {
            motion_cls.__name__: np.array([e.scene_index for e in group], dtype=np.intp)
            for motion_cls, group in buckets.items()
        }
        motion_batches: dict[str, MotionBatch] = {
            motion_cls.__name__: build_batch(motion_cls, group, dtype, xp)
            for motion_cls, group in buckets.items()
        }

        return CompiledScene(
            version=self._topology_version,
            entity_ids=tuple(entity_ids),
            is_static=is_static,
            is_active=is_active,
            slots_by_motion=slots_by_motion,
            motion_batches=motion_batches,
            dtype=dtype,
            backend=backend.name,
        )
    
    
    def __repr__(self) -> str:
        return (
            f"Scene(Entities={len(self._entities)} "
        )  


#-----------------------------------------------------------
# Exceptions
#-----------------------------------------------------------

class DuplicateEntityError(Exception):
    """Raised when an entity UUID is already registered in the scene."""
    
    def __init__(self, entity_id: str) -> None:
        super().__init__(
            f"Entity '{entity_id}' is already registered in this scene."
        )

class EntityNotFoundError(Exception):
    """Raised when a requested entity UUID is absent from the scene."""
    
    def __init__(self, entity_id: str):
        super().__init__(
            f"Entity '{entity_id}' not found in this scene."
        )

class InvalidTopologyError(Exception):
    """Raised when an invalid topology is entered."""
    
    def __init__(self, message: str):
        super().__init__(message)

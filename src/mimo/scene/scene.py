from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..entity.entity import Entity
    
logger = logging.getLogger(__name__)


#################################################################################
# Exceptions
#################################################################################

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


#################################################################################
# Scene
#################################################################################

class Scene:
    
    def __init__(self) -> None:
        self._static_entities: dict[str, Entity] = {}
        self._dynamic_entities: dict[str, Entity] = {}
        
    
    def add_entity(self, entity:Entity, static: bool=False) -> None:
        if entity.id in self._static_entities or entity.id in self._dynamic_entities:
            raise DuplicateEntityError(entity.id)
        
        if static:
            self._static_entities[entity.id] = entity
        else:
            self._dynamic_entities[entity.id] = entity
    
    
    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self._static_entities:
            del self._static_entities[entity_id]
            return
        if entity_id in self._dynamic_entities:
            del self._dynamic_entities[entity_id]
            return
        raise EntityNotFoundError(entity_id)
    
    
    def get_entity(self, entity_id: str) -> Entity:
        
        if entity_id in self._static_entities:
            return self._static_entities[entity_id]
        if entity_id in self._dynamic_entities:
            return self._dynamic_entities[entity_id]
        raise EntityNotFoundError(entity_id)
    
    
    def iter_static(self) -> Iterator[Entity]:
        yield from self._static_entities.values()
    
    def iter_dynamic(self) -> Iterator[Entity]:
        yield from self._dynamic_entities.values()
    
    def iter_all(self) -> Iterator[Entity]:
        yield from self._static_entities.values()
        yield from self._dynamic_entities.values()
    
    
    @property
    def entity_count(self) -> int:
        return len(self._static_entities) + len(self._dynamic_entities)
    
    def __repr__(self) -> str:
        return (
            f"Scene(static={len(self._static_entities)}, "
            f"dynamic={len(self._dynamic_entities)})"
        )   
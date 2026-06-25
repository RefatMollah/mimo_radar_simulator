from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .entity import Entity

class RadarComponent():
    __slots__ = ("_entity",)
    
    def __init__(self) -> None:
        self._entity: Entity | None = None
    
    @property
    def entity(self) -> Entity:
        if self._entity is None:
            raise RuntimeError("Component not attached.")
        
        return self._entity
    
    @property
    def attached(self) -> bool:
        return self._entity is not None
    
    def on_attach(self, entity) -> None:
        self._entity = entity

    def on_detach(self) -> None:
        self._entity = None

    def on_update(self, time: float) -> None:
        pass
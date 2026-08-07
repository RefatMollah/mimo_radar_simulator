from __future__ import annotations

from typing import Iterable, cast, TypeVar, TYPE_CHECKING

from uuid import uuid4
import numpy as np

from ..geometry.platform_state import PlatformState
from ..geometry.motion_model import Motion, StaticMotion

if TYPE_CHECKING:
    from .radar_component import RadarComponent

C = TypeVar("C", bound="RadarComponent")
        
class Entity():
    __slots__ = ("_id", "_motion","_radar_components", "_active", "_slot", "_static")
    
    def __init__(
        self,
        motion: Motion,
        components: Iterable[RadarComponent] | None=None,
        entity_id: str | None = None,
        active: bool = True,
    ) -> None:
        
        self._id = entity_id or str(uuid4())
        self._motion = motion
        self._radar_components: dict[type[RadarComponent], RadarComponent] = {}
        self._active = active
        self._slot: int = -1
        if components is not None:
            for component in components:
                self.add_component(component)
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def active(self) -> bool:
        return self._active
    
    @property
    def slot(self) -> int:
        if self._slot == -1:
            raise RuntimeError(f"Entity {self._id} has not been added to a Scene.")
        return self._slot
    
    @property
    def motion(self) -> Motion:
        return self._motion
    
    @property
    def is_static(self) -> bool:
        return isinstance(self.motion, StaticMotion)
    
    def set_slot(self, slot: int) -> None:
        self._slot = slot
    
    def add_component(self, component: RadarComponent) -> None:
        component_type = type(component)
        if component_type in self._radar_components:
            raise ComponentAlreadyAttachedError(component_type.__name__)
        
        self._radar_components[component_type] = component
        component.on_attach(self)
    
    def has_component(self, component_type: type[C]) -> bool:
        return component_type in self._radar_components
    
    def get_component(self, component_type: type[C]) -> C:
        try:
            return cast(C, self._radar_components[component_type])
        except KeyError:
            raise ComponentNotFoundError(component_type.__name__)
    
    @property
    def static(self) -> np.bool_:
        return self._static
    

    def remove_component(self, component_type: type[C]) -> None:
        component = self.get_component(component_type)
        component.on_detach()
        del self._radar_components[component_type]
    

##################################################################################################################
# Exceptions
##################################################################################################################   
    
class ComponentAlreadyAttachedError(Exception):
    def __init__(self, component_name: str):
        super().__init__(f"Component {component_name} already attached to this entity")

class ComponentNotFoundError(Exception):
    def __init__(self, component_name):
        super().__init__(f"Component {component_name} not attached.")
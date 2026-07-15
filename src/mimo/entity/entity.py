from __future__ import annotations
from typing import Iterable, cast, TypeVar, TYPE_CHECKING
from uuid import uuid4
from src.mimo.geometry.motion_model import MotionModel

if TYPE_CHECKING:
    from .radar_component import RadarComponent

C = TypeVar("C", bound="RadarComponent")
        
class Entity():
    __slots__ = ("_id", "_motion_model","_radar_components", "_active", "_slot")
    
    def __init__(
        self,
        motion_model: MotionModel,
        components: Iterable[RadarComponent] | None = None,
        entity_id: str | None = None,
        active: bool = True,

    ) -> None:
        
        self._id = entity_id or str(uuid4())
        self._motion_model = motion_model
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
    
    def get_state(self, time: float):
        return self._motion_model.get_state(time)
    
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
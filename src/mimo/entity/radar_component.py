from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .entity import Entity

####################################################################
# Base class
####################################################################

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
    
#####################################################################
# Derived Classes
#####################################################################

class TransmitterComponent(RadarComponent):
    __slots__ = ("frequency", "peak_power", "gain_pattern")
    
    def __init__(self, frequency: float = 10e9, peak_power:float = 1000.0) -> None:
        super().__init__()
        self.frequency = frequency
        self.peak_power = peak_power
    

class ReceiverComponent(RadarComponent):
    __slots__ = ("noise_figure", "bandwidth")
    
    def __init__(self, noise_figure: float = 3.0, bandwidth: float = 1e6) -> None:
        super().__init__()


class TargetComponent(RadarComponent):
    __slots__ = ("rcs_sqm",)
    
    def __init__(self, rcs_sqm: float = 1.0) -> None:
        super().__init__()
        self.rcs_sqm = rcs_sqm

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Protocol, Iterable
from numpy.typing import NDArray
import numpy as np

if TYPE_CHECKING:
    from .entity import Entity

class RadarComponent():
    __slots__ = ("_entity",)
    
    def __init__(self) -> None:
        self._entity: Entity | None = None
    
    @property
    def entity(self) -> Entity:
        if self._entity is None:
            raise RuntimeError(f"Component not attached.")
        
        return self._entity
    
    @property
    def attached(self) -> bool:
        return self._entity is not None
    
    def on_attach(self, entity) -> None:
        self._entity = entity

    def on_detach(self) -> None:
        self._entity = None
    
class TargetProperties(RadarComponent):
    __slots__ = ("rcs",)
    
    def __init__(
        self, 
        rcs: float = 1.0,
    ) -> None:
        
        super().__init__()
        self.rcs = rcs   


class RadarNode(RadarComponent):
    __slots__ = ("node_name", "transmitters", "receivers")
    
    def __init__(
        self,
        node_name: str,
        transmitters: Iterable[TransmitterElement] | None = None,
        receivers: Iterable[ReceiverElement] | None = None,
    ) -> None:
        
        super().__init__()
        self.node_name = node_name
        self.transmitters: list[TransmitterElement] = []
        self.receivers: list[ReceiverElement] = []
        
        if transmitters:
            for tx in transmitters:
                self.add_transmitter(tx)
        
        if receivers:
            for rx in receivers:
                self.add_receiver(rx)
    
    def add_transmitter(self, tx: TransmitterElement) -> None:
        tx.bind_to_node(self)
        self.transmitters.append(tx)
    
    def add_receiver(self, rx: ReceiverElement) -> None:
        rx.bind_to_node(self)
        self.receivers.append(rx)
    
    @property    
    def has_transmitters(self) -> bool:
        return bool(self.transmitters)

    @property
    def has_receivers(self) -> bool:
        return bool(self.receivers)



class SensorElement:
    
    __slots__ = ("_node", "index", "pos_offset", "rot_offset")
    
    def __init__(
        self,
        pos_offset: NDArray | None = None,
        rot_offset: NDArray | None = None,
    ) -> None:
        self._node: RadarNode | None = None
        self.index: int | None = None # global slot, assigned by RadarNetwork
        
        self.pos_offset = np.zeros(3) if pos_offset is None else pos_offset 
        self.rot_offset = (
            np.array([1.0, 0.0, 0.0, 0.0])
            if rot_offset is None
            else rot_offset
        )
        
    @property
    def node(self) -> RadarNode:
        if self._node is None:
            raise RuntimeError(f"{type(self).__name__} is not bound to RadarNode.")
        return self._node
    
    @property
    def entity(self) -> Entity:
        return self.node.entity
    
    @property
    def slot(self) -> int:
        return self.node.entity.slot

    @property
    def attached(self) -> bool:
        return self._node is not None and self._node.attached
    
    def bind_to_node(self, node: RadarNode) -> None:
        self._node = node
        
    def on_update(self, node: RadarNode) -> None:
        self._node = node

class TransmitterElement(SensorElement):
    __slots__ = ("frequency", "bandwidth", "peak_power", "gain_pattern")
    
    def __init__(
        self,
        frequency: float = 10e9,
        bandwidth: float = 10e6,
        peak_power: float = 1000.0,
        gain_pattern: Any = None,
        pos_offset: NDArray | None = None,
        rot_offset: NDArray | None = None,
        
    ) -> None:
    
        super().__init__(pos_offset, rot_offset)
        
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.peak_power = peak_power
        self.gain_pattern = gain_pattern

class ReceiverElement(SensorElement):
    __slots__ = ("centre_frequency", "bandwidth", "noise_figure", "gain_pattern")
    
    def __init__(
        self,
        centre_frequency: float = 10e9,
        bandwidth: float = 10e6,
        noise_figure: float = 3.0,
        gain_pattern: Any = None,
        pos_offset: NDArray | None = None,
        rot_offset: NDArray | None = None,
    ) -> None:
        
        super().__init__(pos_offset, rot_offset)
        self.centre_frequency = centre_frequency
        self.bandwidth = bandwidth
        self.noise_figure = noise_figure
        self.gain_pattern = gain_pattern





        
            
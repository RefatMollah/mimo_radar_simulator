from __future__ import annotations
from typing import TYPE_CHECKING, Any, Protocol, Iterable, TypeAlias
import numpy as np

from ..geometry.sensor_steering import FixedBoresight, ScanningPattern, SteeringActuator
from .exceptions import (
    ComponentAlreadyAttachedError,
    ComponentError,
)

if TYPE_CHECKING:
    from .entity import Entity
    
SPEED_OF_LIGHT = 299_792_458.0

ArrayLike: TypeAlias = Any

class Component:
    """
    Base class for all entity components.

    Components are owned by an Entity. They should not be attached to
    more than one entity at a time.
    """

    __slots__ = ("_entity",)

    def __init__(self) -> None:
        self._entity: Entity | None = None

    @property
    def entity(self) -> Entity:
        if self._entity is None:
            raise ComponentError(
                f"{type(self).__name__} is not attached to an Entity."
            )
        return self._entity

    @property
    def entity_or_none(self) -> Entity | None:
        return self._entity

    @property
    def is_attached(self) -> bool:
        return self._entity is not None

    def _attach(self, entity: Entity) -> None:
        if self._entity is not None:
            raise ComponentAlreadyAttachedError(
                f"{type(self).__name__} is already attached to {self._entity.id}."
            )
        self._entity = entity
        self.on_attach(entity)

    def _detach(self) -> None:
        entity = self._entity
        self._entity = None
        self.on_detach(entity)

    def on_attach(self, entity: Entity) -> None:
        """Override in subclasses if needed."""
        pass

    def on_detach(self, entity: Entity | None) -> None:
        """Override in subclasses if needed."""
        pass
    

class RadarTarget(Component):
    """
    Marks an entity as radar-visible/scatterable.

    Initially this just holds scalars RCS. Later it should probably hold
    an RcsModel or ScatteringModel supporting aspect/frequency dependence,
    Swerling fluctuation, micro-Doppler, scattering centers, etc.
    """

    __slots__ = ("rcs",)

    def __init__(self, rcs: float = 1.0) -> None:
        super().__init__()
        self.rcs = rcs


class RadarSensor(Component):
    """
    Radar payload attached to an entity.
    A RadarSensor is an RF front-end container. Its pose comes from the owning
    entity and its RF elements carry any element-level offsets. Steering is a
    separate scanning law or actuator.
    """

    __slots__ = (
        "name",
        "steering",
        "_tx",
        "_rx",
    )

    def __init__(
        self,
        name: str = "",
        *,
        steering: ScanningPattern | SteeringActuator | None = None,
        transmitters: Iterable[TxElement] | None = None,
        receivers: Iterable[RxElement] | None = None,
    ) -> None:
        super().__init__()

        self.name = name
        self.steering = FixedBoresight() if steering is None else steering

        self._tx: list[TxElement] = []
        self._rx: list[RxElement] = []

        if transmitters is not None:
            for tx in transmitters:
                self.add_tx(tx)

        if receivers is not None:
            for rx in receivers:
                self.add_rx(rx)

    @property
    def tx_elements(self) -> tuple[TxElement, ...]:
        return tuple(self._tx)

    @property
    def rx_elements(self) -> tuple[RxElement, ...]:
        return tuple(self._rx)

    @property
    def num_tx(self) -> int:
        return len(self._tx)

    @property
    def num_rx(self) -> int:
        return len(self._rx)

    @property
    def has_tx(self) -> bool:
        return bool(self._tx)

    @property
    def has_rx(self) -> bool:
        return bool(self._rx)

    def add_tx(self, tx: TxElement) -> None:
        if tx.is_bound:
            raise ComponentAlreadyAttachedError(
                f"{tx!r} is already bound to a RadarSensor."
            )

        tx._bind(self, local_index=len(self._tx))
        self._tx.append(tx)

    def add_rx(self, rx: RxElement) -> None:
        if rx.is_bound:
            raise ComponentAlreadyAttachedError(
                f"{rx!r} is already bound to a RadarSensor."
            )

        rx._bind(self, local_index=len(self._rx))
        self._rx.append(rx)


class RfElement:
    """
    Base class for Tx/Rx elements.

    Owned by RadarSensor, not Entity.
    """

    __slots__ = (
        "_sensor",
        "local_index",
        "local_position",
        "local_rotation",
        "gain_pattern",
    )

    def __init__(
        self,
        *,
        local_position: ArrayLike | None = None,
        local_rotation: ArrayLike | None = None,
        gain_pattern: object | None = None,
    ) -> None:
        self._sensor: RadarSensor | None = None
        self.local_index: int | None = None

        self.local_position = (
            np.zeros(3, dtype=np.float64)
            if local_position is None
            else np.asarray(local_position, dtype=np.float64)
        )

        self.local_rotation = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            if local_rotation is None
            else np.asarray(local_rotation, dtype=np.float64)
        )

        if self.local_position.shape != (3,):
            raise ValueError("local_position must have shape (3,).")

        if self.local_rotation.shape != (4,):
            raise ValueError("local_rotation must have shape (4,).")

        self.gain_pattern = gain_pattern

    @property
    def is_bound(self) -> bool:
        return self._sensor is not None

    @property
    def sensor(self) -> RadarSensor:
        if self._sensor is None:
            raise ComponentError(
                f"{type(self).__name__} is not bound to a RadarSensor."
            )
        return self._sensor

    @property
    def entity(self) -> Entity:
        return self.sensor.entity

    def _bind(self, sensor: RadarSensor, local_index: int) -> None:
        if self._sensor is not None:
            raise ComponentAlreadyAttachedError(
                f"{type(self).__name__} is already bound to a RadarSensor."
            )

        self._sensor = sensor
        self.local_index = local_index


class TxElement(RfElement):
    __slots__ = (
        "carrier_frequency_hz",
        "bandwidth_hz",
        "peak_power_w",
        "waveform",
    )

    def __init__(
        self,
        *,
        carrier_frequency_hz: float = 10e9,
        bandwidth_hz: float = 10e6,
        peak_power_w: float = 1000.0,
        waveform: object | None = None,
        gain_pattern: object | None = None,
        local_position: ArrayLike | None = None,
        local_rotation: ArrayLike | None = None,
    ) -> None:
        super().__init__(
            local_position=local_position,
            local_rotation=local_rotation,
            gain_pattern=gain_pattern,
        )

        self.carrier_frequency_hz = carrier_frequency_hz
        self.bandwidth_hz = bandwidth_hz
        self.peak_power_w = peak_power_w
        self.waveform = waveform

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / self.carrier_frequency_hz
    

class RxElement(RfElement):
    __slots__ = (
        "center_frequency_hz",
        "bandwidth_hz",
        "noise_figure_db",
    )

    def __init__(
        self,
        *,
        center_frequency_hz: float = 10e9,
        bandwidth_hz: float = 10e6,
        noise_figure_db: float = 3.0,
        gain_pattern: object | None = None,
        local_position: ArrayLike | None = None,
        local_rotation: ArrayLike | None = None,
    ) -> None:
        super().__init__(
            local_position=local_position,
            local_rotation=local_rotation,
            gain_pattern=gain_pattern,
        )
        self.center_frequency_hz = center_frequency_hz
        self.bandwidth_hz = bandwidth_hz
        self.noise_figure_db = noise_figure_db

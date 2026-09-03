"""Scanning laws and steering actuators for radar RF elements.

Angles use radians. A steering quaternion rotates the element boresight
(``+x``) into the requested azimuth/elevation direction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType
from typing import Any, NamedTuple, Sequence, cast

import numpy as np

from .._array import ArrayLike, DTypeLike, is_jax_namespace
from ._jax_backend import register_steering_batch
from .quat import identity_quats


class ScanningPattern(ABC):
    """Base class for a stateless scanning patterns."""


def _validate_rate(rate: float) -> None:
    if rate < 0:
        raise ValueError("rate must be non-negative")


def _az_el_quaternions(azimuth: Any, elevation: Any, xp: ModuleType, dtype: DTypeLike) -> ArrayLike:
    half_az, half_el = azimuth / 2.0, elevation / 2.0
    cz, sz, ce, se = xp.cos(half_az), xp.sin(half_az), xp.cos(half_el), xp.sin(half_el)
    # qz(azimuth) * qy(-elevation), [w, x, y, z].
    return xp.stack((cz * ce, sz * se, -cz * se, sz * ce), axis=-1).astype(dtype)


@dataclass(frozen=True, slots=True)
class FixedBoresight(ScanningPattern):
    """Static sensors"""
    azimuth: float = 0.0
    elevation: float = 0.0
    xp: ModuleType = np
    dtype: DTypeLike = np.float32


@dataclass(frozen=True, slots=True)
class CircularScan(ScanningPattern):
    """Continuous 360-degree azimuth scan at a fixed elevation and angular rate."""
    rate: float
    elevation: float = 0.0
    azimuth0: float = 0.0
    xp: ModuleType = np
    dtype: DTypeLike = np.float32

    def __post_init__(self) -> None:
        _validate_rate(self.rate)


@dataclass(frozen=True, slots=True)
class SectorScan(ScanningPattern):
    """Constant-rate back-and-forth scan between azimuth limits with fixed elevation."""
    azimuth_start: float
    azimuth_end: float
    rate: float
    elevation: float = 0.0
    xp: ModuleType = np
    dtype: DTypeLike = np.float32

    def __post_init__(self) -> None:
        _validate_rate(self.rate)
        if self.azimuth_end <= self.azimuth_start:
            raise ValueError("azimuth_end must be greater than azimuth_start")


FullAzimuthScan = CircularScan

@dataclass(frozen=True, slots=True)
class ConicalScan(ScanningPattern):
    """Placeholder for a conical steering law."""
    axis: ArrayLike
    half_angle: float
    rate: float
    phase0: float = 0.0
    xp: ModuleType = np
    dtype: DTypeLike = np.float32


@dataclass(frozen=True, slots=True)
class ScanningBlock:
    orientations: ArrayLike


@dataclass(frozen=True, slots=True, kw_only=True)
class ScanningPatternBatch(ABC):
    xp: ModuleType = np
    dtype: DTypeLike = np.float32

    @classmethod
    @abstractmethod
    def from_components(cls, components: Sequence[Any], dtype: DTypeLike, xp: ModuleType) -> "ScanningPatternBatch": ...

    @abstractmethod
    def evaluate(self, time: ArrayLike) -> ScanningBlock: ...


@register_steering_batch(FixedBoresight)
@dataclass(frozen=True, slots=True)
class FixedBoresightBatch(ScanningPatternBatch):
    azimuths: ArrayLike
    elevations: ArrayLike

    @classmethod
    def from_components(cls, components: Sequence[Any], dtype: DTypeLike, xp: ModuleType) -> "FixedBoresightBatch":
        laws = [cast(FixedBoresight, c.steering) for c in components]
        return cls(azimuths=xp.asarray([l.azimuth for l in laws], dtype=dtype), elevations=xp.asarray([l.elevation for l in laws], dtype=dtype), xp=xp, dtype=dtype)

    def evaluate(self, time: ArrayLike) -> ScanningBlock:
        del time
        return ScanningBlock(_az_el_quaternions(self.azimuths, self.elevations, self.xp, self.dtype))


@register_steering_batch(CircularScan)
@dataclass(frozen=True, slots=True)
class CircularScanBatch(ScanningPatternBatch):
    rates: ArrayLike
    elevations: ArrayLike
    azimuth0s: ArrayLike

    @classmethod
    def from_components(cls, components: Sequence[Any], dtype: DTypeLike, xp: ModuleType) -> "CircularScanBatch":
        laws = [cast(CircularScan, c.steering) for c in components]
        return cls(rates=xp.asarray([l.rate for l in laws], dtype=dtype), elevations=xp.asarray([l.elevation for l in laws], dtype=dtype), azimuth0s=xp.asarray([l.azimuth0 for l in laws], dtype=dtype), xp=xp, dtype=dtype)

    def evaluate(self, time: ArrayLike) -> ScanningBlock:
        t = self.xp.asarray(time, dtype=self.dtype)
        return ScanningBlock(_az_el_quaternions(self.azimuth0s + self.rates * t, self.elevations, self.xp, self.dtype))


@register_steering_batch(SectorScan)
@dataclass(frozen=True, slots=True)
class SectorScanBatch(ScanningPatternBatch):
    starts: ArrayLike
    ends: ArrayLike
    rates: ArrayLike
    elevations: ArrayLike

    @classmethod
    def from_components(cls, components: Sequence[Any], dtype: DTypeLike, xp: ModuleType) -> "SectorScanBatch":
        laws = [cast(SectorScan, c.steering) for c in components]
        return cls(starts=xp.asarray([l.azimuth_start for l in laws], dtype=dtype), ends=xp.asarray([l.azimuth_end for l in laws], dtype=dtype), rates=xp.asarray([l.rate for l in laws], dtype=dtype), elevations=xp.asarray([l.elevation for l in laws], dtype=dtype), xp=xp, dtype=dtype)

    def evaluate(self, time: ArrayLike) -> ScanningBlock:
        xp = self.xp
        t = xp.asarray(time, dtype=self.dtype)
        width = self.ends - self.starts
        phase = xp.mod(self.rates * t, 2.0 * width)
        azimuth = self.starts + xp.where(phase <= width, phase, 2.0 * width - phase)
        return ScanningBlock(_az_el_quaternions(azimuth, self.elevations, xp, self.dtype))


@register_steering_batch(ConicalScan)
@dataclass(frozen=True, slots=True)
class ConicalScanBatch(ScanningPatternBatch):
    axes: ArrayLike
    half_angles: ArrayLike
    rates: ArrayLike
    phase0s: ArrayLike

    @classmethod
    def from_components(cls, components: Sequence[Any], dtype: DTypeLike, xp: ModuleType) -> "ConicalScanBatch":
        laws = [cast(ConicalScan, c.steering) for c in components]
        return cls(axes=xp.asarray([l.axis for l in laws], dtype=dtype), half_angles=xp.asarray([l.half_angle for l in laws], dtype=dtype), rates=xp.asarray([l.rate for l in laws], dtype=dtype), phase0s=xp.asarray([l.phase0 for l in laws], dtype=dtype), xp=xp, dtype=dtype)

    # TODO: Implement conical scan
    def evaluate(self, time: ArrayLike) -> ScanningBlock:
        return ScanningBlock(identity_quats(self.axes.shape[0], like=self.axes, dtype=self.dtype))

#-------------------------------------------------------------------
# Closed-Loop Steering actuators
#-------------------------------------------------------------------
class SteeringCommand(NamedTuple):
    az_rate: ArrayLike
    el_rate: ArrayLike


class SteeringState(NamedTuple):
    azimuth: ArrayLike
    elevation: ArrayLike
    az_rate: ArrayLike
    el_rate: ArrayLike


@dataclass(frozen=True, slots=True, kw_only=True)
class SteeringActuator(ABC):
    xp: ModuleType = np
    dtype: DTypeLike = np.float32

    @abstractmethod
    def step(self, state: SteeringState, command: SteeringCommand, dt: float) -> SteeringState: ...


@dataclass(frozen=True, slots=True)
class RateLimitedGimbal(SteeringActuator):
    max_rate: float
    max_accel: float

    def step(self, state: SteeringState, command: SteeringCommand, dt: float) -> SteeringState:
        xp = self.xp
        def slew(curr: Any, target: Any) -> Any:
            return curr + xp.clip(target - curr, -self.max_accel * dt, self.max_accel * dt)
        az_rate = xp.clip(slew(state.az_rate, command.az_rate), -self.max_rate, self.max_rate)
        el_rate = xp.clip(slew(state.el_rate, command.el_rate), -self.max_rate, self.max_rate)
        return SteeringState(state.azimuth + az_rate * dt, state.elevation + el_rate * dt, az_rate, el_rate)


class SteeringActuatorBatch(ABC):
    xp: ModuleType

    @abstractmethod
    def step(self, states: SteeringState, commands: SteeringCommand, dt: float) -> SteeringState: ...


@dataclass(frozen=True, slots=True)
class RateLimitedGimbalBatch(SteeringActuatorBatch):
    max_rates: ArrayLike
    max_accels: ArrayLike
    xp: ModuleType = np
    dtype: DTypeLike = np.float32

    def step(self, states: SteeringState, commands: SteeringCommand, dt: float) -> SteeringState:
        xp = self.xp
        az_rate = xp.clip(states.az_rate + xp.clip(commands.az_rate - states.az_rate, -self.max_accels * dt, self.max_accels * dt), -self.max_rates, self.max_rates)
        el_rate = xp.clip(states.el_rate + xp.clip(commands.el_rate - states.el_rate, -self.max_accels * dt, self.max_accels * dt), -self.max_rates, self.max_rates)
        return SteeringState(states.azimuth + az_rate * dt, states.elevation + el_rate * dt, az_rate, el_rate)


def rollout_actuators(actuator_batch: SteeringActuatorBatch, init_states: SteeringState, commands_over_time: Any, dt: float) -> tuple[SteeringState, Any]:
    if is_jax_namespace(actuator_batch.xp):
        import jax
        def step(state: SteeringState, command: SteeringCommand):
            new_state = actuator_batch.step(state, command, dt)
            return new_state, new_state
        return jax.lax.scan(step, init_states, commands_over_time)
    state = init_states
    history = []
    for command in commands_over_time:
        state = actuator_batch.step(state, command, dt)
        history.append(state)
    return state, history

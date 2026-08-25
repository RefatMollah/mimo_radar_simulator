from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias, Sequence, cast, NamedTuple
from numpy.typing import DTypeLike

from .jax_backend import register_steering_batch
from .quat import identity_quats
from ..scene.radar_network import RadarNetwork, ChannelLink

ArrayLike: TypeAlias = Any
Backend: TypeAlias = Any

class SteeringLaw(ABC):
    """Base class for stateless, closed-form steering laws."""
    

@dataclass(frozen=True, slots=True)
class FixedBoresight(SteeringLaw):
    """Fixed boresight relative to the mount."""
    xp: Backend = np
    dtype: DTypeLike = np.float32

@dataclass(frozen=True, slots=True)
class ConicalScan(SteeringLaw):
    """Conical scan steering law."""
    axis: ArrayLike          # scan cone axis, mount frame (unit vector)
    half_angle: float
    rate: float              # rad/s
    phase0: float = 0.0
    xp: Backend = np
    dtype: DTypeLike = np.float32

@dataclass(frozen=True, slots=True)
class SteeringBlock:
    """Output of a steering law evaluation."""
    orientations: ArrayLike  # (N, 4) quaternions
    
@dataclass(frozen=True, slots=True, kw_only=True)
class SteeringLawBatch(ABC):
    """Batched evaluations for stateless steering laws."""
    xp: Backend = np
    dtype: DTypeLike = np.float32

    @classmethod
    @abstractmethod
    def from_components(cls, components: Sequence[Any], dtype: DTypeLike, xp: Backend) -> SteeringLawBatch: ...

    @abstractmethod
    def evaluate(self, time: ArrayLike) -> SteeringBlock: ...
    

@register_steering_batch(FixedBoresight)
@dataclass(frozen=True, slots=True)
class FixedBoresightBatch(SteeringLawBatch):
    n: int
    
    @classmethod
    def from_components(cls, components: Sequence[Any], dtype: DTypeLike, xp: Backend) -> FixedBoresightBatch:
        return cls(n=len(components), xp=xp, dtype=dtype)
        
    def evaluate(self, time: ArrayLike) -> SteeringBlock:
        return SteeringBlock(orientations=identity_quats(self.n, self.xp, self.dtype))

@register_steering_batch(ConicalScan)
@dataclass(frozen=True, slots=True)
class ConicalScanBatch(SteeringLawBatch):
    axes: ArrayLike          # (N, 3)
    half_angles: ArrayLike   # (N,)
    rates: ArrayLike         # (N,)
    phase0s: ArrayLike       # (N,)
    
    @classmethod
    def from_components(cls, components: Sequence[Any], dtype: DTypeLike, xp: Backend) -> ConicalScanBatch:
        laws = [cast(ConicalScan, c.steering) for c in components]
        return cls(
            axes=xp.stack([l.axis for l in laws]).astype(dtype),
            half_angles=xp.array([l.half_angle for l in laws], dtype=dtype),
            rates=xp.array([l.rate for l in laws], dtype=dtype),
            phase0s=xp.array([l.phase0 for l in laws], dtype=dtype),
            xp=xp,
            dtype=dtype
        )
        
    def evaluate(self, time: ArrayLike) -> SteeringBlock:
        xp = self.xp
        dtype = self.dtype
        t = xp.asarray(time, dtype=dtype)
        
        #TODO: Implement full conical scan
        phase = self.rates * t + self.phase0s
        
        # Placeholder: returning identity quats. Replace with actual conical scan quat generation.
        return SteeringBlock(orientations=identity_quats(self.axes.shape[0], xp, dtype))

        
# -----------------------------------------------------------------------------
# 2. Closed-Loop Steering (Stateful Actuators)
# -----------------------------------------------------------------------------
class SteeringCommand(NamedTuple):
    az_rate: ArrayLike
    el_rate: ArrayLike

class SteeringState(NamedTuple):
    azimuth:   ArrayLike
    elevation: ArrayLike
    az_rate:   ArrayLike
    el_rate:   ArrayLike


@dataclass(frozen=True, slots=True, kw_only=True)    
class SteeringActuator(ABC):
    """Base class for stateful steering plants."""
    xp: Backend
    dtype: DTypeLike
    @abstractmethod
    def step(self, state: SteeringState, command: SteeringCommand, dt: float) -> SteeringState: ...
   
  
@dataclass(frozen=True, slots=True)
class RateLimitedGimbal(SteeringActuator):
    max_rate: float
    max_accel: float
    xp: Backend = np
    dtype: DTypeLike = np.float32

    def step(self, state: SteeringState, command: SteeringCommand, dt: float) -> SteeringState:
        xp = self.xp
        
        def _slew_limit(curr: Any, tgt: Any, max_a: float, dt: float) -> Any:
            diff = tgt - curr
            max_delta = max_a * dt
            return curr + xp.clip(diff, -max_delta, max_delta)
            
        az_rate = _slew_limit(state.az_rate, command.az_rate, self.max_accel, dt)
        el_rate = _slew_limit(state.el_rate, command.el_rate, self.max_accel, dt)
        
        az_rate = xp.clip(az_rate, -self.max_rate, self.max_rate)
        el_rate = xp.clip(el_rate, -self.max_rate, self.max_rate)
        
        return SteeringState(
            azimuth=state.azimuth + az_rate * dt,
            elevation=state.elevation + el_rate * dt,
            az_rate=az_rate,
            el_rate=el_rate
        )

class SteeringActuatorBatch(ABC):
    xp: Backend
    @abstractmethod
    def step(self, states: SteeringState, commands: SteeringCommand, dt: float) -> SteeringState: ...

@dataclass(frozen=True, slots=True)
class RateLimitedGimbalBatch(SteeringActuatorBatch):
    max_rates: ArrayLike
    max_accels: ArrayLike
    xp: Backend = np
    dtype: DTypeLike = np.float32

    def step(self, states: SteeringState, commands: SteeringCommand, dt: float) -> SteeringState:
        xp = self.xp
        
        def _slew_limit(curr: Any, tgt: Any, max_a: Any, dt: float) -> Any:
            diff = tgt - curr
            max_delta = max_a * dt
            return curr + xp.clip(diff, -max_delta, max_delta)
            
        az_rate = _slew_limit(states.az_rate, commands.az_rate, self.max_accels, dt)
        el_rate = _slew_limit(states.el_rate, commands.el_rate, self.max_accels, dt)
        
        az_rate = xp.clip(az_rate, -self.max_rates, self.max_rates)
        el_rate = xp.clip(el_rate, -self.max_rates, self.max_rates)
        
        return SteeringState(
            azimuth=states.azimuth + az_rate * dt,
            elevation=states.elevation + el_rate * dt,
            az_rate=az_rate,
            el_rate=el_rate
        )

def rollout_actuators(
    actuator_batch: SteeringActuatorBatch, 
    init_states: SteeringState, 
    commands_over_time: Any, 
    dt: float
) -> tuple[SteeringState, Any]:
    """Rolls out the actuator batch over time. Uses lax.scan for JAX, Python loop for NumPy."""
    if actuator_batch.xp.__name__ == "jax.numpy":
        import jax
        def step_fn(states: SteeringState, commands: SteeringCommand):
            new_states = actuator_batch.step(states, commands, dt)
            return new_states, new_states
        
        return jax.lax.scan(step_fn, init_states, commands_over_time)
    else:
        states = init_states
        history = []
        for commands in commands_over_time:
            states = actuator_batch.step(states, commands, dt)
            history.append(states)
        return states, history
      
    
    
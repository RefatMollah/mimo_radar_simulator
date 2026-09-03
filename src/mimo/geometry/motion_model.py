from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

from types import ModuleType
from typing import Sequence, cast, Any, TYPE_CHECKING
from numpy.typing import NDArray

from .._array import ArrayInput, ArrayLike, DTypeLike
from .quat import quat_multiply, quat_normalise, axis_angle_delta_quat
from ._jax_backend import register_motion_batch, batch_class_for, _ensure_all_pytrees_registered

if TYPE_CHECKING:
    from ..entity.entity import Entity


#------------------------------------------------
# Motion Classes
#------------------------------------------------

class Motion(ABC):
    
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class StaticMotion(Motion):
    """A fixed, time-invariant state."""

    position: ArrayInput
    orientation: ArrayInput  # quaternion [w, x, y, z]
    
    xp: ModuleType = np
    dtype: DTypeLike = np.float32

    def __post_init__(self) -> None:
        _coerce_vec3(self, "position", self.xp, self.dtype)
        _coerce_quat(self, "orientation", self.xp, self.dtype)
    

@dataclass(frozen=True, slots=True)
class ConstantVelocityMotion(Motion):
    initial_position: ArrayInput
    initial_velocity: ArrayInput
    initial_orientation: ArrayInput
    angular_velocity: ArrayInput
    initial_time: float = 0.0
    
    xp: ModuleType = np
    dtype: DTypeLike = np.float32

    def __post_init__(self) -> None:
        _coerce_vec3(self, "initial_position", self.xp, self.dtype)
        _coerce_vec3(self, "initial_velocity", self.xp, self.dtype)
        _coerce_quat(self, "initial_orientation", self.xp, self.dtype)
        _coerce_vec3(self, "angular_velocity", self.xp, self.dtype)


@dataclass(frozen=True, slots=True)
class ConstantAccelerationMotion(Motion):
    initial_position: ArrayInput
    initial_velocity: ArrayInput
    acceleration: ArrayInput
    initial_orientation: ArrayInput
    angular_velocity: ArrayInput
    initial_time: float = 0.0
    
    xp: ModuleType = np
    dtype: DTypeLike = np.float32

    def __post_init__(self) -> None:
        _coerce_vec3(self, "initial_position", self.xp, self.dtype)
        _coerce_vec3(self, "initial_velocity", self.xp, self.dtype)
        _coerce_vec3(self, "acceleration", self.xp, self.dtype)
        _coerce_quat(self, "initial_orientation", self.xp, self.dtype)
        _coerce_vec3(self, "angular_velocity", self.xp, self.dtype)

    
def _coerce_array(obj: object, name: str, expected_shape: tuple[int, ...], xp: ModuleType = np, dtype: DTypeLike = np.float32) -> None:
    value = xp.asarray(getattr(obj, name), dtype=dtype)
    if value.shape != expected_shape:
        raise ValueError(
            f"{type(obj).__name__}.{name} must have shape {expected_shape};"
            f"got {value.shape}."
        )
    object.__setattr__(obj, name, value)

def _coerce_vec3(obj: object, name: str, xp: ModuleType = np, dtype: DTypeLike = np.float32) -> None:
    _coerce_array(obj, name, (3,), xp, dtype)

def _coerce_quat(obj: object, name: str, xp: ModuleType = np, dtype: DTypeLike = np.float32) -> None:
    _coerce_array(obj, name, (4,), xp, dtype)
    norm = xp.linalg.norm(getattr(obj, name))
    if not xp.isclose(norm, 1.0, atol=1e-6):
        raise ValueError(
            f"{type(obj).__name__}.{name} must be a normalised quaternion "
            f"(unit norm); got norm={norm!r}."     
        )


#------------------------------------------------
# Motion Batches
#------------------------------------------------

@dataclass
class MotionBlock:
    """Kinematic result for one motion kind's entities, in the same row
    order as that kind's batch (and its slots_by_kind index array)."""

    positions: ArrayLike
    velocities: ArrayLike
    accelerations: ArrayLike
    orientations: ArrayLike
    angular_rates: ArrayLike


def build_batch(motion_cls: type[Motion], entities: Sequence[Entity], dtype: DTypeLike, xp: ModuleType = np) -> MotionBatch:
    # Ensure the batch is created using the provided backend `xp` so that
    # JAX dispatch uses `jax.numpy` (jnp) and NumPy dispatch uses `numpy`.
    return batch_class_for(motion_cls).from_entities(entities, dtype, xp)


@dataclass(frozen=True, slots=True, kw_only=True)
class MotionBatch(ABC):
    xp: ModuleType = np
    dtype: DTypeLike = np.float32
    
    @classmethod
    @abstractmethod
    def from_entities(cls, entities: Sequence[Entity], dtype: DTypeLike, xp: ModuleType = np) -> MotionBatch:
        ...
    
    @abstractmethod
    def evaluate(self, time: ArrayLike) -> MotionBlock:
        ...
    

@register_motion_batch(StaticMotion)
@dataclass(frozen=True, slots=True)
class StaticBatch(MotionBatch):
    positions: ArrayLike
    orientations: ArrayLike
    
    @classmethod
    def from_entities(cls, entities: Sequence[Entity], dtype: DTypeLike, xp: ModuleType = np) -> StaticBatch:
        motions = [cast(StaticMotion, e.motion) for e in entities]
        return cls(
            positions=xp.stack([m.position for m in motions]).astype(dtype),
            orientations=xp.stack([m.orientation for m in motions]).astype(dtype),
            
            xp=xp,
            dtype=dtype,         
        )
    
    def evaluate(self, time: Any) -> MotionBlock:
        xp = self.xp
        
        del time
        return MotionBlock(
            positions=self.positions,
            velocities=xp.zeros_like(self.positions),
            accelerations=xp.zeros_like(self.positions),
            orientations=self.orientations,
            angular_rates=xp.zeros_like(self.positions),          
        )


@register_motion_batch(ConstantVelocityMotion)
@dataclass(frozen=True, slots=True)
class ConstantVelocityBatch(MotionBatch):
    initial_positions: ArrayLike
    initial_velocities: ArrayLike
    initial_orientations: ArrayLike
    angular_rates: ArrayLike
    initial_times: ArrayLike
    
    @classmethod
    def from_entities(cls, entities: Sequence[Entity], dtype: DTypeLike, xp: ModuleType = np) -> ConstantVelocityBatch:
        motions = [cast(ConstantVelocityMotion, e.motion) for e in entities]
        return cls(
            initial_positions=xp.stack([m.initial_position for m in motions]).astype(dtype),
            initial_velocities=xp.stack([m.initial_velocity for m in motions]).astype(dtype),
            initial_orientations=xp.stack([m.initial_orientation for m in motions]).astype(dtype),
            angular_rates=xp.stack([m.angular_velocity for m in motions]).astype(dtype),
            initial_times=xp.array([m.initial_time for m in motions], dtype=dtype),
            
            xp=xp,
            dtype=dtype,             
        )
        
    def evaluate(self, time: Any) -> MotionBlock:
        xp = self.xp
        dtype = self.dtype
        
        dt = time - self.initial_times
        dt_col = dt[:, None]
        positions = self.initial_positions + self.initial_velocities * dt_col
        dq = axis_angle_delta_quat(self.angular_rates, dt, dtype=dtype)
        orientations = quat_normalise(quat_multiply(self.initial_orientations, dq, dtype=dtype))
        
        return MotionBlock(
            positions=positions,
            velocities=self.initial_velocities,
            accelerations=xp.zeros_like(positions),
            orientations=orientations,
            angular_rates=self.angular_rates,            
        )


@register_motion_batch(ConstantAccelerationMotion)
@dataclass(frozen=True, slots=True)
class ConstantAccelerationBatch(MotionBatch):
    initial_positions: ArrayLike
    initial_velocities: ArrayLike
    accelerations: ArrayLike
    initial_orientations: ArrayLike
    angular_rates: ArrayLike
    initial_times: ArrayLike
        
    @classmethod
    def from_entities(cls, entities: Sequence[Entity], dtype: DTypeLike, xp: ModuleType = np) -> MotionBatch:
        motions = [cast(ConstantAccelerationMotion, e.motion) for e in entities]
        return cls(
            initial_positions=xp.stack([m.initial_position for m in motions]).astype(dtype),
            initial_velocities=xp.stack([m.initial_velocity for m in motions]).astype(dtype),
            accelerations=xp.stack([m.acceleration for m in motions]).astype(dtype),
            initial_orientations=xp.stack([m.initial_orientation for m in motions]).astype(dtype),
            angular_rates=xp.stack([m.angular_velocity for m in motions]).astype(dtype),
            initial_times=xp.array([m.initial_time for m in motions], dtype=dtype),
            
            xp=xp,
            dtype=dtype,             
        )
    
    def evaluate(self, time: Any) -> MotionBlock:
        xp = self.xp
        dtype = self.dtype
        
        dt = time - self.initial_times
        dt_col = dt[:, None]
        positions = self.initial_positions + self.initial_velocities * dt_col + 0.5 * self.accelerations * dt_col ** 2
        velocities = self.initial_velocities + self.accelerations * dt_col
        dq = axis_angle_delta_quat(self.angular_rates, dt, dtype=dtype)
        orientations = quat_normalise(quat_multiply(self.initial_orientations, dq, dtype=dtype))
        
        return MotionBlock(
            positions=positions,
            velocities=velocities,
            accelerations=self.accelerations,
            orientations=orientations,
            angular_rates=self.angular_rates,            
        )
    
_ensure_all_pytrees_registered()

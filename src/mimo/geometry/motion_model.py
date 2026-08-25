from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

from typing import Sequence, cast, TypeAlias, Any, Union, TYPE_CHECKING, TypeVar, Callable
from numpy.typing import NDArray, DTypeLike

from .quat import quat_multiply, quat_normalise, axis_angle_delta_quat
from .jax_backend import register_motion_batch, batch_class_for, _ensure_all_pytrees_registered

Backend: TypeAlias = Any

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from ..entity.entity import Entity
else:
    JaxArray = Any
    
ArrayLike: TypeAlias = Union[NDArray, JaxArray]


#------------------------------------------------
# Motion Classes
#------------------------------------------------

class Motion(ABC):
    
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class StaticMotion(Motion):
    """A fixed, time-invariant state."""

    position: ArrayLike
    velocity: ArrayLike
    orientation: ArrayLike  # quaternion [w, x, y, z]
    angular_velocity: ArrayLike
    
    xp: Backend = np
    dtype: DTypeLike = np.float32

    def __post_init__(self) -> None:
        _coerce_vec3(self, "position", self.xp, self.dtype)
        _coerce_vec3(self, "velocity", self.xp, self.dtype)
        _coerce_quat(self, "orientation", self.xp, self.dtype)
        _coerce_vec3(self, "angular_velocity", self.xp, self.dtype)
    


@dataclass(frozen=True, slots=True)
class ConstantVelocityMotion(Motion):
    initial_position: ArrayLike
    initial_velocity: ArrayLike
    initial_orientation: ArrayLike
    angular_velocity: ArrayLike
    initial_time: float = 0.0
    
    xp: Backend = np
    dtype: DTypeLike = np.float32

    def __post_init__(self) -> None:
        _coerce_vec3(self, "initial_position", self.xp, self.dtype)
        _coerce_vec3(self, "initial_velocity", self.xp, self.dtype)
        _coerce_quat(self, "initial_orientation", self.xp, self.dtype)
        _coerce_vec3(self, "angular_velocity", self.xp, self.dtype)


@dataclass(frozen=True, slots=True)
class ConstantAccelerationMotion(Motion):
    initial_position: ArrayLike
    initial_velocity: ArrayLike
    acceleration: ArrayLike
    initial_orientation: ArrayLike
    angular_velocity: ArrayLike
    initial_time: float = 0.0
    
    xp: Backend = np
    dtype: DTypeLike = np.float32

    def __post_init__(self) -> None:
        _coerce_vec3(self, "initial_position", self.xp, self.dtype)
        _coerce_vec3(self, "initial_velocity", self.xp, self.dtype)
        _coerce_vec3(self, "acceleration", self.xp, self.dtype)
        _coerce_quat(self, "initial_orientation", self.xp, self.dtype)
        _coerce_vec3(self, "angular_velocity", self.xp, self.dtype)

    
def _coerce_array(obj: object, name: str, expected_shape: tuple[int, ...], xp: Backend = np, dtype: DTypeLike = np.float32) -> None:
    value = xp.asarray(getattr(obj, name), dtype=dtype)
    if value.shape != expected_shape:
        raise ValueError(
            f"{type(obj).__name__}.{name} must have shape {expected_shape};"
            f"got {value.shape}."
        )
    object.__setattr__(obj, name, value)

def _coerce_vec3(obj: object, name: str, xp: Backend = np, dtype: DTypeLike = np.float32) -> None:
    _coerce_array(obj, name, (3,), xp, dtype)

def _coerce_quat(obj: object, name: str, xp: Backend = np, dtype: DTypeLike = np.float32) -> None:
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


def build_batch(motion_cls: type[Motion], entities: Sequence[Entity], dtype: DTypeLike, xp: Backend = np) -> MotionBatch:
    return batch_class_for(motion_cls).from_entities(entities, dtype)


@dataclass(frozen=True, slots=True, kw_only=True)
class MotionBatch(ABC):
    xp: Backend = np
    dtype: DTypeLike = np.float32
    
    @classmethod
    @abstractmethod
    def from_entities(cls, entities: Sequence[Entity], dtype: DTypeLike, xp: Backend = np) -> MotionBatch:
        ...
    
    @abstractmethod
    def evaluate(self, time: ArrayLike) -> MotionBlock:
        ...
    

@register_motion_batch(StaticMotion)
@dataclass(frozen=True, slots=True)
class StaticBatch(MotionBatch):
    positions: ArrayLike
    velocities: ArrayLike
    orientations: ArrayLike
    angular_rates: ArrayLike
    
    @classmethod
    def from_entities(cls, entities: Sequence[Entity], dtype: DTypeLike, xp: Backend = np) -> StaticBatch:
        motions = [cast(StaticMotion, e.motion) for e in entities]
        return cls(
            positions=xp.stack([m.position for m in motions]).astype(dtype),
            velocities=xp.stack([m.velocity for m in motions]).astype(dtype),
            orientations=xp.stack([m.orientation for m in motions]).astype(dtype),
            angular_rates=xp.stack([m.angular_velocity for m in motions]).astype(dtype),        
        )
    
    def evaluate(self, time: Any) -> MotionBlock:
        xp = self.xp
        
        del time
        return MotionBlock(
            positions=self.positions,
            velocities=self.velocities,
            accelerations=xp.zeros_like(self.positions),
            orientations=self.orientations,
            angular_rates=self.angular_rates,            
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
    def from_entities(cls, entities: Sequence[Entity], dtype: DTypeLike, xp: Backend = np) -> ConstantVelocityBatch:
        motions = [cast(ConstantVelocityMotion, e.motion) for e in entities]
        return cls(
            initial_positions=xp.stack([m.initial_position for m in motions]).astype(dtype),
            initial_velocities=xp.stack([m.initial_velocity for m in motions]).astype(dtype),
            initial_orientations=xp.stack([m.initial_orientation for m in motions]).astype(dtype),
            angular_rates=xp.stack([m.angular_velocity for m in motions]).astype(dtype),
            initial_times=xp.array([m.initial_time for m in motions], dtype=dtype),            
        )
        
    def evaluate(self, time: Any) -> MotionBlock:
        xp = self.xp
        dtype = self.dtype
        
        dt = time - self.initial_times
        dt_col = dt[:, None]
        positions = self.initial_positions + self.initial_velocities * dt_col
        dq = axis_angle_delta_quat(self.angular_rates, dt, xp=xp)
        orientations = quat_normalise(quat_multiply(self.initial_orientations, dq, xp=xp, dtype=dtype), xp=xp)
        
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
    def from_entities(cls, entities: Sequence[Entity], dtype: DTypeLike, xp: Backend = np) -> MotionBatch:
        motions = [cast(ConstantAccelerationMotion, e.motion) for e in entities]
        return cls(
            initial_positions=xp.stack([m.initial_position for m in motions]).astype(dtype),
            initial_velocities=xp.stack([m.initial_velocity for m in motions]).astype(dtype),
            accelerations=xp.stack([m.acceleration for m in motions]).astype(dtype),
            initial_orientations=xp.stack([m.initial_orientation for m in motions]).astype(dtype),
            angular_rates=xp.stack([m.angular_velocity for m in motions]).astype(dtype),
            initial_times=xp.array([m.initial_time for m in motions], dtype=dtype),            
        )
    
    def evaluate(self, time: Any) -> MotionBlock:
        xp = self.xp
        dtype = self.dtype
        
        dt = time - self.initial_times
        dt_col = dt[:, None]
        positions = self.initial_positions + self.initial_velocities * dt_col + 0.5 * self.accelerations * dt_col ** 2
        velocities = self.initial_velocities + self.accelerations * dt_col
        dq = axis_angle_delta_quat(self.angular_rates, dt, xp=xp)
        orientations = quat_normalise(quat_multiply(self.initial_orientations, dq, xp=xp, dtype=dtype), xp=xp)
        
        return MotionBlock(
            positions=positions,
            velocities=velocities,
            accelerations=self.accelerations,
            orientations=orientations,
            angular_rates=self.angular_rates,            
        )
    
_ensure_all_pytrees_registered()
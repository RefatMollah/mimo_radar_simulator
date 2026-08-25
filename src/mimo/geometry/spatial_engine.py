""" """

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from numpy.typing import NDArray
from dataclasses import dataclass, replace, fields

from typing import TYPE_CHECKING, Mapping, Sequence, Any, cast, TypeAlias
from numpy.typing import DTypeLike, NDArray

from .jax_backend import _maybe_register_pytree
from .motion_model import (
    MotionBatch,
    MotionBlock,
)

if TYPE_CHECKING:
    from ..scene.scene import CompiledScene, RadarEngagements, CompiledChannels
    from ..scene.snapshot_builder import SceneSnapshot


ArrayLike: TypeAlias = Any
Backend: TypeAlias = Any


_FIELD_WIDTHS: tuple[tuple[str, int], ...] = (
    ("positions",     3),
    ("velocities",    3),
    ("accelerations", 3),
    ("orientations",  4),
    ("angular_rates", 3),
)

@dataclass(frozen=True, slots=True)
class State:
    """Full-scene numeric result."""

    time: float | None
    positions: ArrayLike
    velocities: ArrayLike
    accelerations: ArrayLike
    orientations: ArrayLike
    angular_rates: ArrayLike
    
        
_maybe_register_pytree(State)


@dataclass
class DenseCompiledScene:
    n: int
    dtype: np.dtype
    dense_batches: Mapping[str, MotionBatch]
    masks: Mapping[str, NDArray[np.bool_]]


def _scatter_rows(
    base: ArrayLike, 
    indices: ArrayLike, 
    values: ArrayLike,
    *, 
    xp: Backend,
) -> ArrayLike:
    
    if indices.shape[0] == 0:
        return base
    
    if xp is np:
        result = base.copy()
        result[indices] = values
        return result
    
    return base.at[indices].set(values)


def _empty_state_arrays(n: int, dtype: Any, xp: Backend) -> dict[str, ArrayLike]:
    return{
        name: xp.zeros((n, width), dtype=dtype)
        for name, width in _FIELD_WIDTHS
    }


def _assemble_sparse(
    n: int,
    dtype: Any,
    blocks: Sequence[tuple[ArrayLike, MotionBlock]],
    xp: Backend
) -> State:
    """Evaluate compact batches, then scatter each field into scene slots."""
    out = _empty_state_arrays(n, dtype, xp)
    if not blocks:
        return State(time=None, **out)
    
    indices = xp.concatenate([idx for idx, _ in blocks])
    
    for name, _ in _FIELD_WIDTHS:
        values = xp.concatenate([getattr(block, name) for _, block in blocks])
        out[name] = _scatter_rows(out[name], indices, values, xp=xp)
    
    return State(time=None, **out)


def _state_at_impl(scene: CompiledScene, time: ArrayLike) -> State:
    xp = scene.xp
    evaluated: list[tuple[ArrayLike, MotionBlock]] = []
    
    for motion_name, batch in scene.motion_batches.items():
        indices = scene.slots_by_motion[motion_name]
        if indices.size == 0:
            continue
        block = batch.evaluate(time)
        evaluated.append((indices, block))
        
    state = _assemble_sparse(scene.n, scene.dtype, evaluated, xp=xp)
    
    return replace(state, time=time)


_jitted_state_at = jax.jit(_state_at_impl)


def state_at(scene: CompiledScene, time: ArrayLike) -> State:

    if scene.backend == "jax":
        time_arr = jnp.asarray(time, dtype=scene.dtype)
        return _jitted_state_at(scene, time_arr)

    return _state_at_impl(scene, time)
    

def _densify_batch(batch: MotionBatch, indices: ArrayLike, n: int, xp: Backend) -> MotionBatch:
    kwargs: dict[str, ArrayLike] = {}
    
    skip_fields = {"xp", "dtype"}
    
    for field in fields(batch):
        name = field.name
        
        if name in skip_fields:
            continue

        compact = getattr(batch, name)
        
        full = xp.zeros((n,) + compact.shape[1:], dtype=compact.dtype)
        kwargs[name] = _scatter_rows(full, indices, compact, xp=xp)
    
    kwargs["xp"] = batch.xp
    kwargs["dtype"] = batch.dtype
    
    return type(batch)(**kwargs)


def densify(scene: CompiledScene, *, xp: Backend = np) -> DenseCompiledScene:
    dense_batches: dict[str, MotionBatch] = {}
    masks: dict[str, ArrayLike] = {}
    
    for motion_name, batch in scene.motion_batches.items():
        indices = scene.slots_by_motion[motion_name]
        dense_batches[motion_name] = _densify_batch(batch, indices, scene.n, xp)
        
        mask = xp.zeros((scene.n, 1), dtype=bool)
        if indices.size:
            mask = _scatter_rows(
                mask,
                indices,
                xp.ones((indices.size, 1), dtype=bool),
                xp=xp
            )
        masks[motion_name] = mask

    return DenseCompiledScene(
        n=scene.n,
        dtype=scene.dtype,
        dense_batches=dense_batches,
        masks=masks,
    )
    

def state_at_dense(scene: DenseCompiledScene, time: ArrayLike, *, xp: Backend = np) -> State:
    out = _empty_state_arrays(scene.n, scene.dtype, xp)
    
    for kind, batch in scene.dense_batches.items():
        block = batch.evaluate(time)
        mask = scene.masks[kind]
        
        for name, _ in _FIELD_WIDTHS:
            out[name] = xp.where(mask, getattr(block, name), out[name])
        
    return State(time=time, **out)
     
 
def check_causality(
    batches: Mapping[str, MotionBatch],
    time: ArrayLike,
) -> None:
    """Validate reference-time constraints on the Python/NumPy side."""
    requested = np.asarray(time)

    for motion_name, batch in batches.items():
        raw_initial_times = getattr(batch, "initial_times", None)
        if raw_initial_times is None:
            continue

        initial_times = np.asarray(raw_initial_times)
        if np.any(requested < initial_times):
            raise ValueError(
                f"Cannot evaluate motion class {motion_name!r} before its "
                f"reference time (requested t={time!r})."
            )  
        

QUATERNION_CONJUGATE = np.array([1.0, -1.0, -1.0, -1.0])
        
@dataclass(frozen=True, slots=True)
class BistaticGeometry:
    """
    Container for coputed bistatic radar geometry.
    Coordinate frame: east-north-up (ENU)
    """
    tx_los:         ArrayLike
    rx_los:         ArrayLike
    
    tx_range:       ArrayLike
    rx_range:       ArrayLike
    
    bistatic_range_rate: ArrayLike
    
    tx_azimuth:     ArrayLike
    tx_elevation:   ArrayLike
    rx_azimuth:     ArrayLike
    rx_elevation:   ArrayLike

@dataclass(frozen=True, slots=True)
class SensorWorldPoses:
    """One row per link (mirrors SensorOffsets)"""
    positions:    ArrayLike
    orientations: ArrayLike
    

def bistatic_geometry(sc: SceneSnapshot, engagements: RadarEngagements):
    """Calculates the bistatic geometry for a given set of engagements."""
    tx_idx, tgt_idx, rx_idx = engagements.indices.tx_slots, engagements.indices.target_slots, engagements.indices.rx_slots        
    
    tx_pos, tgt_pos, rx_pos = sc.positions[tx_idx], sc.positions[tgt_idx], sc.positions[rx_idx]
    tx_vel, tgt_vel, rx_vel = sc.velocities[tx_idx], sc.velocities[tgt_idx], sc.velocities[rx_idx]
    tx_ori, tgt_ori, rx_ori = sc.orientations[tx_idx], sc.orientations[tgt_idx], sc.orientations[rx_idx]
    
    tx_range_vec = tgt_pos - tx_pos
    rx_range_vec = tgt_pos - rx_pos

    tx_range_mag = np.linalg.norm(tx_range_vec, axis=-1, keepdims=True)
    rx_range_mag = np.linalg.norm(rx_range_vec, axis=-1, keepdims=True)
    
    tx_los = np.divide(
        tx_range_vec, tx_range_mag,
        out=np.zeros_like(tx_range_vec),
        where=tx_range_mag > 0
    )
    rx_los = np.divide(
        rx_range_vec, rx_range_mag,
        out=np.zeros_like(rx_range_vec),
        where=rx_range_mag > 0
    )
    
    bistatic_vector = tx_los + rx_los
    
    range_rate = (
        np.einsum("...i,...i->...", bistatic_vector, tgt_vel) -
        np.einsum("...i,...i->...", tx_los, tx_vel) -
        np.einsum("...i,...i->...", rx_los, rx_vel)
    )
    
    tx_los_rot = rotate_into_sensor_frame(tx_ori, tx_los)
    rx_los_rot = rotate_into_sensor_frame(rx_ori, rx_los)
    
    tx_azimuth = np.atan2(tx_los_rot[..., 1], tx_los_rot[..., 0])
    rx_azimuth = np.atan2(rx_los_rot[..., 1], rx_los_rot[..., 0])
    
    tx_rho = np.hypot(tx_los_rot[..., 0], tx_los_rot[..., 1])
    rx_rho = np.hypot(rx_los_rot[..., 0], rx_los_rot[..., 1])
    
    tx_elevation = np.atan2(tx_los_rot[..., 2], tx_rho)
    rx_elevation = np.atan2(tx_los_rot[..., 2], rx_rho)
    
    return BistaticGeometry(
        tx_los=tx_los,
        rx_los=rx_los,
        tx_range=tx_range_mag,
        rx_range=rx_range_mag,
        bistatic_range_rate=range_rate,
        tx_azimuth=tx_azimuth,
        tx_elevation=tx_elevation,
        rx_azimuth=rx_azimuth,
        rx_elevation=rx_elevation
    )

#FIXME: fix backend inconsistencies
def compile_sensor_world_poses(sc: SceneSnapshot, channel: CompiledChannels) -> SensorWorldPoses:
    """Apply each link's mounting offset to its parent entity's current pose."""    
    link_slots = channel.link_slots
    pos_offsets = channel.sensor_offsets.pos_offsets
    rot_offsets = channel.sensor_offsets.rot_offsets
    
    entity_pos = sc.positions[link_slots]     # (N, 2, 3)
    entity_ori = sc.orientations[link_slots]  # (N, 2, 4)
    
    sensor_pos = entity_pos + rotate_sensor_to_world(entity_ori, pos_offsets)
    sensor_ori = _quat_multiply(entity_ori, rot_offsets)
    
    return SensorWorldPoses(sensor_pos, sensor_ori)
    
def rotate_into_sensor_frame(orientations: ArrayLike, vectors: ArrayLike) -> ArrayLike:
    """
    Rotate a batch of vectors with a batch of quaternions.
    Convention: [w, x, y, z]
    q  -> body-to-world
    q' -> world-to-body
    """
    q = _normalise(orientations)
    q_conj = q * QUATERNION_CONJUGATE
    
    zeros = np.zeros((*vectors.shape[:-1], 1))
    v_quat = np.concatenate((zeros, vectors), axis=-1, dtype=np.float32)
    
    # Passive rotation: q'vq
    rotated = _quat_multiply(q_conj, _quat_multiply(v_quat, q)) 
    
    return rotated[..., 1:]

def rotate_sensor_to_world(orientations: ArrayLike, vectors: ArrayLike,*, dtype: DTypeLike=np.float32) -> ArrayLike:
    q = _normalise(orientations)
    q_conj = q * QUATERNION_CONJUGATE 
    
    zeros = np.zeros((*vectors.shape[:-1], 1))
    v_quat = np.concatenate((zeros, vectors), axis=1, dtype=dtype)
    
    
    rotated = _quat_multiply(q, _quat_multiply(v_quat, q_conj))
    
    return rotated[..., 1:]

def _quat_multiply(q1: ArrayLike, q2: ArrayLike) -> ArrayLike:
    """Hamilton porduct of two arrays of quaternions."""
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)
    
    return np.concatenate((
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 - y2*z1 + z2*y1,
        w2*y1 + x2*z1 + y2*w1 - z2*x1,
        w2*z1 - x2*y1 + y2*x1 + z2*w1
    ), axis=-1, dtype=np.float32)
    
def _normalise(v: ArrayLike) -> ArrayLike:
    mag = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, mag, out=np.zeros_like(v), where= mag>0)
    
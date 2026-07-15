""" """

import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from dataclasses import dataclass

from ..scene.scene import Scene, RadarNetwork, SensorOffsets, EngagementIndices, RadarEngagements, CompiledChannels
from ..scene.scene_snapshot import SceneSnapshot
from ..entity.radar_component import TargetProperties, RadarNode


QUATERNION_CONJUGATE = np.array([1.0, -1.0, -1.0, -1.0])

@dataclass(frozen=True, slots=True)
class BistaticGeometry:
    """
    Container for coputed bistatic radar geometry.
    Coordinate frame: east-north-up (ENU)
    """
    tx_los:         NDArray[np.float64]
    rx_los:         NDArray[np.float64]
    
    tx_range:       NDArray[np.float64]
    rx_range:       NDArray[np.float64]
    
    bistatic_range_rate: NDArray[np.float64]
    
    tx_azimuth:     NDArray[np.float64]
    tx_elevation:   NDArray[np.float64]
    rx_azimuth:     NDArray[np.float64]
    rx_elevation:   NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class SensorWorldPoses:
    """One row per link (mirrors SensorOffsets)"""
    positions:    NDArray[np.float64]
    orientations: NDArray[np.float64]
    

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
    
def rotate_into_sensor_frame(orientations: NDArray[np.float64], vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Rotate a batch of vectors with a batch of quaternions.
    Convention: [w, x, y, z]
    q  -> body-to-world
    q' -> world-to-body
    """
    q = _normalise(orientations)
    q_conj = q * QUATERNION_CONJUGATE
    
    zeros = np.zeros((*vectors.shape[:-1], 1))
    v_quat = np.concatenate((zeros, vectors), axis=-1)
    
    # Passive rotation: q'vq
    rotated = _quat_multiply(q_conj, _quat_multiply(v_quat, q)) 
    
    return rotated[..., 1:]

def rotate_sensor_to_world(orientations: NDArray[np.float64], vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    q = _normalise(orientations)
    q_conj = q * QUATERNION_CONJUGATE 
    
    zeros = np.zeros((*vectors.shape[:-1], 1))
    v_quat = np.concatenate((zeros, vectors), axis=1)
    
    
    rotated = _quat_multiply(q, _quat_multiply(v_quat, q_conj))
    
    return rotated[..., 1:]

def _quat_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton porduct of two arrays of quaternions."""
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)
    
    return np.concatenate((
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 - y2*z1 + z2*y1,
        w2*y1 + x2*z1 + y2*w1 - z2*x1,
        w2*z1 - x2*y1 + y2*x1 + z2*w1
    ), axis=-1)
    
def _normalise(v: NDArray[np.float64]) -> NDArray[np.float64]:
    mag = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, mag, out=np.zeros_like(v), where= mag>0)
    
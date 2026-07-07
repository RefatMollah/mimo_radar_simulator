import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from dataclasses import dataclass

from ..scene.scene import Scene
from ..scene.scene_snapshot import SceneSnapshot
from ..entity.radar_component import RadarComponent, TargetComponent, TransmitterComponent, ReceiverComponent


@dataclass(frozen=True, slots=True)
class EngagementIndices:
    tx_idx:   NDArray[np.int_]
    tgt_idx:  NDArray[np.int_]
    rx_idx:   NDArray[np.int_]
    
    
class EngagementManager:
    
    def __init__(self, scene: Scene):
        self._scene = scene
        
        self._engagements = EngagementIndices(
            np.empty(0, dtype=np.int_),
            np.empty(0, dtype=np.int_),
            np.empty(0, dtype=np.int_)
        )
        self.update()
    
    @property
    def engagements(self) -> EngagementIndices:
        return self._engagements
        
    def update(self) -> None:
        tx_idx = []
        tgt_idx = []
        rx_idx = []
        
        for index, entity in enumerate(self._scene.iter_all()):
            
            if entity.has_component(TransmitterComponent):
                tx_idx.append(index)
            if entity.has_component(TargetComponent):
                tgt_idx.append(index)
            if entity.has_component(ReceiverComponent):
                rx_idx.append(index)
        
        self._engagements = self._build_engagements(
            np.asarray(tx_idx, dtype=np.int_),
            np.asarray(tgt_idx, dtype=np.int_),
            np.asarray(rx_idx, dtype=np.int_)
        )
      
    @staticmethod
    def _build_engagements(
        tx_idx:   NDArray[np.int_],
        tgt_idx:  NDArray[np.int_],
        rx_idx:   NDArray[np.int_],
    ) -> EngagementIndices:
        
        if tx_idx.size==0 or tgt_idx.size==0 or rx_idx.size==0:
            empty = np.empty(0, dtype=np.int_)
            return EngagementIndices(empty, empty, empty)
        
        Tx, Tgt, Rx = np.meshgrid(tx_idx, tgt_idx, rx_idx, indexing="ij")
        
        tx_flat = Tx.ravel()
        tgt_flat = Tgt.ravel()
        rx_flat = Rx.ravel() 

        # An entity may be both a transmitter and receiver (monostatic radar). 
        # The reflection target cannot be the receiving or transmitting entity.
        mask = (tx_flat != tgt_flat) & (rx_flat != tgt_flat)
        
        return EngagementIndices(
            tx_flat[mask],
            tgt_flat[mask],
            rx_flat[mask],
        )

###############################################################################################################
# Computing Geometry
###############################################################################################################      

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
    

def compute_geometry(sc: SceneSnapshot, engagements: EngagementIndices):
    """Calculates the bistatic geometry for a given set of engagements."""
    tx_idx, tgt_idx, rx_idx = engagements.tx_idx, engagements.tgt_idx, engagements.rx_idx
    
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
    
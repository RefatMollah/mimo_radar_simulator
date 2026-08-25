from __future__ import annotations

import numpy as np
from typing import Any, TypeAlias, Mapping
from numpy.typing import NDArray
from dataclasses import dataclass
from collections import defaultdict 

from .quat import identity_quats, quat_multiply, quat_normalise, quat_rotate
from jax_backend import steering_batch_class_for
from .sensor_steering import (
    SteeringLaw, 
    SteeringActuator, 
    SteeringActuatorBatch, 
    SteeringLawBatch,
    SteeringState,
)
from .spatial_engine import State
from ..scene.scene import Scene, BackendContext


ArrayLike: TypeAlias = Any

@dataclass(frozen=True, slots=True)
class ComponentMountOffsets:
    """Per-component mounting offsets."""
    pos_offsets: NDArray[np.float32]   # (n_components, 3)
    rot_offsets: NDArray[np.float32]   # (n_components, 4)

@dataclass(frozen=True, slots=True)
class CompiledSensorRig:
    version: int
    component_ids: tuple[str, ...]
    platform_slots: NDArray[np.intp]
    slots_by_steering: Mapping[str, NDArray[np.intp]]
    steering_batches: Mapping[str, SteeringLawBatch]
    actuator_batches: Mapping[str, SteeringActuatorBatch]
    mount_offsets: ComponentMountOffsets
    dtype: np.dtype[Any]
    backend: str
    
    @property
    def n(self) -> int:
        return len(self.component_ids)
    
    @property
    def xp(self):
        if self.backend == "jax":
            import jax.numpy as jnp
            return jnp
        else:
            return np

_SENSOR_RIG_PYTREE_REGISTERED = False
def _register_sensor_rig_pytree() -> None:
    global _SENSOR_RIG_PYTREE_REGISTERED
    if _SENSOR_RIG_PYTREE_REGISTERED:
        return
    import jax
    jax.tree_util.register_dataclass(
        CompiledSensorRig,
        data_fields=(
            "platform_slots", 
            "slots_by_steering", 
            "steering_batches", 
            "actuator_batches", 
            "mount_offsets"
        ),
        meta_fields=("version", "component_ids", "dtype", "backend"),
    )
    _SENSOR_RIG_PYTREE_REGISTERED = True


class SensorRig:
    """Manages sensor components, their mount offsets, and steering laws/actuators."""
    def __init__(self, *, backend: BackendContext = BackendContext.numpy()) -> None:
        self._components: dict[int, str] = {}
        self._slots_by_id: dict[str, int] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._topology_version = 0
        
        self._steering: dict[int, SteeringLaw | SteeringActuator] = {}
        self._platform_ids: dict[int, str] = {}
        self._mount_offsets: dict[int, tuple[ArrayLike, ArrayLike]] = {}
        
        self._backend = backend

    @property
    def topology_version(self) -> int:
        return self._topology_version

    def add_component(
        self, 
        component_id: str, 
        platform_id: str, 
        mount_pos: ArrayLike, 
        mount_rot: ArrayLike, 
        steering: SteeringLaw | SteeringActuator
    ) -> int:
        if component_id in self._slots_by_id:
            raise ValueError(f"Component {component_id!r} already registered.")
            
        slot = self._free_slots.pop() if self._free_slots else self._next_slot
        if slot == self._next_slot:
            self._next_slot += 1
            
        self._slots_by_id[component_id] = slot
        self._components[slot] = component_id
        self._steering[slot] = steering
        self._platform_ids[slot] = platform_id
        self._mount_offsets[slot] = (mount_pos, mount_rot)
        self._topology_version += 1
        return slot

    def compile(self, scene: Scene) -> CompiledSensorRig:
        if self._backend.name == "jax":
            _register_sensor_rig_pytree()
            
        xp = self._backend.xp
        dtype = self._backend.dtype
        n = self._next_slot
        
        component_ids = [""] * n
        platform_slots = np.zeros(n, dtype=np.intp)
        
        buckets_law: dict[type[SteeringLaw], list[int]] = defaultdict(list)
        buckets_act: dict[type[SteeringActuator], list[int]] = defaultdict(list)
        
        pos_offsets = []
        rot_offsets = []
        
        for slot in range(n):
            comp_id = self._components.get(slot, "")
            component_ids[slot] = comp_id
            
            if slot in self._platform_ids:
                plat_id = self._platform_ids[slot]
                platform_slots[slot] = scene._slots_by_id.get(plat_id, -1)
                
            if slot in self._steering:
                steer = self._steering[slot]
                if isinstance(steer, SteeringLaw):
                    buckets_law[type(steer)].append(slot)
                elif isinstance(steer, SteeringActuator):
                    buckets_act[type(steer)].append(slot)
                    
            if slot in self._mount_offsets:
                p, r = self._mount_offsets[slot]
                pos_offsets.append(p)
                rot_offsets.append(r)
            else:
                pos_offsets.append(np.zeros(3))
                rot_offsets.append(np.array([1.0, 0.0, 0.0, 0.0]))

        rig_offsets = ComponentMountOffsets(
            pos_offsets=xp.stack(pos_offsets).astype(dtype),
            rot_offsets=xp.stack(rot_offsets).astype(dtype)
        )

        slots_by_steering = {}
        steering_batches = {}
        for law_cls, slots in buckets_law.items():
            slots_by_steering[law_cls.__name__] = np.array(slots, dtype=np.intp)
            # Create dummy component objects for the batch builder
            comps = [type('Comp', (), {'steering': self._steering[s]})() for s in slots]
            steering_batches[law_cls.__name__] = steering_batch_class_for(law_cls).from_components(comps, dtype, xp)

        actuator_batches = {}
        for act_cls, slots in buckets_act.items():
            # For actuators, we might just initialize the batch with default/max params
            # depending on how you want to handle heterogeneous actuators.
            # Here we assume homogeneous batches for simplicity.
            pass 

        return CompiledSensorRig(
            version=self._topology_version,
            component_ids=tuple(component_ids),
            platform_slots=platform_slots,
            slots_by_steering=slots_by_steering,
            steering_batches=steering_batches,
            actuator_batches=actuator_batches,
            mount_offsets=rig_offsets,
            dtype=np.dtype(dtype),
            backend=self._backend.name,
        )

def resolve_sensor_frames(
    platform_state: State,
    compiled_rig: CompiledSensorRig,
    steering_orientations: ArrayLike | None = None,
    actuator_states: SteeringState | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    """Resolves world-frame positions and orientations for all sensor components."""
    xp = compiled_rig.xp
    dtype = compiled_rig.dtype
    
    plat_pos = platform_state.positions[compiled_rig.platform_slots]
    plat_ori = platform_state.orientations[compiled_rig.platform_slots]
    
    mount_pos = compiled_rig.mount_offsets.pos_offsets
    mount_rot = compiled_rig.mount_offsets.rot_offsets
    
    if steering_orientations is not None:
        steer_ori = steering_orientations
    elif actuator_states is not None:
        # Convert az/el to quaternion (placeholder logic)
        # In reality, use a proper euler-to-quat function
        az = actuator_states.azimuth
        el = actuator_states.elevation
        steer_ori = identity_quats(compiled_rig.n, xp, dtype) # Replace with actual az/el conversion
    else:
        steer_ori = identity_quats(compiled_rig.n, xp, dtype)
        
    # Compose: World = Platform * Mount * Steering
    rot_mount_pos = quat_rotate(plat_ori, mount_pos, xp=xp)
    positions = plat_pos + rot_mount_pos
    
    ori1 = quat_multiply(plat_ori, mount_rot, xp=xp, dtype=dtype)
    orientations = quat_multiply(ori1, steer_ori, xp=xp, dtype=dtype)
    orientations = quat_normalise(orientations, xp=xp)
    
    return positions, orientations
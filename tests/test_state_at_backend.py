import pytest
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass

from mimo.entity.entity import Entity
from mimo.scene.scene import Scene, BackendContext
from mimo.geometry.spatial_engine import state_at
from mimo.geometry.motion_model import (
    StaticMotion,
    ConstantVelocityMotion,
    ConstantAccelerationMotion,
)

def static_motion():
    return StaticMotion(
        position=np.zeros(3),
        orientation=np.array([1, 0, 0, 0]),
    )
    
def constant_velocity_motion():
    return ConstantVelocityMotion(
        initial_position=np.zeros(3),
        initial_velocity=np.ones(3),
        initial_orientation=np.array([1, 0, 0, 0]),
        angular_velocity=np.zeros(3),
    )
    
def constant_acceleration_motion():
    return ConstantAccelerationMotion(
        initial_position=np.zeros(3),
        initial_velocity=np.ones(3),
        acceleration=np.ones(3),
        initial_orientation=np.array([1, 0, 0, 0]),
        angular_velocity=np.zeros(3),
    )


@pytest.fixture
def create_entity():
    def _create_entity(motion):
        return Entity(motion=motion)
    
    return _create_entity


class TestStateAtBackendAndJIT:
    def test_dispatch_and_jit(self, create_entity):
        # Test NumPy Dipatch
        scene = Scene()
        e1 = create_entity(static_motion())
        scene.add_entity(e1)
        compiled_np = scene.compile()
        state_np = state_at(compiled_np, 1.0)
        assert isinstance(state_np.positions, np.ndarray)
        
        # Test JAX Dispatch
        scene = Scene(backend=BackendContext.jax())
        scene.add_entity(e1)
        compiled_jax = scene.compile()
        state_jax = state_at(compiled_jax, 1.0)
        assert isinstance(state_jax.positions, jax.Array)
        
    
class TestNumericalCorrectness:
    def test_slot_alignment_kinematics(self, create_entity):
        scene = Scene()
        
        e1 = create_entity(
            StaticMotion(
                position=np.array([10.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        scene.add_entity(e1)
        
        e_temp = create_entity(static_motion())
        scene.add_entity(e_temp)
        scene.remove_entity(e_temp.id)
        
        e2 = create_entity(
            ConstantVelocityMotion(
                initial_position=np.array([0.0, 5.0, 0.0]),
                initial_velocity=np.array([2.0, 0.0, 0.0]),
                initial_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.zeros(3),
            )
        )
        scene.add_entity(e2)
        
        compiled = scene.compile()
        state = state_at(compiled, time=2.0)
        
        # Verify Slot 0 (Static): Should remain at initial position
        np.testing.assert_allclose(state.positions[0], [10.0, 0.0, 0.0])
        
        # Verify Slot 1 (Consant Velocity): Should move to position [4, 5, 0]
        np.testing.assert_allclose(state.positions[1], [4.0, 5.0, 0.0]) 


class TestPytreeRegistration:
    def test_registration(self, create_entity):
        scene = Scene(backend=BackendContext.jax())
        e1 = create_entity(
            StaticMotion(
                position=np.array([10.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )        
        scene.add_entity(e1)
        
        compiled = scene.compile()
        
        leaves = jax.tree_util.tree_leaves(compiled)
        assert len(leaves) > 0, "CompiledScene should have traceable leaves."
        
        jitted_state_at = jax.jit(state_at)
        state = jitted_state_at(compiled, 1.5)
        
        assert isinstance(state.positions, jax.Array)
        np.testing.assert_allclose(np.asarray(state.positions[0]), [10.0, 0.0, 0.0])
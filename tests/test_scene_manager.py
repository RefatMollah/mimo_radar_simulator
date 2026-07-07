import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock

from src.mimo.scene.scene_manager import SceneManager, _EntityBatch, _collect_entities
from src.mimo.scene.scene import Scene
from src.mimo.geometry.motion_model import MotionModel
from src.mimo.entity.entity import Entity

#############################################################################
# Fixtures
#############################################################################

@pytest.fixture
def make_state():
    
    def _make_state(i=0):
        return SimpleNamespace(
            position=np.array([i, i+1, i+2], dtype=np.float64),
            velocity=np.array([10+i, 11+i, 12+i], dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            angular_velocity=np.array([20+i, 21+i, 22+i], dtype=np.float64),
        )
    return _make_state

@pytest.fixture
def make_entity(make_state):
    
    def _make_entity(entity_id=None, state=None, raises=None):
        entity = MagicMock(spec=Entity)
        entity.id = entity_id or f"entity-{id(entity)}"
        if raises is not None:
            entity.get_state.side_effect = raises
        else:
            entity.get_state.return_value = state if state is not None else make_state()
        return entity
    
    return _make_entity

@pytest.fixture
def scene():
    s = MagicMock(spec=Scene)
    s.iter_static.return_value = []
    s.iter_dynamic.return_value = []
    return s

class TestInitStaticCaching:
    
    def test_no_static_entities_gives_empty_arrays(self, scene):
        scene.iter_static.return_value = []
        
        manager = SceneManager(scene, start_time=0.0)
        
        assert manager._static_batch.n == 0
        assert manager._static_batch.ids == ()
        assert manager._static_batch.positions.shape == (0, 3)
        assert manager._static_batch.velocities.shape == (0, 3)
        assert manager._static_batch.orientations.shape == (0, 4)
        assert manager._static_batch.angular_velocities.shape == (0, 3)
    
    def test_single_static_entity_cached(self, scene, make_entity, make_state):
        state = make_state(i=5)
        e1 = make_entity("e1", state=state)
        scene.iter_static.return_value = [e1]
        
        manager = SceneManager(scene, start_time=1.0)
        
        assert manager._static_batch.n == 1
        assert manager._static_batch.ids == ("e1",)
        assert manager._static_batch.positions.shape == (1,3)
        np.testing.assert_array_equal(manager._static_batch.positions[0], state.position)
        np.testing.assert_array_equal(manager._static_batch.velocities[0], state.velocity)
        np.testing.assert_array_equal(manager._static_batch.orientations[0], state.orientation)
        np.testing.assert_array_equal(manager._static_batch.angular_velocities[0], state.angular_velocity)
    
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from src.mimo.geometry.motion_model import (
    Motion,
    StaticMotion,
    ConstantVelocityMotion,
    ConstantAccelerationMotion,
)
from src.mimo.entity.entity import Entity
from src.mimo.scene.scene import Scene, DuplicateEntityError, EntityNotFoundError

#####################################################################################
# Fixtures
#####################################################################################


def static_motion():
    return StaticMotion(
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=np.array([1, 0, 0, 0]),
        angular_velocity=np.zeros(3),
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

#####################################################################################
# Add and get tests
#####################################################################################

def test_add_and_get_static_entity(create_entity):
    e1 = create_entity(static_motion())
    sc = Scene()
    sc.add_entity(e1)
    
    assert sc.get_entity(e1.id) == e1

def test_add_and_get_dynamic_entity(create_entity):
    e1 = create_entity(constant_velocity_motion)
    sc = Scene()
    sc.add_entity(e1)
    
    assert sc.get_entity(e1.id) == e1

def test_add_duplicate_entity_raises_error(create_entity):
    e1 = create_entity(static_motion())
    sc = Scene()
    sc.add_entity(e1)
    
    with pytest.raises(DuplicateEntityError):
        sc.add_entity(e1)

def test_missing_entity_raises_error():
    sc = Scene()
    
    with pytest.raises(EntityNotFoundError):
        sc.get_entity("random_id")

#####################################################################################
# Remove Tests
#####################################################################################        

def test_remove_entity(create_entity):
    e1 = create_entity(static_motion())
    sc = Scene()
    sc.add_entity(e1)
    sc.remove_entity(e1.id)
    
    with pytest.raises(EntityNotFoundError):
        sc.get_entity(e1)

def test_remove_missing_entity_raises_error(create_entity):
    e1 = create_entity(static_motion())
    sc = Scene()
    sc.add_entity(e1)
    
    with pytest.raises(EntityNotFoundError):
        sc.remove_entity("Non_existent_id")        

########################################################################################
# Iterator Tests
########################################################################################

def test_iter_static_entities(create_entity):
    
    e1 = create_entity(static_motion())
    e2 = create_entity(static_motion())
    e3 = create_entity(constant_velocity_motion())
        
    sc = Scene()
    sc.add_entity(e1)
    sc.add_entity(e2)
    sc.add_entity(e3)
    
    actual_entities = list(sc.iter_static())
    
    assert len(actual_entities) == 2
    assert {e.id for e in actual_entities} == {e1.id, e2.id}
    
def test_iter_dynamic_entities(create_entity):
    e1 = create_entity(constant_velocity_motion())
    e2 = create_entity(constant_acceleration_motion())
    e3 = create_entity(static_motion())
    
    sc = Scene()
    sc.add_entity(e1)
    sc.add_entity(e2)
    sc.add_entity(e3)
    
    actual_entities = list(sc.iter_dynamic())
    
    assert len(actual_entities) == 2
    assert {e.id for e in actual_entities} == {e1.id, e2.id}
    
def test_test_iter_all(create_entity):
    e1 = create_entity(static_motion())
    e2 = create_entity(static_motion())
    e3 = create_entity(static_motion())
    
    sc = Scene()
    sc.add_entity(e1)
    sc.add_entity(e2)
    
    actual_entities = list(sc.iter_all())
    
    assert len(actual_entities) == 2
    assert {e.id for e in actual_entities} == {e1.id, e2.id}
import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from src.mimo.geometry.motion_model import MotionModel
from src.mimo.entity.entity import Entity
from src.mimo.scene.scene import Scene, DuplicateEntityError, EntityNotFoundError

#####################################################################################
# Fixtures
#####################################################################################

@pytest.fixture
def mock_motion_model():
    return Mock(spec=MotionModel)

@pytest.fixture
def entity_fixture(mock_motion_model):
    return Entity(mock_motion_model)

@pytest.fixture
def create_entity(mock_motion_model):
    
    def _create_entity():
        return Entity(mock_motion_model)
    
    return _create_entity

#####################################################################################
# Add and get tests
#####################################################################################

def test_add_and_get_static_entity(create_entity):
    e1 = create_entity()
    sc = Scene()
    sc.add_entity(e1, static=True)
    
    assert sc.get_entity(e1.id) == e1

def test_add_and_get_dynamic_entity(create_entity):
    e1 = create_entity()
    sc = Scene()
    sc.add_entity(e1, static=False)
    
    assert sc.get_entity(e1.id) == e1

def test_add_duplicate_entity_raises_error(create_entity):
    e1 = create_entity()
    sc = Scene()
    sc.add_entity(e1, static=True)
    
    with pytest.raises(DuplicateEntityError):
        sc.add_entity(e1, static=False)

def test_missing_entity_raises_error():
    sc = Scene()
    
    with pytest.raises(EntityNotFoundError):
        sc.get_entity("random_id")

#####################################################################################
# Remove Tests
#####################################################################################        

def test_remove_entity(create_entity):
    e1 = create_entity()
    sc = Scene()
    sc.add_entity(e1, static=True)
    sc.remove_entity(e1.id)
    
    with pytest.raises(EntityNotFoundError):
        sc.get_entity(e1)

def test_remove_missing_entity_raises_error(create_entity):
    e1 = create_entity()
    sc = Scene()
    sc.add_entity(e1)
    
    with pytest.raises(EntityNotFoundError):
        sc.remove_entity("Non_existent_id")        

########################################################################################
# Iterator Tests
########################################################################################

def test_iter_static_entities(create_entity):
    e1 = create_entity()
    e2 = create_entity()
    e3 = create_entity()
    
    sc = Scene()
    sc.add_entity(e1, static=True)
    sc.add_entity(e2, static=True)
    sc.add_entity(e3, static=False)
    
    actual_entities = list(sc.iter_static())
    
    assert len(actual_entities) == 2
    assert {e.id for e in actual_entities} == {e1.id, e2.id}
    
    def test_iter_dynamic_entities(create_entity):
        e1 = create_entity()
        e2 = create_entity()
        e3 = create_entity()
        
        sc = Scene()
        sc.add_entity(e1, static=False)
        sc.add_entity(e2, static=False)
        sc.add_entity(e3, static=True)
        
        actual_entities = list(sc.iter_dynamic())
        
        assert len(actual_entities) == 2
        assert {e.id for e in actual_entities} == {e1.id, e2.id}
    
    def test_test_iter_all(create_entity):
        e1 = create_entity()
        e2 = create_entity()
        e3 = create_entity()
        
        sc = Scene()
        sc.add_entity(e1, static=False)
        sc.add_entity(e2, static=True)
        
        actual_entities = list(sc.iter_all())
        
        assert len(actual_entities) == 2
        assert {e.id for e in actual_entities} == {e1.id, e2.id}
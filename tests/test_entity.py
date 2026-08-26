import pytest
from unittest.mock import Mock, MagicMock

from mimo.entity.radar_component import RadarComponent
from mimo.geometry.motion_model import Motion
from mimo.entity.entity import (
    Entity, 
    ComponentNotFoundError, 
    ComponentAlreadyAttachedError
)

class MockComponent(RadarComponent):
    pass

@pytest.fixture
def mock_motion_model():
    return Mock(spec=Motion)

@pytest.fixture
def entity_fixture(mock_motion_model):
    return Entity(mock_motion_model)

@pytest.fixture
def create_entity(mock_motion_model):
    
    def _create_entity():
        return Entity(motion=mock_motion_model)

    return _create_entity

def test_entities_generate_unique_uuids(create_entity):
    e1 = create_entity()
    e2 = create_entity()
    
    assert e1.id != e2.id

def test_entity_add_get_component(mock_motion_model):
    e_1 = Entity(mock_motion_model)
    component = MockComponent()
    
    e_1.add_component(component)
    retrieved_component = e_1.get_component(MockComponent)
    
    assert retrieved_component is component

def test_duplicate_component_raises_error(mock_motion_model):
    e_1 = Entity(mock_motion_model)
    component = MockComponent()
    e_1.add_component(component)
    
    with pytest.raises(ComponentAlreadyAttachedError):
        e_1.add_component(MockComponent())

def test_missing_component_raises_error(entity_fixture):
    
    with pytest.raises(ComponentNotFoundError):
        entity_fixture.get_component(MockComponent)

def test_lifecycle_hooks_called_once(entity_fixture):
    
    component = MockComponent()
    
    # Spy on lifecyle methods
    component.on_attach = MagicMock(side_effect=component.on_attach)
    component.on_detach = MagicMock(side_effect=component.on_detach)
    
    # Test attach lifecycle
    entity_fixture.add_component(component)
    component.on_attach.assert_called_once_with(entity_fixture)
    component.on_detach.assert_not_called()
    
    # Reset mock counters for clean detach isolation.
    component.on_attach.reset_mock()
    
    # Test detach lifecyle
    entity_fixture.remove_component(MockComponent)
    component.on_detach.assert_called_once()
    component.on_attach.assert_not_called()

def test_component_back_reference_cleared_on_removal(entity_fixture):
    
    component = MockComponent()
    
    entity_fixture.add_component(component)
    assert component.attached is True
    assert component.entity is entity_fixture
    
    entity_fixture.remove_component(MockComponent)
    assert component._entity is None
    assert component.attached is False

    with pytest.raises(RuntimeError):
        _ = component.entity
    

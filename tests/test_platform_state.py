import pytest
import numpy as np
from src.mimo.geometry.platform_state import PlatformState

@pytest.fixture
def origin_platform():
    """Provides a stationery platform at the origin."""
    return PlatformState(position = np.array([0, 0, 0]),
                         velocity = np.array([0, 0, 0]),
                         orientation = np.array([1, 0, 0, 0]),
                         angular_velocity=np.zeros(3),
                         time = 0.0)

def test_initialisation_shape_errors():
    """Ensure __post_init__ catches invalid array shapes."""
    with pytest.raises(ValueError):
        PlatformState(position = np.array([0, 0]),
                      velocity = np.zeros(0),
                      orientation = np.array([1, 0, 0, 0]),
                      angular_velocity=np.zeros(3),
                      time = 0.0)
        

def test_distance_and_relative_position(origin_platform):
    target = PlatformState(
        position= np.array([3, 4, 0]),
        velocity= np.zeros(3),
        orientation= np.array([1, 0, 0, 0]),
        angular_velocity=np.zeros(3),
        time = 0.0
    )
    
    np.testing.assert_allclose(
        origin_platform.relative_position_to(target),
        np.array([3, 4, 0])
    )
    assert origin_platform.distance_to(target)

def test_radial_velocity(origin_platform):
    """Assert the sign and value of the closing velocity."""
    
    p2 = PlatformState(position = np.array([10, 0, 0]),
                       velocity = np.array([-1, 0, 0]),
                       orientation = np.array([1, 0, 0, 0]),
                       angular_velocity= np.zeros(3),
                       time = 0)
    
    radial_velocity = origin_platform.radial_velocity_to(p2)
    
    np.testing.assert_allclose(
        radial_velocity,
        -1
    )


    
    
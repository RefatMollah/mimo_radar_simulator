import pytest
import numpy as np

from src.mimo.geometry.motion_model import (
    Motion,
    StaticMotion,
    ConstantVelocityMotion,
    ConstantAccelerationMotion,
)

@pytest.fixture
def origin_platform():
    """Provides a stationery platform at the origin."""
    return StaticMotion(
        position= np.zeros(3),
        velocity= np.zeros(3),
        orientation= np.array([1, 0, 0, 0]),
        angular_velocity=np.zeros(3), 
    )

def test_initialisation_shape_errors():
    """Ensure __post_init__ catches invalid array shapes."""
    with pytest.raises(ValueError):
        StaticMotion(
            position = np.array([0, 0]),
            velocity = np.zeros(0),
            orientation = np.array([1, 0, 0, 0]),
            angular_velocity=np.zeros(3),
        )
        




    
    
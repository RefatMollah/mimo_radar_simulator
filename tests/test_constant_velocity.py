import numpy as np
import pytest

from src.mimo.geometry.motion_model import ConstantVelocity
from src.mimo.geometry.motion_model import PlatformState

def test_constant_velocity_propagation():
    state = PlatformState(
        position = np.array([0.0, 0.0, 0.0]),
        velocity = np.array([10.0, 0.0, 0.0]),
        orientation = np.array([1, 0, 0, 0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        time = 0.0
    )
    
    model = ConstantVelocity(state)
    
    propagated = model.get_state(5.0)
    
    np.testing.assert_allclose(
        propagated.position,
        np.array([50.0, 0.0, 0.0])
    )
    
    np.testing.assert_allclose(
        propagated.velocity,
        state.velocity
    )
    assert propagated.time == 5.0

    
def test_constant_velocity_past_time_raises():

    state = PlatformState(
        position=np.zeros(3),
        velocity=np.ones(3),
        orientation=np.array([1, 0, 0, 0]),
        angular_velocity=np.zeros(3),
        time=10.0,
    )

    model = ConstantVelocity(state)

    with pytest.raises(ValueError):
        model.get_state(5.0) 
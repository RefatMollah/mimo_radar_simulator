import numpy as np
from abc import ABC, abstractmethod
from .platform_state import *


class MotionModel(ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def get_state(self, time: float) -> PlatformState:
        pass

class StationaryModel(MotionModel):
    
    def __init__(self, initial_state: PlatformState) -> None:
        super().__init__()
        self.initial_state = initial_state
    
    def get_state(self, time: float) -> PlatformState:
        return PlatformState(
            position=self.initial_state.position,
            velocity=self.initial_state.velocity,
            orientation=self.initial_state.orientation,
            angular_velocity=self.initial_state.angular_velocity,
            time = time
        )

class ConstantVelocity(MotionModel):
    
    def __init__(self, initial_state : PlatformState) -> None:
        super().__init__()
        self.initial_state = initial_state
                
    def get_state(self, time: float) -> PlatformState:
        dt = time - self.initial_state.time
        if (dt < 0):
            raise ValueError("Cannot extrapolate state into the past.")
        
        position = (self.initial_state.position 
                        + self.initial_state.velocity * dt)
        
        omega = self.initial_state.angular_velocity
        omega_norm = np.linalg.norm(omega)
        
        if omega_norm < 1e-12:
            orientation = self.initial_state.orientation
        else:
            axis = omega / omega_norm
            theta = np.linalg.norm(omega) * dt
            dq = np.array([
                    np.cos(theta/2),
                    *(axis*np.sin(theta/2))
                ])
            
            orientation = quat.normalise(
                quat.quat_multiply(
                    self.initial_state.orientation,
                    dq
                )
            )    
        
        return PlatformState(
                    position= position,
                    velocity= self.initial_state.velocity,
                    orientation= orientation,
                    angular_velocity= self.initial_state.angular_velocity,
                    time= time
                )

    
"""Kinematics, quaternion, and sensor-steering primitives."""

from .motion_model import ConstantAccelerationMotion, ConstantVelocityMotion, StaticMotion
from .quat import axis_angle_delta_quat, quat_multiply, quat_normalise, quat_rotate
from .spatial_engine import State, state_at
from .sensor_steering import (
    CircularScan,
    ConicalScan,
    FixedBoresight,
    FullAzimuthScan,
    RateLimitedGimbal,
    SectorScan,
    ScanningPattern,
    SteeringActuator,
    SteeringCommand,
    SteeringState,
)

__all__ = ["CircularScan", "ConicalScan", "ConstantAccelerationMotion", "ConstantVelocityMotion", "FixedBoresight", "FullAzimuthScan", "RateLimitedGimbal", "ScanningPattern", "SectorScan", "State", "StaticMotion", "SteeringActuator", "SteeringCommand", "SteeringState", "axis_angle_delta_quat", "quat_multiply", "quat_normalise", "quat_rotate", "state_at"]

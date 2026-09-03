import numpy as np
import pytest

from mimo.entity.radar_component import RadarSensor, RxElement, TxElement
from mimo.geometry.quat import quat_rotate
from mimo.geometry.sensor_steering import (
    CircularScan,
    FixedBoresight,
    SectorScan,
    ScanningPatternBatch,
)
from mimo.geometry._jax_backend import steering_batch_class_for


def evaluate(law, time):
    component = type("Component", (), {"steering": law})()
    batch_type = steering_batch_class_for(type(law))
    return batch_type.from_components([component], np.float64, np).evaluate(time).orientations[0]


def test_radar_sensor_has_no_mount_pose_and_defaults_to_fixed_boresight():
    sensor = RadarSensor(transmitters=[TxElement()], receivers=[RxElement()])

    assert isinstance(sensor.steering, FixedBoresight)
    assert not hasattr(sensor, "local_position")
    assert not hasattr(sensor, "local_rotation")


def test_circular_scan_rotates_boresight_at_fixed_elevation():
    q = evaluate(CircularScan(rate=np.pi / 2, elevation=np.pi / 6), 1.0)
    direction = quat_rotate(q[None, :], np.array([[1.0, 0.0, 0.0]]), dtype=np.float64)[0]

    np.testing.assert_allclose(direction, [0.0, np.cos(np.pi / 6), np.sin(np.pi / 6)], atol=1e-12)


def test_sector_scan_sweeps_back_and_forth():
    law = SectorScan(-1.0, 1.0, rate=1.0, elevation=0.25)
    q_start = evaluate(law, 0.0)
    q_end = evaluate(law, 2.0)
    q_back = evaluate(law, 4.0)

    np.testing.assert_allclose(q_start, q_back, atol=1e-12)
    assert not np.allclose(q_start, q_end)


def test_scan_parameters_are_validated():
    with pytest.raises(ValueError):
        CircularScan(rate=-1.0)
    with pytest.raises(ValueError):
        SectorScan(1.0, 1.0, rate=1.0)

import math

import numpy as np
import pytest

from ttcoach import geometry as g


def test_angle_3pt_right_angle_2d():
    assert g.angle_3pt((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)


def test_angle_3pt_straight_arm_3d_is_180():
    assert g.angle_3pt((0, 0, 0), (1, 1, 1), (2, 2, 2)) == pytest.approx(180.0)


def test_angle_3pt_nan_when_points_coincide_or_missing():
    assert math.isnan(g.angle_3pt((0, 0), (0, 0), (1, 1)))
    assert math.isnan(g.angle_3pt((np.nan, 0), (0, 0), (1, 1)))


def test_wrap_deg_range():
    assert g.wrap_deg(190) == pytest.approx(-170)
    assert g.wrap_deg(-180) == pytest.approx(180)
    assert g.wrap_deg(540) == pytest.approx(180)
    assert g.wrap_deg(-10) == pytest.approx(-10)


def test_yaw_deg_measures_horizontal_direction():
    # world coords: x right, y down, z depth; yaw ignores y
    assert g.yaw_deg((0, 0, 0), (1, 5, 0)) == pytest.approx(0.0)
    assert g.yaw_deg((0, 0, 0), (0, -3, 1)) == pytest.approx(90.0)


def test_lean_from_vertical_deg():
    # y points down, so "up" is -y
    assert g.lean_from_vertical_deg((0, 0, 0), (0, -1, 0)) == pytest.approx(0.0)
    assert g.lean_from_vertical_deg((0, 0, 0), (1, -1, 0)) == pytest.approx(45.0)
    assert g.lean_from_vertical_deg((0, 0, 0), (0, -1, 1)) == pytest.approx(45.0)


def test_point_line_distance_3d():
    # elbow 0.3 m to the side of a vertical torso line
    assert g.point_line_distance((0.3, -0.5, 0), (0, 0, 0), (0, -1, 0)) == pytest.approx(0.3)


def test_horizontal_turn_deg_sign_follows_forward_motion():
    # playing shoulder rotates from +x toward +z (forward), facing direction +z
    turn = g.horizontal_turn_deg(start=(1, 0, 0), end=(0, 0, 1), forward=(0, 0, 1))
    assert turn == pytest.approx(90.0)
    back = g.horizontal_turn_deg(start=(0, 0, 1), end=(1, 0, 0), forward=(0, 0, 1))
    assert back == pytest.approx(-90.0)

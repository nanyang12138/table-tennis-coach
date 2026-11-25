import math

import numpy as np
import pytest

from conftest import BASE, build_series, mirror
from ttcoach import metrics
from ttcoach.signals import Phases

PH = Phases(ready=0, backswing_end=10, mid_forward=15, contact=20, finish=30, recovery=35)


def frames(n=40, **overrides_by_frame):
    return [dict() for _ in range(n)]


def test_elbow_angle_matches_constructed_arm():
    # upper arm straight down, forearm 60 deg off the straight line -> 120 deg at the elbow
    c60, s60 = math.cos(math.radians(60)), math.sin(math.radians(60))
    arm = {"right_shoulder": (-0.2, -0.5, 0.0), "right_elbow": (-0.2, -0.25, 0.0),
           "right_wrist": (-0.2, -0.25 + 0.25 * c60, -0.25 * s60)}
    m = metrics.stroke_metrics(build_series([arm] * 40), PH, hand="right")
    assert m["elbow_angle_deg"]["contact"] == pytest.approx(120.0, abs=0.5)


def test_durations_follow_series_fps():
    m100 = metrics.stroke_metrics(build_series([{}] * 40, fps=100), PH, hand="right")
    m200 = metrics.stroke_metrics(build_series([{}] * 40, fps=200), PH, hand="right")
    assert (m100["backswing_ms"], m100["forward_ms"], m100["follow_ms"]) == pytest.approx((100, 100, 100))
    assert (m200["backswing_ms"], m200["forward_ms"], m200["follow_ms"]) == pytest.approx((50, 50, 50))


def test_weight_shift_delta_positive_when_body_moves_over_other_foot():
    # world coords are hip-centred: hips over the right foot -> right ankle under the hip
    over_right = {"right_ankle": (0.0, 0.9, 0.0), "left_ankle": (0.6, 0.9, 0.0)}
    over_left = {"right_ankle": (-0.6, 0.9, 0.0), "left_ankle": (0.0, 0.9, 0.0)}
    poses = [over_right] * 20 + [over_left] * 20
    m = metrics.stroke_metrics(build_series(poses), PH, hand="right")
    ws = m["weight_shift"]
    assert ws["backswing_end"] == pytest.approx(0.0, abs=0.05)
    assert ws["finish"] == pytest.approx(1.0, abs=0.05)
    assert ws["delta"] > 0.8


def test_side_guess_forehand_when_wrist_out_on_playing_side():
    fh = {"right_wrist": (-0.55, -0.2, -0.25)}
    m = metrics.stroke_metrics(build_series([fh] * 40), PH, hand="right")
    assert m["side_guess"] == "forehand"


def test_side_guess_backhand_when_wrist_in_front_of_body():
    bh = {"right_wrist": (0.05, -0.25, -0.35)}
    m = metrics.stroke_metrics(build_series([bh] * 40), PH, hand="right")
    assert m["side_guess"] == "backhand"


def test_left_hander_is_mirrored():
    fh = dict(BASE)
    fh["right_wrist"] = (-0.55, -0.2, -0.25)
    lefty = mirror(fh)
    m = metrics.stroke_metrics(build_series([lefty] * 40), PH, hand="left")
    assert m["side_guess"] == "forehand"


def test_contact_height_level_between_hip_and_shoulder():
    # shoulders at y=-0.5, hips at y=0 (world and image are the same projection here)
    at_hip = metrics.stroke_metrics(build_series([{"right_wrist": (-0.4, 0.0, -0.2)}] * 40), PH, hand="right")
    at_sh = metrics.stroke_metrics(build_series([{"right_wrist": (-0.4, -0.5, -0.2)}] * 40), PH, hand="right")
    mid = metrics.stroke_metrics(build_series([{"right_wrist": (-0.4, -0.25, -0.2)}] * 40), PH, hand="right")
    assert at_hip["contact_height_level"] == pytest.approx(0.0, abs=0.02)
    assert at_sh["contact_height_level"] == pytest.approx(1.0, abs=0.02)
    assert mid["contact_height_level"] == pytest.approx(0.5, abs=0.02)


def test_shoulder_turn_positive_when_playing_shoulder_comes_forward():
    # backswing: right shoulder pulled back (+z); finish: right shoulder forward (-z, toward camera)
    back = {"right_shoulder": (-0.17, -0.5, 0.1), "left_shoulder": (0.17, -0.5, -0.1)}
    fwd = {"right_shoulder": (-0.17, -0.5, -0.1), "left_shoulder": (0.17, -0.5, 0.1)}
    poses = [back] * 20 + [fwd] * 20
    m = metrics.stroke_metrics(build_series(poses), PH, hand="right")
    assert m["shoulder_turn_deg"] > 30


def test_x_factor_is_angle_between_shoulder_and_hip_lines():
    coiled = {"right_shoulder": (-0.17, -0.5, 0.1), "left_shoulder": (0.17, -0.5, -0.1)}
    m = metrics.stroke_metrics(build_series([coiled] * 40), PH, hand="right")
    expected = abs(math.degrees(math.atan2(0.2, 0.34)))
    assert m["x_factor_deg"] == pytest.approx(expected, abs=1.0)


def test_low_visibility_marks_stroke_unreliable():
    m = metrics.stroke_metrics(build_series([{}] * 40, vis={"right_wrist": 0.2}), PH, hand="right")
    assert "right_wrist" in m["quality"]["low_visibility_parts"]
    assert m["quality"]["reliable"] is False


def test_knee_angle_is_none_when_leg_not_visible():
    # legs hidden behind the table: MediaPipe still guesses positions but with low visibility
    m = metrics.stroke_metrics(build_series([{}] * 40, vis={"right_knee": 0.2, "right_ankle": 0.1}), PH, hand="right")
    assert m["knee_angle_deg"]["playing"]["contact"] is None
    assert m["knee_min_deg"]["playing"] is None
    assert m["weight_shift"]["delta"] is None
    assert m["knee_angle_deg"]["other"]["contact"] is not None


def test_elbow_angle_is_none_when_arm_not_visible():
    m = metrics.stroke_metrics(build_series([{}] * 40, vis={"right_elbow": 0.1}), PH, hand="right")
    assert m["elbow_angle_deg"]["contact"] is None


def test_sequence_not_judged_when_fps_too_low():
    m = metrics.stroke_metrics(build_series([{}] * 40, fps=25), PH, hand="right")
    assert m["sequence_order_ok"] is None


def test_all_appendix_c_names_present():
    m = metrics.stroke_metrics(build_series([{}] * 40), PH, hand="right")
    for key in ["side_guess", "backswing_ms", "forward_ms", "follow_ms", "elbow_angle_deg", "knee_angle_deg",
                "knee_min_deg", "trunk_lean_deg", "shoulder_turn_deg", "hip_turn_deg", "x_factor_deg",
                "shoulder_width_ratio", "weight_shift", "contact_height_level", "contact_forward_m",
                "contact_lateral_m", "elbow_body_gap_m", "swing_path", "finish_height_level", "head_motion",
                "free_arm_height_level", "peak_hand_speed", "sequence_ms", "sequence_order_ok",
                "stance_width_ratio", "recovery_dist", "quality"]:
        assert key in m, key
    assert set(m["elbow_angle_deg"]) == {"backswing_end", "contact", "finish"}
    assert set(m["knee_angle_deg"]) == {"playing", "other"}
    assert set(m["knee_angle_deg"]["playing"]) == {"ready", "backswing_end", "contact"}
    assert set(m["swing_path"]) >= {"up", "horiz", "angle_deg"}
    assert set(m["sequence_ms"]) == {"hip", "shoulder", "elbow"}


def test_hand_speed_ignores_whole_body_movement():
    # the whole player slides sideways (footwork) without moving the arm relative to the body
    poses = []
    for i in range(30):
        dx = 0.02 * i
        poses.append({k: (x + dx, y, z) for k, (x, y, z) in BASE.items()})
    sp = metrics.hand_speed(build_series(poses, fps=50), "right")
    assert np.nanmax(sp) < 0.05


def test_hand_speed_uses_shoulders_not_jittery_hips():
    # hips jitter (hidden behind the table) while the arm is still relative to the shoulders
    poses = []
    for i in range(30):
        j = 0.05 if i % 2 else -0.05
        poses.append({"left_hip": (0.12, j, 0.0), "right_hip": (-0.12, j, 0.0)})
    sp = metrics.hand_speed(build_series(poses, fps=50), "right")
    assert np.nanmax(sp) < 0.05


def test_hand_speed_measures_sustained_motion_in_torso_lengths_per_second():
    # torso length is 0.5 m; wrist moves 0.5 m/s sideways = 1 torso/s, sampled at 120 fps
    fps = 120
    poses = [{"right_wrist": (-0.3 - 0.5 * i / fps, -0.05, -0.15)} for i in range(60)]
    sp = metrics.hand_speed(build_series(poses, fps=fps), "right")
    assert np.nanmedian(sp[10:50]) == pytest.approx(1.0, rel=0.05)


def test_photo_metrics_reports_joint_angles_for_one_frame():
    c60, s60 = math.cos(math.radians(60)), math.sin(math.radians(60))
    arm = {"right_shoulder": (-0.2, -0.5, 0.0), "right_elbow": (-0.2, -0.25, 0.0),
           "right_wrist": (-0.2, -0.25 + 0.25 * c60, -0.25 * s60)}
    s = build_series([arm], fps=1)
    m = metrics.photo_metrics(s, hand="right")
    assert m["elbow_angle_deg"]["playing"] == pytest.approx(120.0, abs=0.5)
    assert set(m["knee_angle_deg"]) == {"playing", "other"}
    assert m["trunk_lean_deg"] == pytest.approx(0.0, abs=0.5)
    assert "stance_width_ratio" in m and "wrist_height_level" in m


def test_video_metrics_rhythm_and_consistency():
    strokes = [{"t_contact_s": 1.0, "side_guess": "forehand", "metrics": {"elbow_angle_deg": {"contact": 100}}},
               {"t_contact_s": 2.0, "side_guess": "forehand", "metrics": {"elbow_angle_deg": {"contact": 110}}},
               {"t_contact_s": 3.5, "side_guess": "forehand", "metrics": {"elbow_angle_deg": {"contact": 120}}}]
    v = metrics.video_metrics(strokes)
    assert v["rhythm"]["intervals_s"] == pytest.approx([1.0, 1.5])
    fh = v["consistency"]["forehand"]["elbow_angle_deg.contact"]
    assert fh["mean"] == pytest.approx(110.0)
    assert fh["n"] == 3


def test_side_guess_uses_shoulders_when_hips_hidden():
    fh = {"right_wrist": (-0.55, -0.2, -0.25)}
    m = metrics.stroke_metrics(build_series([fh] * 40, vis={"left_hip": 0.2, "right_hip": 0.2}), PH, hand="right")
    assert m["side_guess"] == "forehand"


def test_contact_height_level_estimated_when_hips_hidden():
    # wrist halfway between shoulder (y=-0.5) and hip (y=0) heights; hips hidden behind the table
    s = build_series([{"right_wrist": (-0.4, -0.25, -0.2)}] * 40, vis={"left_hip": 0.2, "right_hip": 0.2})
    m = metrics.stroke_metrics(s, PH, hand="right")
    assert m["contact_height_level"] == pytest.approx(0.5, abs=0.05)
    assert any("hip" in n for n in m["quality"]["notes"])


def test_photo_metrics_respect_visibility():
    s = build_series([{}], fps=1, vis={"right_ankle": 0.2, "left_ankle": 0.2})
    m = metrics.photo_metrics(s, hand="right")
    assert m["knee_angle_deg"]["playing"] is None
    assert m["stance_width_ratio"] is None
    assert m["elbow_angle_deg"]["playing"] is not None

"""Regression tests for problems found when fresh agents used the skill end to end."""
import math

import numpy as np
import pytest

from conftest import BASE, build_series
from ttcoach import metrics, pipeline, render, report
from ttcoach.signals import Phases

PH = Phases(ready=0, backswing_end=10, mid_forward=15, contact=20, finish=30, recovery=35)


def test_timeline_frames_stay_on_the_frame_grid():
    # 8.5 s is half a frame at 25 fps; rounding used to drop every other frame
    assert pipeline.timeline_frames(8.5, 9.8, 0.04, 25.0) == list(range(213, 246))
    assert pipeline.timeline_frames(1.0, 2.0, 0.2, 25.0) == [25, 30, 35, 40, 45, 50]


def test_each_task_run_gets_its_own_folder(tmp_path):
    a = pipeline.task_dir(tmp_path, "timeline", timeline=(8.48, 9.8, 0.04))
    b = pipeline.task_dir(tmp_path, "timeline", timeline=(5.0, 6.0, 0.04))
    assert a != b and a.name.startswith("timeline_") and b.name.startswith("timeline_")
    m1 = pipeline.task_dir(tmp_path, "strokes")
    m1.mkdir()
    m2 = pipeline.task_dir(tmp_path, "strokes")
    assert m1 != m2 and m2.name.startswith("measured_")
    at = pipeline.task_dir(tmp_path, "at", at=[1.12, 1.2])
    assert at.name.startswith("at_")


def test_left_right_swapped_frames_are_blanked():
    poses = [dict() for _ in range(40)]
    swapped = {"left_shoulder": BASE["right_shoulder"], "right_shoulder": BASE["left_shoulder"],
               "left_hip": BASE["right_hip"], "right_hip": BASE["left_hip"]}
    for i in (20, 21, 22):
        poses[i] = swapped
    s = build_series(poses, fps=50)
    pipeline.clean_series(s)
    assert not s.present[20] and not s.present[22]
    assert s.present[10] and s.present[30]


def test_implausible_turn_marks_stroke_unreliable():
    # shoulders flip between backswing end and finish -> a ~180 deg "turn" that no body can do
    flip = {"left_shoulder": BASE["right_shoulder"], "right_shoulder": BASE["left_shoulder"]}
    poses = [dict()] * 20 + [flip] * 20
    m = metrics.stroke_metrics(build_series(poses), PH, hand="right")
    assert m["quality"]["reliable"] is False
    assert any("implausible" in n for n in m["quality"]["notes"])


def test_legs_hidden_for_most_of_the_stroke_blank_leg_metrics():
    # knee visible in a few frames only (the table edge mistaken for a leg): stroke-level gate
    s = build_series([{}] * 40)
    s.vis[:, 26] = 0.3          # right_knee mostly hidden
    s.vis[18:22, 26] = 0.7      # but "visible" around contact
    m = metrics.stroke_metrics(s, PH, hand="right")
    assert m["knee_angle_deg"]["playing"]["contact"] is None
    assert m["knee_min_deg"]["playing"] is None


def test_quality_coverage_uses_the_swing_not_the_ready_frames():
    s = build_series([{}] * 40)
    s.present[0:8] = False      # nobody detected in the ready frames
    m = metrics.stroke_metrics(s, PH, hand="right")
    assert m["quality"]["reliable"] is True


def test_shoulder_width_ratio_reported_at_four_phases():
    m = metrics.stroke_metrics(build_series([{}] * 40), PH, hand="right")
    assert set(m["shoulder_width_ratio"]) == {"ready", "backswing_end", "contact", "finish"}


def test_photo_metrics_include_shoulder_width_ratio():
    m = metrics.photo_metrics(build_series([{}], fps=1), hand="right")
    assert m["shoulder_width_ratio"] == pytest.approx(0.8, abs=0.02)


def test_hand_box_reaches_along_the_forearm():
    xy = np.full((33, 2), np.nan)
    vis = np.ones(33)
    xy[14] = (100.0, 100.0)   # right elbow
    xy[16] = (200.0, 100.0)   # right wrist, arm pointing +x
    xy[20] = (210.0, 100.0)
    box = render.hand_box(xy, vis, "right", torso_orig=100.0, frame_shape=(1000, 1000, 3))
    cx = (box[0] + box[2]) / 2.0
    assert cx > 215.0          # centre moved past the hand toward the racket head
    assert box[2] - box[0] >= 170


def test_given_side_overrides_guess():
    s = build_series([{}] * 100, fps=50)
    st = pipeline.measure_strokes(s, "right", [(0.2, 0.4, 0.5, 0.7, 1.0)], side="backhand")
    assert st[0]["side_guess"] == "backhand" and st[0]["side_given"] is True


def test_consistency_only_counts_reliable_strokes():
    def mk(t, v, ok):
        return {"t_contact_s": t, "side_guess": "forehand",
                "metrics": {"elbow_angle_deg": {"contact": v}, "quality": {"reliable": ok}}}
    v = metrics.video_metrics([mk(1, 100, True), mk(2, 110, True), mk(3, 300, False)])
    assert v["consistency"]["forehand"]["elbow_angle_deg.contact"]["n"] == 2


def test_auto_summary_sends_reader_to_timeline_not_to_a_typical_stroke(tmp_path):
    res = {"kind": "video", "task": "auto", "mode": "pose", "hand": "right", "player": "auto",
           "video": {"path": "x.mp4", "duration_s": 3, "file_fps": 25, "fps_source": "timestamps", "real_fps": 25,
                     "slowmo_factor": 1, "analysis_fps": 25, "stride": 1, "width": 640, "height": 480},
           "warnings": [], "files": [],
           "strokes": [{"index": 1, "source": "auto", "t_contact_s": 1.0, "side_guess": "forehand", "phases": {},
                        "metrics": {"quality": {"reliable": True}}}]}
    res["examples"] = report.pick_examples(res["strokes"], task="auto")
    text = report.write(tmp_path, res).read_text(encoding="utf-8")
    assert "最典型" not in text
    assert "timeline_" in text


def test_skeleton_jump_at_a_phase_frame_blanks_that_phase():
    poses = [dict() for _ in range(40)]
    poses[30] = {"right_wrist": (0.25, -0.6, -0.15), "right_elbow": (0.1, -0.4, 0.0)}  # finish frame glitch
    m = metrics.stroke_metrics(build_series(poses), PH, hand="right")
    assert m["elbow_angle_deg"]["finish"] is None
    assert m["finish_height_level"] is None
    assert m["quality"]["reliable"] is False
    assert any("finish" in n and "unstable" in n for n in m["quality"]["notes"])
    assert m["elbow_angle_deg"]["contact"] is not None


def test_elbow_angle_range_over_neighbouring_frames():
    m = metrics.stroke_metrics(build_series([{}] * 40), PH, hand="right")
    lo, med, hi = m["elbow_angle_range_deg"]["contact"]
    assert lo <= med <= hi


def test_shoulder_width_minimum_between_ready_and_contact():
    poses = [dict() for _ in range(40)]
    for i in range(12, 16):   # trunk turned sideways: shoulders close together in the image
        poses[i] = {"left_shoulder": (0.05, -0.5, -0.19), "right_shoulder": (-0.05, -0.5, 0.19)}
    m = metrics.stroke_metrics(build_series(poses), PH, hand="right")
    sw = m["shoulder_width_min"]
    assert sw["value"] == pytest.approx(0.2, abs=0.03)
    assert 12 <= sw["i"] <= 15


def test_trunk_tilt_2d():
    m = metrics.stroke_metrics(build_series([{}] * 40), PH, hand="right")
    assert m["trunk_tilt_2d_deg"]["contact"] == pytest.approx(0.0, abs=0.5)
    lean = {"left_shoulder": (0.2 + 0.5, -0.5, 0.0), "right_shoulder": (-0.2 + 0.5, -0.5, 0.0)}
    m2 = metrics.stroke_metrics(build_series([lean] * 40), PH, hand="right")
    assert m2["trunk_tilt_2d_deg"]["contact"] == pytest.approx(45.0, abs=1.0)


def test_arm_falling_from_a_high_finish_is_not_a_forward_swing():
    # wrist above the shoulder, moving down and across the body (recovery after a high finish)
    pts = [(-0.30 + 0.03 * k, -0.9 + 0.05 * k) for k in range(20)]
    s = build_series([{"right_wrist": (x, y, -0.2)} for x, y in pts], fps=50)
    sc = pipeline.swing_scores(s, "right")
    assert sc[5] < -0.2

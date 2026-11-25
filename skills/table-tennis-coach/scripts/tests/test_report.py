from ttcoach import report


def _stroke(i):
    return {"index": i, "source": "given", "t_contact_s": 1.0 * i, "side_guess": "forehand",
            "phases": {}, "metrics": {"quality": {"reliable": True}, "elbow_angle_deg": {"contact": 120},
                                      "knee_min_deg": {"playing": 140}, "shoulder_turn_deg": 20,
                                      "contact_height_level": 0.5, "forward_ms": 120}}


def test_measured_summary_points_to_measured_images(tmp_path):
    res = {"kind": "video", "task": "strokes", "mode": "pose", "hand": "right", "player": "auto",
           "video": {"path": "x.mp4", "duration_s": 3, "file_fps": 240, "fps_source": "timestamps", "real_fps": 240,
                     "slowmo_factor": 1, "analysis_fps": 120, "stride": 2, "width": 1920, "height": 1080},
           "warnings": [], "strokes": [_stroke(1), _stroke(2)], "files": []}
    res["examples"] = report.pick_examples(res["strokes"])
    text = report.write(tmp_path, res).read_text(encoding="utf-8")
    assert "measured_01_sheet.jpg" in text
    assert "stroke_01_sheet.jpg" not in text
    assert "按你给的时刻" in text


def test_photo_summary_lists_measurements(tmp_path):
    res = {"kind": "image", "task": "photo", "mode": "image-pose", "hand": "right", "player": "auto",
           "video": {"path": "grip.jpg", "width": 800, "height": 600}, "warnings": [], "strokes": [], "files": [],
           "photo_metrics": {"elbow_angle_deg": {"playing": 136.2, "other": 120.0},
                             "knee_angle_deg": {"playing": None, "other": 101.9}, "trunk_lean_deg": 12.0,
                             "stance_width_ratio": None, "wrist_height_level": 0.4, "free_arm_height_level": 0.1,
                             "quality": {"mean_visibility": 0.7, "low_visibility_parts": ["right_ankle"]}}}
    text = report.write(tmp_path, res).read_text(encoding="utf-8")
    assert "136" in text and "看不清" in text


def test_at_summary_lists_each_moment(tmp_path):
    res = {"kind": "video", "task": "at", "mode": "pose", "hand": "right", "player": "auto",
           "video": {"path": "x.mp4", "duration_s": 3, "file_fps": 30, "fps_source": "timestamps", "real_fps": 30,
                     "slowmo_factor": 1, "analysis_fps": 30, "stride": 1, "width": 640, "height": 480},
           "warnings": [], "strokes": [], "files": [],
           "at": [{"index": 1, "t_file": 1.2, "frame": 36, "pose_found": True,
                   "metrics": {"elbow_angle_deg": {"playing": 101.0, "other": 90.0},
                               "knee_angle_deg": {"playing": 140.0, "other": 150.0}, "trunk_lean_deg": 20.0,
                               "stance_width_ratio": 1.4, "wrist_height_level": 0.6, "free_arm_height_level": 0.2,
                               "quality": {"mean_visibility": 0.9, "low_visibility_parts": []}}},
                  {"index": 2, "t_file": 2.0, "frame": 60, "pose_found": False}]}
    text = report.write(tmp_path, res).read_text(encoding="utf-8")
    assert "at_01.jpg" in text and "1.20" in text and "101" in text
    assert "没认出人" in text

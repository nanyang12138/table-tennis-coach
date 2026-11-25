import json
import os

import cv2
import numpy as np
import pytest

from conftest import build_series
from ttcoach import pipeline, video

VIDEOS = os.environ.get("TTCOACH_TEST_VIDEOS")  # folder with the downloaded Commons test clips


def vid(name):
    if not VIDEOS:
        return None
    for root, _, files in os.walk(VIDEOS):
        if name in files:
            return os.path.join(root, name)
    return None


def write_static_video(path, n=40, fps=30.0, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    frame = np.full((size[1], size[0], 3), 128, np.uint8)
    for _ in range(n):
        w.write(frame)
    w.release()


def codes(result):
    return {w["code"] for w in result["warnings"]}


def test_static_video_without_person_reports_no_strokes(tmp_path):
    src = tmp_path / "static.avi"
    write_static_video(src)
    out = tmp_path / "out"
    res = pipeline.run(str(src), str(out), frames_only=True)
    assert "NO_STROKES" in codes(res)
    assert (out / "analysis.json").exists() and (out / "summary.md").exists()
    assert (out / "timeline_01.jpg").exists()  # evenly spaced frames for looking at the whole clip
    json.loads((out / "analysis.json").read_text(encoding="utf-8"))


def test_capture_fps_sets_slowmo_factor(tmp_path):
    src = tmp_path / "static.avi"
    write_static_video(src, fps=30.0)
    res = pipeline.run(str(src), str(tmp_path / "o"), frames_only=True, capture_fps=240)
    assert res["video"]["slowmo_factor"] == pytest.approx(8.0, rel=0.05)
    assert res["video"]["real_fps"] == pytest.approx(240.0, rel=0.05)


def test_missing_input_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.run(str(tmp_path / "nope.mp4"), str(tmp_path / "o"))


def test_detect_strokes_finds_three_swings():
    fps = 50.0
    t = np.arange(0, 6, 1 / fps)
    sp = 0.3 + sum(8 * np.exp(-0.5 * ((t - c) / 0.06) ** 2) for c in (1.0, 2.5, 4.0))
    peaks = pipeline.detect_strokes(sp, fps)
    assert [round(t[p], 1) for p in peaks] == [1.0, 2.5, 4.0]


def test_detect_strokes_ignores_small_wobble():
    fps = 50.0
    t = np.arange(0, 4, 1 / fps)
    sp = 0.3 + 0.4 * np.sin(t * 7) ** 2
    assert pipeline.detect_strokes(sp, fps) == []


def _wrist_path(zs, fps=50):
    # facing the camera means forward = -z (nose side); wrist moves along z only
    return build_series([{"right_wrist": (-0.4, -0.2, z)} for z in zs], fps=fps)


def test_forward_score_positive_when_hand_moves_toward_facing_direction():
    s = _wrist_path(np.linspace(0.2, -0.4, 30))   # moving toward the camera = forward
    assert pipeline.forward_score(s, "right", 15) > 0.5


def test_forward_score_negative_for_recovery_motion():
    s = _wrist_path(np.linspace(-0.4, 0.2, 30))   # moving back toward the body
    assert pipeline.forward_score(s, "right", 15) < -0.5


def test_stroke_candidates_drop_backward_and_untracked_peaks():
    fps = 50
    n = 200
    zs = np.full(n, 0.0)
    # forward swing around frame 50 (z decreasing fast), recovery around frame 80 (z increasing fast)
    zs[40:61] = np.linspace(0.2, -0.4, 21)
    zs[61:70] = -0.4
    zs[70:91] = np.linspace(-0.4, 0.2, 21)
    zs[91:] = 0.2
    zs[:40] = 0.2
    s = build_series([{"right_wrist": (-0.4, -0.2, z)} for z in zs], fps=fps)
    s.present[150:] = False
    peaks, dropped = pipeline.stroke_candidates(s, "right", [50, 80, 160])
    assert peaks == [50]
    assert {d["reason"] for d in dropped} == {"moving back (backswing or recovery)", "player not tracked at the peak"}


def _front_view_forehand(fps=50):
    """Right-hander facing the camera: backswing to the playing side (-x) and down, then forward swing across
    the body (+x) and up, then recovery straight down."""
    pts = []
    pts += [(-0.30, -0.05)] * 10
    pts += [(-0.30 - 0.03 * k, -0.05 + 0.01 * k) for k in range(1, 16)]          # backswing: -x, down (y grows)
    x0, y0 = pts[-1]
    pts += [(x0 + 0.04 * k, y0 - 0.04 * k) for k in range(1, 11)]                # forward swing: +x, up
    x1, y1 = pts[-1]
    pts += [(x1, y1 + 0.03 * k) for k in range(1, 16)]                           # recovery: down
    pts += [pts[-1]] * 10
    return build_series([{"right_wrist": (x, y, -0.2)} for x, y in pts], fps=fps)


def test_swing_score_front_view_forehand():
    s = _front_view_forehand()
    sc = pipeline.swing_scores(s, "right")
    assert sc[17] < -0.2      # backswing
    assert sc[30] > 0.3       # forward swing
    assert sc[45] < -0.2      # recovery


def test_merge_close_peaks_joins_one_continuous_swing():
    fps = 25.0
    t = np.arange(0, 3, 1 / fps)
    # two bumps 0.32 s apart with no dip between them: one swing seen as two peaks
    sp = 0.3 + 4 * np.exp(-0.5 * ((t - 1.0) / 0.12) ** 2) + 4.5 * np.exp(-0.5 * ((t - 1.32) / 0.12) ** 2)
    peaks = [int(round(1.0 * fps)), int(round(1.32 * fps))]
    assert pipeline.merge_close_peaks(sp, peaks, fps) == [peaks[1]]


def test_merge_close_peaks_keeps_two_separate_swings():
    fps = 25.0
    t = np.arange(0, 3, 1 / fps)
    sp = 0.3 + 4 * np.exp(-0.5 * ((t - 1.0) / 0.04) ** 2) + 4.5 * np.exp(-0.5 * ((t - 1.4) / 0.04) ** 2)
    peaks = [int(round(1.0 * fps)), int(round(1.4 * fps))]
    assert pipeline.merge_close_peaks(sp, peaks, fps) == peaks


def test_small_hand_travel_is_not_a_stroke():
    from ttcoach.signals import Phases
    ph = Phases(ready=0, backswing_end=10, mid_forward=15, contact=20, finish=30, recovery=35)
    still = build_series([{}] * 40, fps=50)
    assert pipeline.too_small(still, "right", ph) is True
    moving = build_series([{}] * 20 + [{"right_wrist": (0.1, -0.5, -0.3)}] * 20, fps=50)
    assert pipeline.too_small(moving, "right", ph) is False


def test_clean_series_removes_tracking_jumps():
    from conftest import BASE
    poses = [dict() for _ in range(40)]
    # frame 20: the tracker jumps to a wrong spot two torso lengths away (cut / wrong person)
    poses[20] = {k: (x + 1.0, y, z) for k, (x, y, z) in BASE.items()}
    s = build_series(poses, fps=50)
    n = pipeline.clean_series(s)
    assert n >= 1
    assert not s.present[20]
    assert np.all(np.isnan(s.img[20]))
    assert s.present[10] and s.present[30]


def test_hand_mismatch_warning_when_other_wrist_swings():
    poses = []
    for i in range(60):
        x = 0.25 + 0.3 * np.sin(i / 3.0)  # left wrist swings a lot, right wrist still
        poses.append({"left_wrist": (x, -0.1, -0.2)})
    s = build_series(poses, fps=50)
    assert "HAND_MISMATCH" in {w["code"] for w in pipeline.series_warnings(s, "right")}
    assert "HAND_MISMATCH" not in {w["code"] for w in pipeline.series_warnings(s, "left")}


def test_multi_person_warning():
    s = build_series([{}] * 50, fps=50)
    s.n_people[:] = 2
    assert "MULTI_PERSON" in {w["code"] for w in pipeline.series_warnings(s, "right")}


def test_slowmo_suspected_from_long_forward_swings():
    assert pipeline.slowmo_suspected([900, 1100, 1000]) is True
    assert pipeline.slowmo_suspected([150, 200, 180]) is False
    assert pipeline.slowmo_suspected([]) is False


@pytest.mark.skipif(not vid("tt_evworo.webm"), reason="test videos not available")
def test_probe_uses_frame_timestamps_when_metadata_fps_is_bogus():
    info = video.probe(vid("tt_evworo.webm"))
    assert 10 <= info.file_fps <= 120
    assert "FPS_METADATA_BAD" in {w["code"] for w in info.warnings}


@pytest.mark.skipif(not vid("bloopers.ogv"), reason="test videos not available")
def test_probe_warns_when_decoding_stops_early():
    info = video.probe(vid("bloopers.ogv"))
    assert "DECODE_STOPPED_EARLY" in {w["code"] for w in info.warnings}
    assert info.frames < 600


def _pose_ready():
    try:
        from ttcoach import pose
        pose.ensure_model("full")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (vid("topspin_drive.ogv") and _pose_ready()), reason="needs test video + mediapipe model")
def test_forehand_clip_detects_forehand_strokes(tmp_path):
    # Commons "Topspin Drive.ogv": right-handed player, forehand topspin, filmed from across the table.
    # Ground truth from looking at the frames (file seconds): forward swings around these contact times.
    truth = [5.6, 7.4, 9.15, 11.45, 13.4, 15.5]
    res = pipeline.run(vid("topspin_drive.ogv"), str(tmp_path / "fh"), hand="right")
    cands = [s["t_contact_file_s"] for s in res["strokes"]]
    found = [t for t in truth if any(abs(t - c) <= 0.4 for c in cands)]
    # 25 fps slow demo strokes: auto candidates are approximate; Claude verifies them on the images
    assert len(found) >= 4, (cands, found)
    assert all(s["source"] == "auto" for s in res["strokes"])
    assert "LOW_FPS" in codes(res)  # the clip is 25 fps
    assert (tmp_path / "fh" / "stroke_01_sheet.jpg").exists()
    assert (tmp_path / "fh" / "stroke_01_zoom.jpg").exists()
    assert (tmp_path / "fh" / "stroke_01_strip.jpg").exists()
    assert (tmp_path / "fh" / "timeline_01.jpg").exists()


@pytest.mark.skipif(not (vid("topspin_drive.ogv") and _pose_ready()), reason="needs test video + mediapipe model")
def test_given_phases_are_measured_and_cached(tmp_path):
    out = tmp_path / "m"
    res = pipeline.run(vid("topspin_drive.ogv"), str(out), hand="right", strokes="5.28,5.44,5.60,5.72,6.00")
    assert res["task"] == "strokes" and len(res["strokes"]) == 1
    st = res["strokes"][0]
    assert st["phases"]["contact"]["t_file"] == pytest.approx(5.60, abs=0.03)
    assert (out / "measured_1" / "measured_01_sheet.jpg").exists()
    assert (out / "pose_cache.npz").exists()
    res2 = pipeline.run(vid("topspin_drive.ogv"), str(out), hand="right", at="5.60")
    assert res2["task"] == "at" and res2["at"][0]["pose_found"]
    assert (out / "at_5.60" / "at_01.jpg").exists()
    res3 = pipeline.run(vid("topspin_drive.ogv"), str(out), hand="right", strokes="5.28,5.44,5.60,5.72,6.00",
                        side="forehand")
    assert res3["out_dir"].endswith("measured_2") and res3["strokes"][0]["side_guess"] == "forehand"


def test_single_frame_landmark_dropout_is_repaired():
    s = build_series([{}] * 40, fps=50)
    j = 16  # right_wrist
    s.vis[20, j] = 0.2
    s.img[20, j] = s.img[20, j] + 30.0  # the blurred frame's guess is off
    before = (s.img[19, j] + s.img[21, j]) / 2
    pipeline.clean_series(s)
    assert s.vis[20, j] >= 0.5
    assert np.allclose(s.img[20, j], before)


def test_long_dropout_is_not_repaired():
    s = build_series([{}] * 40, fps=50)
    j = 26  # right_knee hidden behind the table for many frames
    s.vis[10:30, j] = 0.2
    pipeline.clean_series(s)
    assert np.all(s.vis[12:28, j] < 0.5)


@pytest.mark.skipif(not vid("tt_evworo.webm"), reason="test videos not available")
def test_bogus_frame_count_is_not_reported_as_early_stop():
    # metadata says 30000 fps and 685170 frames: both are garbage, the file itself decodes completely
    info = video.probe(vid("tt_evworo.webm"))
    assert "DECODE_STOPPED_EARLY" not in {w["code"] for w in info.warnings}

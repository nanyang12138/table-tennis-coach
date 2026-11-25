import numpy as np
import pytest

from conftest import build_series
from ttcoach import pipeline


def test_parse_strokes_arg_two_strokes():
    got = pipeline.parse_strokes("1.0,1.2,1.4,1.6,1.9; 3.0,3.1,3.3,3.5,3.8")
    assert got == [(1.0, 1.2, 1.4, 1.6, 1.9), (3.0, 3.1, 3.3, 3.5, 3.8)]


def test_parse_strokes_rejects_wrong_count_or_order():
    with pytest.raises(ValueError):
        pipeline.parse_strokes("1.0,1.2,1.4")
    with pytest.raises(ValueError):
        pipeline.parse_strokes("1.0,1.4,1.2,1.6,1.9")


def test_parse_times():
    assert pipeline.parse_times("0.5, 1.25,2") == [0.5, 1.25, 2.0]
    with pytest.raises(ValueError):
        pipeline.parse_times("a,b")


def test_measure_strokes_uses_given_phase_times():
    s = build_series([{}] * 100, fps=50)  # t = 0, 0.02, ... 1.98
    strokes = pipeline.measure_strokes(s, "right", [(0.2, 0.4, 0.5, 0.7, 1.0)])
    assert len(strokes) == 1
    ph = strokes[0]["phases"]
    assert [ph[k]["i"] for k in ("ready", "backswing_end", "contact", "finish", "recovery")] == [10, 20, 25, 35, 50]
    assert ph["mid_forward"]["i"] == 22
    assert strokes[0]["source"] == "given"
    assert strokes[0]["metrics"]["forward_ms"] == pytest.approx(100)


def test_measure_strokes_rejects_times_outside_clip():
    s = build_series([{}] * 50, fps=50)
    with pytest.raises(ValueError):
        pipeline.measure_strokes(s, "right", [(0.2, 0.4, 0.5, 0.7, 5.0)])


def test_timeline_times_cover_range():
    ts = pipeline.timeline_times(1.0, 2.0, 0.25)
    assert ts == pytest.approx([1.0, 1.25, 1.5, 1.75, 2.0])


def test_auto_timeline_step_keeps_pages_bounded():
    step = pipeline.auto_timeline_step(duration=30.0, per_page=20, max_pages=6)
    assert 30.0 / step <= 20 * 6 + 1
    assert step >= 0.1


def test_pose_cache_roundtrip(tmp_path):
    s = build_series([{}] * 10, fps=50)
    f = tmp_path / "pose.npz"
    pipeline.save_series(s, f, key="abc")
    back = pipeline.load_series(f, key="abc")
    assert back is not None
    assert np.allclose(back.img, s.img, equal_nan=True) and back.fps == s.fps
    assert pipeline.load_series(f, key="other") is None

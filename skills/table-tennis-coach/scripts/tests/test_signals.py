import math

import numpy as np
import pytest

from ttcoach import signals as s


def bump(n, center, width, height):
    x = np.arange(n)
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


def test_fill_gaps_interpolates_short_runs_only():
    x = np.array([0.0, np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, 7.0])
    y = s.fill_gaps(x, max_gap=2)
    assert y[1] == pytest.approx(1.0)
    assert all(math.isnan(v) for v in y[3:7])


def test_fill_gaps_works_per_column_on_2d():
    x = np.array([[0.0, 10.0], [np.nan, np.nan], [2.0, 30.0]])
    y = s.fill_gaps(x, max_gap=1)
    assert y[1].tolist() == pytest.approx([1.0, 20.0])


def test_smooth_keeps_length_and_constants():
    x = np.full(9, 3.0)
    y = s.smooth(x, 5)
    assert y.shape == x.shape
    assert np.allclose(y, 3.0)


def test_smooth_ignores_nan_neighbours():
    x = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
    assert np.allclose(s.smooth(x, 3), 1.0)


def test_find_peaks_three_bumps():
    x = bump(300, 50, 5, 10) + bump(300, 150, 5, 8) + bump(300, 250, 5, 12)
    assert s.find_peaks(x, min_height=3, min_dist=20) == [50, 150, 250]


def test_find_peaks_min_distance_keeps_higher_peak():
    x = bump(200, 100, 3, 10) + bump(200, 108, 3, 6)
    assert s.find_peaks(x, min_height=3, min_dist=20) == [100]


def test_find_peaks_ignores_low_peaks():
    x = bump(200, 50, 5, 10) + bump(200, 150, 5, 2)
    assert s.find_peaks(x, min_height=3, min_dist=20) == [50]


def stroke_speed(fps=100):
    """quiet ready -> backswing bump -> reversal near zero -> big forward peak -> decay -> quiet."""
    n = 2 * fps
    x = 0.05 + bump(n, 0.55 * fps, 0.08 * fps, 3.0) + bump(n, 1.0 * fps, 0.05 * fps, 10.0)
    return x


def test_segment_orders_phases_and_finds_reversal():
    fps = 100
    sp = stroke_speed(fps)
    ph = s.segment(sp, peak=100, fps=fps)
    assert ph.ready < ph.backswing_end < ph.mid_forward < ph.contact < ph.finish < ph.recovery
    assert ph.contact == 100
    # reversal (speed minimum between the two bumps) is around 0.75-0.85 s
    assert 70 <= ph.backswing_end <= 88
    # ready happens before the backswing bump starts rising
    assert ph.ready <= 45


def test_segment_respects_neighbour_limits():
    fps = 100
    sp = stroke_speed(fps)
    ph = s.segment(sp, peak=100, fps=fps, lo=80, hi=120)
    assert ph.ready >= 80 and ph.recovery <= 119


def test_segment_on_coarse_sampling_is_non_decreasing():
    fps = 25
    sp = stroke_speed(fps)
    ph = s.segment(sp, peak=25, fps=fps)
    seq = [ph.ready, ph.backswing_end, ph.mid_forward, ph.contact, ph.finish, ph.recovery]
    assert seq == sorted(seq)
    assert ph.contact == 25

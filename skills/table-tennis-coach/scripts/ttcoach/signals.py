"""1-D signal helpers: gap filling, smoothing, peak picking and stroke phase segmentation."""
from dataclasses import dataclass, asdict

import numpy as np


def fill_gaps(x, max_gap):
    """Linearly interpolate interior NaN runs no longer than max_gap (per column for 2-D input)."""
    a = np.array(x, dtype=float, copy=True)
    if a.ndim == 1:
        return _fill_1d(a, max_gap)
    flat = a.reshape(a.shape[0], -1)
    for j in range(flat.shape[1]):
        flat[:, j] = _fill_1d(flat[:, j], max_gap)
    return flat.reshape(a.shape)


def _fill_1d(a, max_gap):
    n = len(a)
    i = 0
    while i < n:
        if np.isnan(a[i]):
            j = i
            while j < n and np.isnan(a[j]):
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                a[i:j] = np.interp(np.arange(i, j), [i - 1, j], [a[i - 1], a[j]])
            i = j
        else:
            i += 1
    return a


def smooth(x, win):
    """Centred moving average that ignores NaN; keeps length."""
    a = np.asarray(x, dtype=float)
    win = max(1, int(win))
    if win == 1:
        return a.copy()
    half = win // 2
    ok = np.isfinite(a)
    vals = np.where(ok, a, 0.0)
    k = np.ones(2 * half + 1)
    num = np.convolve(vals, k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def find_peaks(x, min_height, min_dist):
    """Indices of local maxima >= min_height, keeping the higher peak within min_dist samples."""
    a = np.nan_to_num(np.asarray(x, dtype=float), nan=-np.inf)
    n = len(a)
    cand = []
    for i in range(n):
        left = a[i - 1] if i > 0 else -np.inf
        right = a[i + 1] if i < n - 1 else -np.inf
        if a[i] >= min_height and a[i] >= left and a[i] > right:
            cand.append(i)
    cand.sort(key=lambda i: -a[i])
    kept = []
    for i in cand:
        if all(abs(i - k) >= min_dist for k in kept):
            kept.append(i)
    return sorted(kept)


@dataclass
class Phases:
    ready: int
    backswing_end: int
    mid_forward: int
    contact: int
    finish: int
    recovery: int

    def as_dict(self):
        return asdict(self)


def segment(speed, peak, fps, lo=0, hi=None):
    """Split one stroke around its hand-speed peak into six phase indices.

    speed: smoothed hand speed (any unit); peak: index of the stroke's speed peak (taken as contact);
    lo/hi: neighbouring-stroke limits (indices) the phases must stay inside.
    """
    sp = np.nan_to_num(np.asarray(speed, dtype=float), nan=0.0)
    n = len(sp)
    hi = n if hi is None else min(int(hi), n)
    lo = max(0, int(lo))
    peak = int(peak)
    pk = sp[peak]

    # backswing end: first local minimum walking back from the peak that is clearly below the peak
    limit_b = max(lo, peak - int(round(0.6 * fps)))
    b = None
    i = peak
    while i > limit_b:
        if sp[i - 1] > sp[i] and sp[i] <= 0.5 * pk:
            b = i
            break
        i -= 1
    if b is None:
        b = limit_b + int(np.argmin(sp[limit_b:peak])) if peak > limit_b else peak

    # ready: before the backswing motion rises (15% of the backswing speed bump above its floor)
    limit_r = max(lo, b - int(round(1.0 * fps)))
    if b > limit_r:
        seg = sp[limit_r:b + 1]
        top = limit_r + int(np.argmax(seg))
        if top < b:
            base = float(np.min(sp[limit_r:top + 1]))
            thr = base + 0.15 * (sp[top] - base)
            r = top
            while r > limit_r and sp[r] > thr:
                r -= 1
        else:
            r = max(limit_r, b - int(round(0.3 * fps)))
    else:
        r = b

    # finish: hand slows down after contact (local minimum, or below 15% of the peak above the floor)
    limit_f = min(hi - 1, peak + int(round(0.8 * fps)))
    f = None
    if limit_f > peak:
        base_after = float(np.min(sp[peak:limit_f + 1]))
        thr_f = base_after + 0.15 * (pk - base_after)
        for j in range(peak + 1, limit_f + 1):
            local_min = j < n - 1 and sp[j + 1] > sp[j] and sp[j] <= 0.35 * pk
            if local_min or sp[j] <= thr_f:
                f = j
                break
        if f is None:
            f = peak + 1 + int(np.argmin(sp[peak + 1:limit_f + 1]))
    else:
        f = peak

    rc = min(hi - 1, f + int(round(0.3 * fps)))
    rc = max(rc, f)
    m = (b + peak) // 2
    r = min(r, b)
    return Phases(ready=int(r), backswing_end=int(b), mid_forward=int(m), contact=peak, finish=int(f), recovery=int(rc))

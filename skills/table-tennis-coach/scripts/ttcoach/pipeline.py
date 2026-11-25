"""Video or photo -> tracked pose -> strokes -> metrics -> images + analysis.json + summary.md."""
import os
from pathlib import Path

import cv2
import numpy as np

from . import __version__, geometry, metrics, render, report, signals, video
from . import pose as pose_mod
from .landmarks import IDX, side_names
from .video import warn

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PHASES = ["ready", "backswing_end", "mid_forward", "contact", "finish", "recovery"]
PHASE_LABEL = {"ready": "READY", "backswing_end": "BACKSWING END", "mid_forward": "FORWARD SWING",
               "contact": "CONTACT (est.)", "finish": "FINISH", "recovery": "RECOVERY"}
MIN_STROKE_SPEED = 1.5   # torso lengths per second, hand relative to hips
MAX_FILE_SECONDS = 90.0  # analyse at most this much file time unless --end is given
MAX_STROKES = 60
TILE_W = 480


def _odd(n):
    n = max(1, int(n))
    return n if n % 2 else n + 1


def detect_strokes(speed, fps, min_gap_s=0.35, floor=MIN_STROKE_SPEED):
    """Peak indices of clear hand-speed bursts (one per stroke)."""
    sp = np.asarray(speed, float)
    fin = sp[np.isfinite(sp)]
    if len(fin) < 3:
        return []
    med, p95 = float(np.median(fin)), float(np.percentile(fin, 95))
    thr = max(floor, med + 0.35 * (p95 - med))
    peaks = signals.find_peaks(sp, thr, max(1, int(round(min_gap_s * fps))))
    win = int(round(0.8 * fps))
    keep = []
    for p in peaks:  # a much weaker burst next to a strong one is that stroke's backswing, not a stroke
        stronger = [sp[q] for q in peaks if q != p and abs(q - p) <= win and sp[q] > sp[p]]
        if stronger and sp[p] < 0.5 * max(stronger):
            continue
        keep.append(p)
    return keep


def detect_motion_strokes(energy, fps, min_gap_s=0.4):
    """Frames-only fallback: bursts of whole-image motion."""
    e = np.asarray(energy, float)
    fin = e[np.isfinite(e)]
    if len(fin) < 3:
        return []
    med, p95 = float(np.median(fin)), float(np.percentile(fin, 95))
    if p95 < 1.5 * med + 1.0:
        return []
    return signals.find_peaks(e, med + 0.5 * (p95 - med), max(1, int(round(min_gap_s * fps))))


FORWARD_MIN = -0.2  # hand moving clearly backwards at the speed peak -> backswing or recovery, not a stroke


def facing_median(s):
    """Median horizontal facing direction of the player over the clip (players mostly face the table)."""
    fs = np.array([geometry.facing_vector(s.world[i, IDX["left_shoulder"]], s.world[i, IDX["right_shoulder"]],
                                          s.world[i, IDX["nose"]]) for i in range(len(s))])
    ok = np.isfinite(fs[:, 0]) & np.isfinite(fs[:, 2]) if len(fs) else np.array([], bool)
    if not ok.any():
        return None
    v = np.array([np.median(fs[ok, 0]), 0.0, np.median(fs[ok, 2])])
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _forward3(s, hand):
    """3-D (MediaPipe world) direction of the playing hand relative to the shoulders, smoothed ~0.12 s.

    Returns (forward_n, up_n): velocity components along the clip's median facing direction and up,
    each divided by the speed (so -1..1). Depth is noisy frame to frame; over a swing its sign is usable.
    """
    p, _ = side_names(hand)
    n = len(s)
    f = facing_median(s)
    if f is None or n < 3:
        return np.zeros(n), np.zeros(n)
    w = s.world[:, IDX[p + "_wrist"]] - (s.world[:, IDX["left_shoulder"]] + s.world[:, IDX["right_shoulder"]]) / 2.0
    w = metrics.median3(w)
    win = _odd(round(0.12 * s.fps))
    for k in range(3):
        w[:, k] = signals.smooth(signals.fill_gaps(w[:, k], max(1, int(round(0.1 * s.fps)))), win)
    v = np.gradient(w, axis=0) * s.fps
    mag = np.linalg.norm(v, axis=1) + 1e-9
    fwd = (v[:, 0] * f[0] + v[:, 2] * f[2]) / mag
    up = -v[:, 1] / mag
    return np.nan_to_num(fwd, nan=0.0), np.nan_to_num(up, nan=0.0)


def forward_scores(s, hand):
    """Per frame: forward + 0.5*up direction of the hand in 3-D (about -1.5..1.5)."""
    fwd, up = _forward3(s, hand)
    return np.clip(fwd + 0.5 * up, -1.5, 1.5)


def forward_score(s, hand, i):
    return float(forward_scores(s, hand)[i])


def _lateral_world(s, hand):
    """Wrist offset toward the playing side of the shoulder centre, metres, per frame (NaN when unknown)."""
    p, o = side_names(hand)
    w = s.world
    hipmid = (w[:, IDX["left_shoulder"]] + w[:, IDX["right_shoulder"]]) / 2.0
    u = (w[:, IDX[p + "_shoulder"]] - w[:, IDX[o + "_shoulder"]])[:, [0, 2]]
    nu = np.linalg.norm(u, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = u / nu
    rel = (w[:, IDX[p + "_wrist"]] - hipmid)[:, [0, 2]]
    return np.sum(rel * u, axis=1)


def swing_scores(s, hand):
    """Per frame: does the hand move like a FORWARD swing (positive) or like a backswing/recovery (negative)?

    View-aware 2-D cues plus a weak 3-D cue:
    - front view (shoulder line visible): a forehand swings across the body toward the non-playing side,
      a backhand toward the playing side;
    - side view (nose clearly ahead of the shoulders): forward = the way the player faces in the image;
    - up counts half (topspin strokes rise; pushes and chops do not, the 3-D forward cue covers them).
    """
    p, o = side_names(hand)
    n = len(s)
    T = metrics.torso_px(s)
    if n < 3 or not np.isfinite(T) or T <= 0:
        return np.zeros(n)
    rel = metrics.hand_rel_shoulders_px(s, hand) / T
    win = _odd(round(0.06 * s.fps))
    for k in range(2):
        rel[:, k] = signals.smooth(signals.fill_gaps(rel[:, k], max(1, int(round(0.1 * s.fps)))), win)
    v = metrics.central_velocity(rel, s.fps)
    mag = np.linalg.norm(v, axis=1) + 1e-9
    L = s.img[:, IDX[p + "_shoulder"]] - s.img[:, IDX[o + "_shoulder"]]
    nL = np.linalg.norm(L, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Lu = L / nL[:, None]
    rho_l = np.clip(np.nan_to_num(nL / T / 0.6, nan=0.0), 0.0, 1.0)
    shm = (s.img[:, IDX["left_shoulder"]] + s.img[:, IDX["right_shoulder"]]) / 2.0
    nx = np.nan_to_num((s.img[:, IDX["nose"], 0] - shm[:, 0]) / T, nan=0.0)
    rho_f = np.clip(np.abs(nx) / 0.3, 0.0, 1.0) * (1.0 - rho_l)
    lat3 = _lateral_world(s, hand)
    side = np.where(lat3 > metrics.FOREHAND_LATERAL_M, -1.0, np.where(lat3 < metrics.BACKHAND_LATERAL_M, 1.0, 0.0))
    side[~np.isfinite(lat3)] = 0.0
    lat_term = rho_l * side * np.nan_to_num(np.sum(v * Lu, axis=1), nan=0.0)
    fwd2_term = rho_f * np.sign(nx) * np.nan_to_num(v[:, 0], nan=0.0)
    up = -np.nan_to_num(v[:, 1], nan=0.0)
    sc2 = (lat_term + fwd2_term + 0.5 * up) / mag
    fwd3, _ = _forward3(s, hand)
    return np.clip(np.nan_to_num(sc2, nan=0.0) + 0.4 * fwd3, -2.0, 2.0)


def repair_dropouts(s, vis_min=0.5):
    """A joint that is clearly visible in the frames before and after, but not in one frame (motion blur),
    gets the average of its neighbours. Longer gaps stay as they are. Returns how many points were repaired."""
    n, fixed = len(s), 0
    for j in range(s.vis.shape[1]):
        v = s.vis[:, j]
        for i in range(1, n - 1):
            if v[i] < vis_min and v[i - 1] >= vis_min and v[i + 1] >= vis_min \
                    and s.present[i - 1] and s.present[i] and s.present[i + 1]:
                s.img[i, j] = (s.img[i - 1, j] + s.img[i + 1, j]) / 2.0
                s.world[i, j] = (s.world[i - 1, j] + s.world[i + 1, j]) / 2.0
                s.vis[i, j] = min(v[i - 1], v[i + 1])
                fixed += 1
    return fixed


def left_right_swaps(s, min_width_torso=0.3):
    """Frames whose left/right shoulder order is opposite to the clip's usual order (the pose model
    mixed up left and right). Only judged when the shoulders are clearly apart in the image."""
    L = s.img[:, IDX["left_shoulder"]] - s.img[:, IDX["right_shoulder"]]
    T = metrics.torso_px(s)
    out = np.zeros(len(s), bool)
    ok = s.present & np.all(np.isfinite(L), axis=1)
    if ok.sum() < 5 or not np.isfinite(T) or T <= 0:
        return out
    Lm = np.median(L[ok], axis=0)
    if np.linalg.norm(Lm) < min_width_torso * T:
        return out
    wide = np.linalg.norm(L, axis=1) > min_width_torso * T
    out = ok & wide & (L @ Lm < 0)
    return out


def clean_series(s, max_jump_torso=0.5, max_scale_change=0.3):
    """Repair one-frame joint dropouts, then blank frames where tracking jumps (cut, switched person).
    Returns how many frames were blanked."""
    repair_dropouts(s)
    swapped = left_right_swaps(s)
    s.present[swapped] = False
    s.img[swapped] = np.nan
    s.world[swapped] = np.nan
    s.vis[swapped] = 0.0
    sh = (s.img[:, IDX["left_shoulder"]] + s.img[:, IDX["right_shoulder"]]) / 2.0
    hp = (s.img[:, IDX["left_hip"]] + s.img[:, IDX["right_hip"]]) / 2.0
    tl = np.linalg.norm(sh - hp, axis=1)
    T = metrics.torso_px(s)
    if not np.isfinite(T) or T <= 0:
        return 0
    bad = np.zeros(len(s), bool)
    for i in range(1, len(s) - 1):
        if not s.present[i]:
            continue
        nb = [j for j in (i - 1, i + 1) if s.present[j]]
        if not nb:
            continue
        # a frame that is far from BOTH neighbours (or from its only neighbour) is a jump
        far = all(np.linalg.norm(sh[i] - sh[j]) > max_jump_torso * T or
                  abs(tl[i] - tl[j]) > max_scale_change * T for j in nb)
        if far:
            bad[i] = True
    s.present[bad] = False
    s.img[bad] = np.nan
    s.world[bad] = np.nan
    s.vis[bad] = 0.0
    return int(bad.sum())


def stroke_candidates(s, hand, peaks):
    """Keep speed peaks that are real forward swings of a tracked player; report the rest."""
    sc = swing_scores(s, hand)
    k2 = max(1, int(round(0.1 * s.fps)))
    keep, dropped = [], []
    for pk in peaks:
        lo, hi = max(0, pk - k2), min(len(s), pk + k2 + 1)
        if not s.present[pk] or np.mean(s.present[lo:hi]) < 0.5:
            dropped.append({"i": int(pk), "t": round(float(s.t[pk]), 3), "reason": "player not tracked at the peak"})
        elif sc[pk] < FORWARD_MIN:
            dropped.append({"i": int(pk), "t": round(float(s.t[pk]), 3), "reason": "moving back (backswing or recovery)"})
        else:
            keep.append(int(pk))
    return keep, dropped


def merge_close_peaks(speed, peaks, fps, max_gap_s=0.45, dip_ratio=0.5):
    """Join peaks that belong to one continuous fast motion (speed never dips in between)."""
    sp = np.nan_to_num(np.asarray(speed, float), nan=0.0)
    out = []
    for p in sorted(peaks):
        if out and (p - out[-1]) <= max_gap_s * fps:
            q = out[-1]
            if sp[q:p + 1].min() > dip_ratio * min(sp[q], sp[p]):
                if sp[p] > sp[q]:
                    out[-1] = p
                continue
        out.append(p)
    return out


MIN_TRAVEL_TORSO = 0.35


def too_small(s, hand, ph, min_travel=MIN_TRAVEL_TORSO):
    """True when the hand barely moves between backswing end and finish (jitter, not a stroke)."""
    rel = metrics.hand_rel_shoulders_px(s, hand)
    T = metrics.torso_px(s)
    d = np.linalg.norm(rel[ph.finish] - rel[ph.backswing_end])
    if not (np.isfinite(d) and np.isfinite(T) and T > 0):
        return False
    return bool(d / T < min_travel)


def _wrist_activity(s, side):
    rel = s.img[:, IDX[side + "_wrist"]] - (s.img[:, IDX["left_hip"]] + s.img[:, IDX["right_hip"]]) / 2.0
    if len(s) < 3:
        return 0.0
    sp = np.linalg.norm(np.gradient(rel, axis=0), axis=1)
    sp = sp[np.isfinite(sp)]
    return float(np.percentile(sp, 90)) if len(sp) else 0.0


def series_warnings(s, hand):
    out = []
    mine, other = side_names(hand)
    a_p, a_o = _wrist_activity(s, mine), _wrist_activity(s, other)
    if a_o > 1.3 * a_p and a_o > 0:
        out.append(warn("HAND_MISMATCH", "另一只手的动作比%s手更大：可能是视频被镜像（手机前置摄像头自拍）、球员其实是%s手持拍，"
                                         "或者 --hand 填错了。请看图确认持拍手；如有需要用 --hand 重新分析。" % (
                                             "右" if hand == "right" else "左", "左" if hand == "right" else "右")))
    if len(s) and np.mean(s.n_people >= 2) > 0.2:
        out.append(warn("MULTI_PERSON", "画面里经常不止一个人（对手、教练、路人）。脚本默认跟踪画面里最大的人；"
                                        "如果跟错了人，用 --player left/right/center 指定。"))
    if len(s) and np.mean(s.present) < 0.6:
        out.append(warn("LOW_COVERAGE", "有 %.0f%% 的帧没认出球员（被遮挡、出画、太小或太暗），数字只能参考。" % (100 * (1 - np.mean(s.present)))))
    return out


def slowmo_suspected(forward_ms):
    vals = [v for v in forward_ms if v is not None]
    return bool(vals) and float(np.median(vals)) > 600.0


# ----------------------------------------------------------------------------------------------
# Claude-driven measurement helpers: times are FILE times in seconds (what a video player shows
# and what our image labels print as t=...). Durations inside metrics use real time.
# ----------------------------------------------------------------------------------------------

def parse_times(arg):
    """'0.5, 1.25,2' -> [0.5, 1.25, 2.0]"""
    if isinstance(arg, (list, tuple)):
        return [float(x) for x in arg]
    try:
        return [float(x) for x in str(arg).replace(" ", "").split(",") if x != ""]
    except ValueError:
        raise ValueError("times must be numbers separated by commas, e.g. 1.20,1.48")


def parse_strokes(arg):
    """'r,b,c,f,rc; r,b,c,f,rc' -> list of 5-tuples (ready, backswing_end, contact, finish, recovery)."""
    if isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], (list, tuple)):
        groups = [tuple(float(x) for x in g) for g in arg]
    else:
        groups = [tuple(parse_times(g)) for g in str(arg).split(";") if g.strip()]
    for g in groups:
        if len(g) != 5:
            raise ValueError("each stroke needs 5 times: ready,backswing_end,contact,finish,recovery (got %d)" % len(g))
        if any(b <= a for a, b in zip(g, g[1:])):
            raise ValueError("stroke times must increase: ready < backswing_end < contact < finish < recovery: %s" % (g,))
    return groups


def _nearest_index(s, t_real):
    if len(s) == 0:
        raise ValueError("no analysed frames")
    tol = 1.5 / s.fps
    if t_real < s.t[0] - tol or t_real > s.t[-1] + tol:
        raise ValueError("time %.3fs is outside the analysed part of the video (%.3f-%.3fs)" % (t_real, s.t[0], s.t[-1]))
    return int(np.argmin(np.abs(s.t - t_real)))


def _add_file_times(m, slowmo):
    sw = m.get("shoulder_width_min")
    if isinstance(sw, dict) and sw.get("t") is not None:
        sw["t_file"] = round(sw["t"] * slowmo, 3)
    return m


def _phases_dict(s, ph, slowmo):
    return {name: {"i": int(idx), "frame": int(s.frame_idx[idx]), "t": round(float(s.t[idx]), 3),
                   "t_file": round(float(s.t[idx]) * slowmo, 3)} for name, idx in ph.as_dict().items()}


def measure_strokes(s, hand, phase_times, slowmo=1.0, side=None):
    """Metrics for strokes whose phase times (file seconds) were chosen by looking at the frames.
    side ('forehand'/'backhand'), when given, replaces the script's guess for every stroke."""
    if side not in (None, "forehand", "backhand"):
        raise ValueError("side must be forehand or backhand")
    out = []
    for k, g in enumerate(parse_strokes(phase_times)):
        r, b, c, f, rc = [_nearest_index(s, t / slowmo) for t in g]
        ph = signals.Phases(ready=r, backswing_end=b, mid_forward=(b + c) // 2, contact=c, finish=f, recovery=rc)
        m = _add_file_times(metrics.stroke_metrics(s, ph, hand), slowmo)
        if side:
            m["side_guess"] = side
        out.append({"index": k + 1, "source": "given", "t_contact_s": round(float(s.t[c]), 3),
                    "t_contact_file_s": round(float(s.t[c]) * slowmo, 3), "side_guess": m["side_guess"],
                    "side_given": bool(side), "phases": _phases_dict(s, ph, slowmo), "metrics": m})
    return out


def timeline_frames(t0, t1, step, fps):
    """Source frame indices from t0 to t1 (file seconds) every `step` seconds, snapped to the frame grid."""
    first = int(np.ceil(t0 * fps - 1e-6))
    last = int(np.floor(t1 * fps + 1e-6))
    k = max(1, int(round(step * fps)))
    return list(range(first, last + 1, k))


def task_dir(out, task, timeline=None, at=None):
    """Folder for one run: auto -> out; each --timeline / --at / --strokes run gets its own sub-folder."""
    out = Path(out)
    if task == "timeline":
        return out / ("timeline_%.2f-%.2f" % (float(timeline[0]), float(timeline[1])))
    if task == "at":
        ts = ["%.2f" % t for t in at[:6]]
        return out / ("at_" + "_".join(ts) + ("_etc" if len(at) > 6 else ""))
    if task == "strokes":
        k = 1
        while (out / ("measured_%d" % k)).exists():
            k += 1
        return out / ("measured_%d" % k)
    return out


def timeline_times(t0, t1, step):
    n = int(np.floor((t1 - t0) / step + 1e-9)) + 1
    return [round(t0 + k * step, 6) for k in range(max(n, 1))]


def auto_timeline_step(duration, per_page=20, max_pages=6):
    raw = duration / float(per_page * max_pages)
    return max(0.1, float(np.ceil(raw / 0.05 - 1e-9) * 0.05))


def save_series(s, path, key):
    np.savez_compressed(str(path), key=np.array(key), t=s.t, frame_idx=s.frame_idx, img=s.img, vis=s.vis,
                        world=s.world, present=s.present, n_people=s.n_people,
                        meta=np.array([s.fps, s.width, s.height], float))


def load_series(path, key):
    from .series import PoseSeries
    try:
        d = np.load(str(path), allow_pickle=False)
    except (OSError, ValueError):
        return None
    if str(d["key"]) != key:
        return None
    fps, w, h = d["meta"]
    return PoseSeries(t=d["t"], frame_idx=d["frame_idx"], img=d["img"], vis=d["vis"], world=d["world"],
                      present=d["present"].astype(bool), n_people=d["n_people"], fps=float(fps), width=int(w), height=int(h))


def _phase_text(name, m):
    e = m.get("elbow_angle_deg") or {}
    k = (m.get("knee_angle_deg") or {}).get("playing") or {}

    def f(v, fmt="%.0f"):
        return "-" if v is None else fmt % v
    if name == "ready":
        return ["knee(play) %s" % f(k.get("ready"))]
    if name == "backswing_end":
        return ["elbow %s  knee(play) %s" % (f(e.get("backswing_end")), f(k.get("backswing_end")))]
    if name == "contact":
        return ["elbow %s  height %s" % (f(e.get("contact")), f(m.get("contact_height_level"), "%.2f"))]
    if name == "finish":
        return ["elbow %s  height %s" % (f(e.get("finish")), f(m.get("finish_height_level"), "%.2f"))]
    return []


def _tlabel(t_real, slowmo):
    """Image label: file time (as in any video player) and, for slow-motion files, real time."""
    if abs(slowmo - 1.0) < 1e-6:
        return "t=%.3fs" % t_real
    return "t=%.3fs (real %.3fs)" % (t_real * slowmo, t_real)


class _Frames:
    """Nearest analysed pose for any source frame index."""

    def __init__(self, s):
        self.s = s
        self.sorted = np.array(sorted(int(f) for f in s.frame_idx)) if s is not None and len(s) else np.array([], int)
        self.map = {int(f): j for j, f in enumerate(s.frame_idx)} if s is not None else {}

    def nearest(self, frame):
        if not len(self.sorted):
            return None
        j = int(np.argmin(np.abs(self.sorted - frame)))
        return self.map[int(self.sorted[j])]


def _render_timeline(path, frames_to_draw, s, s_a, hand, slowmo, file_fps, out_dir, prefix, per_page=20, cols=4, tile_w=330):
    """Pages of frames (file indices) with skeleton and time labels; returns written file names."""
    nf = _Frames(s)
    tiles = {}
    for fidx, fr in video.iter_frames(path, keep=set(frames_to_draw)):
        t = render.resize_w(fr, tile_w)
        i = nf.nearest(fidx)
        if i is not None and s.present[i]:
            render.draw_pose(t, s.img[i], s.vis[i], hand, scale=(tile_w / float(fr.shape[1])) / s_a, thickness=1)
        render.put_lines(t, [_tlabel(fidx / file_fps / slowmo, slowmo), "frame %d" % fidx], scale=0.42)
        tiles[fidx] = t
    order = [f for f in sorted(frames_to_draw) if f in tiles]
    names = []
    for pnum, k in enumerate(range(0, len(order), per_page), 1):
        g = render.grid([tiles[f] for f in order[k:k + per_page]], cols)
        name = "%s_%02d.jpg" % (prefix, pnum)
        cv2.imwrite(str(Path(out_dir) / name), g, [cv2.IMWRITE_JPEG_QUALITY, 82])
        names.append(name)
    return names


def _render_strokes(strokes, s, path, out, hand, s_a, vinfo, prefix="stroke"):
    """Per stroke: 6-phase sheet, 10-frame strip, hand zoom around contact, raw contact frame."""
    if not strokes:
        return
    slowmo, file_fps, stride = vinfo["slowmo_factor"], vinfo["file_fps"], vinfo["stride"]
    zoom_step = max(1, stride // 2)
    T_orig = metrics.torso_px(s) / s_a
    nf = _Frames(s)
    needed = {}
    for st in strokes:
        k = st["index"]
        for name in PHASES:
            needed.setdefault(st["phases"][name]["frame"], []).append((k, "tile", name))
        c = st["phases"]["contact"]["frame"]
        for off in (-2, -1, 0, 1, 2):
            needed.setdefault(c + off * zoom_step, []).append((k, "zoom", off))
        needed.setdefault(c, []).append((k, "raw", 0))
        a = st["phases"]["ready"]["frame"] - int(round(0.1 * file_fps * slowmo))
        z = st["phases"]["recovery"]["frame"] + int(round(0.1 * file_fps * slowmo))
        for j, fidx in enumerate(np.linspace(max(0, a), max(a + 1, z), 10).astype(int)):
            needed.setdefault(int(fidx), []).append((k, "strip", j))
    tiles, zooms, strips = {}, {}, {}
    by_index = {st["index"]: st for st in strokes}
    for fidx, fr in video.iter_frames(path, keep=set(needed)):
        for k, role, what in needed.get(fidx, []):
            st = by_index[k]
            i = nf.nearest(fidx)
            if role in ("tile", "strip"):
                w = TILE_W if role == "tile" else 300
                t = render.resize_w(fr, w)
                if i is not None and s.present[i]:
                    render.draw_pose(t, s.img[i], s.vis[i], hand, scale=(w / float(fr.shape[1])) / s_a,
                                     thickness=2 if role == "tile" else 1)
                if role == "tile":
                    render.put_lines(t, ["%s  %s" % (PHASE_LABEL[what], _tlabel(st["phases"][what]["t"], slowmo))]
                                     + _phase_text(what, st["metrics"]))
                    tiles[(k, what)] = t
                else:
                    render.put_lines(t, [_tlabel(fidx / file_fps / slowmo, slowmo)], scale=0.42)
                    strips[(k, what)] = t
            elif role == "zoom":
                if i is None or not s.present[i]:
                    continue
                box = render.hand_box(s.img[i] / s_a, s.vis[i], hand, T_orig, fr.shape)
                if box is not None:
                    zz = render.crop(fr, box, 300)
                    dt = (fidx - st["phases"]["contact"]["frame"]) / file_fps / slowmo
                    render.put_lines(zz, ["C%+d  %+.3fs" % (what, dt)], scale=0.45)
                    zooms[(k, what)] = zz
            elif role == "raw":
                raw = fr if s_a == 1.0 else cv2.resize(fr, (int(fr.shape[1] * s_a), int(fr.shape[0] * s_a)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / ("%s_%02d_contact_raw.jpg" % (prefix, k))), raw, [cv2.IMWRITE_JPEG_QUALITY, 90])
    for st in strokes:
        k = st["index"]
        m = st["metrics"]
        src = "auto candidate - verify" if st.get("source") == "auto" else "phases given"
        sheet = render.grid([tiles[(k, n)] for n in PHASES if (k, n) in tiles], 3)
        if sheet is not None:
            sheet = render.banner(sheet, "%s %d | %s | guess: %s | contact %s | %.0f fps real | reliable: %s" % (
                prefix.upper(), k, src, st["side_guess"], _tlabel(st["t_contact_s"], slowmo), vinfo["real_fps"],
                "yes" if m["quality"]["reliable"] else "NO"))
            cv2.imwrite(str(out / ("%s_%02d_sheet.jpg" % (prefix, k))), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        sp = [strips[(k, j)] for j in range(10) if (k, j) in strips]
        if sp:
            g = render.banner(render.grid(sp, 5), "%s %d | 10 evenly spaced frames from before READY to after RECOVERY" % (prefix.upper(), k))
            cv2.imwrite(str(out / ("%s_%02d_strip.jpg" % (prefix, k))), g, [cv2.IMWRITE_JPEG_QUALITY, 82])
        zs = [zooms[(k, o)] for o in (-2, -1, 0, 1, 2) if (k, o) in zooms]
        if zs:
            zg = render.banner(render.grid(zs, 5), "%s %d | playing hand + racket around contact (no overlay)" % (prefix.upper(), k))
            cv2.imwrite(str(out / ("%s_%02d_zoom.jpg" % (prefix, k))), zg, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _auto_strokes(result, s, hand):
    W = result["warnings"]
    fps = s.fps
    speed = metrics.hand_speed(s, hand)
    speed = signals.fill_gaps(speed, max(1, int(round(0.1 * fps))))
    speed = signals.smooth(speed, _odd(round(0.05 * fps)))
    # only forward swings count: zero the speed while the hand travels back (backswing / recovery)
    gated = np.where(swing_scores(s, hand) >= FORWARD_MIN, speed, 0.0)
    peaks, dropped = stroke_candidates(s, hand, merge_close_peaks(speed, detect_strokes(gated, fps), fps))
    if len(peaks) > MAX_STROKES:
        W.append(warn("MANY_STROKES", "检测到 %d 个候选板，只保留前 %d 个。" % (len(peaks), MAX_STROKES)))
        peaks = peaks[:MAX_STROKES]
    slowmo = result["video"]["slowmo_factor"]
    strokes = []
    for k, pk in enumerate(peaks):
        lo = 0 if k == 0 else (peaks[k - 1] + pk) // 2
        hi = len(s) if k == len(peaks) - 1 else (pk + peaks[k + 1]) // 2
        ph = signals.segment(speed, pk, fps, lo, hi)
        if too_small(s, hand, ph):
            dropped.append({"i": int(pk), "t": round(float(s.t[pk]), 3), "reason": "hand travel too small for a stroke"})
            continue
        m = _add_file_times(metrics.stroke_metrics(s, ph, hand), slowmo)
        strokes.append({"index": len(strokes) + 1, "source": "auto", "t_contact_s": round(float(s.t[pk]), 3),
                        "t_contact_file_s": round(float(s.t[pk]) * slowmo, 3), "side_guess": m["side_guess"],
                        "phases": _phases_dict(s, ph, slowmo), "metrics": m})
    result["dropped_candidates"] = dropped
    if not strokes:
        W.append(warn("NO_STROKES", "没有自动找到明显的挥拍。请看 timeline_*.jpg 自己找击球时刻，再用 --strokes 测量。"))
    if slowmo == 1.0 and slowmo_suspected([st["metrics"]["forward_ms"] for st in strokes]):
        W.append(warn("POSSIBLE_SLOWMO", "向前挥拍平均超过 0.6 秒，比正常击球慢很多：很可能是慢动作导出的视频。请用 --capture-fps 告诉脚本真实拍摄帧率（例如 240），否则时间和速度类数字都不对。"))
    return strokes, speed


def run(input_path, out_dir, hand="right", capture_fps=None, start=None, end=None, player="auto",
        model="full", frames_only=False, max_side=960, target_fps=120.0, strokes=None, at=None, timeline=None,
        side=None):
    """Analyse a video (auto candidates), or measure Claude-chosen moments (--strokes / --at / --timeline)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    if hand not in ("right", "left"):
        raise ValueError("hand must be right or left")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if Path(input_path).suffix.lower() in IMAGE_EXTS:
        return run_image(input_path, out, hand=hand, player=player, model=model, frames_only=frames_only,
                         max_side=max_side)
    given_strokes = parse_strokes(strokes) if strokes else None
    at_times = parse_times(at) if at else None

    info = video.probe(input_path)
    warnings = list(info.warnings)
    file_fps = info.file_fps
    if capture_fps and capture_fps < 0.95 * file_fps:
        warnings.append(warn("CAPTURE_FPS_IGNORED", "--capture-fps %.0f 比文件帧率 %.1f 还低，已忽略。" % (capture_fps, file_fps)))
        capture_fps = None
    real_fps = float(capture_fps) if capture_fps else file_fps
    slowmo = real_fps / file_fps
    stride = max(1, int(round(real_fps / target_fps)))
    analysis_fps = real_fps / stride
    first = max(0, int(round((start or 0.0) * file_fps)))
    last = info.frames - 1 if end is None else min(info.frames - 1, int(round(end * file_fps)))
    if end is None and (last - first + 1) / file_fps > MAX_FILE_SECONDS:
        last = first + int(MAX_FILE_SECONDS * file_fps) - 1
        warnings.append(warn("TRUNCATED", "视频较长，只分析了从 %.1f 秒起的 %d 秒（文件时间）。要分析其他片段，用 --start/--end 指定。" % (
            first / file_fps, MAX_FILE_SECONDS)))
    if last < first:
        raise ValueError("--start/--end 选出的片段是空的")
    if real_fps < 60:
        warnings.append(warn("LOW_FPS", "真实帧率只有 %.0f 帧/秒：击球瞬间多半没拍到，拍面角度、击球点和发力顺序只能粗看。"
                                        "建议用手机慢动作（120 或 240 帧）重拍；如果这本来就是慢动作导出的视频，用 --capture-fps 告诉脚本真实帧率。" % real_fps))
    s_a = video.scale_for(info.width, info.height, max_side)

    def analysed(fr):
        if s_a == 1.0:
            return fr
        return cv2.resize(fr, (int(round(fr.shape[1] * s_a)), int(round(fr.shape[0] * s_a))), interpolation=cv2.INTER_AREA)

    mode, why = "pose", ""
    if frames_only:
        mode, why = "frames-only", "frames-only requested"
    else:
        ok, why = pose_mod.available()
        if ok:
            try:
                model_path = pose_mod.ensure_model(model)
            except Exception as e:
                ok, why = False, "pose model unavailable: %s" % e
        if not ok:
            mode = "frames-only"
            warnings.append(warn("POSE_UNAVAILABLE", "没能启用骨架识别（%s）。这次只抽画面、不出角度数据；运行 scripts/setup_env.py 安装 mediapipe 后可以得到完整分析。" % why))
    if mode != "pose" and (given_strokes or at_times):
        raise RuntimeError("--strokes / --at 需要骨架识别（mediapipe），但现在不可用：%s" % why)

    vinfo = {"path": os.path.basename(input_path), "file_fps": round(file_fps, 3), "fps_source": info.fps_source,
             "real_fps": round(real_fps, 2), "slowmo_factor": round(slowmo, 3), "stride": stride,
             "analysis_fps": round(analysis_fps, 2), "frames": info.frames, "width": info.width,
             "height": info.height, "duration_s": round(info.duration_s, 2),
             "analysed_range_s": [round(first / file_fps, 2), round(last / file_fps, 2)]}
    task = "strokes" if given_strokes else "at" if at_times else "timeline" if timeline else "auto"
    sub = task_dir(out, task, timeline=timeline, at=at_times)
    sub.mkdir(parents=True, exist_ok=True)
    result = {"kind": "video", "task": task, "tool_version": __version__, "mode": mode, "hand": hand, "player": player,
              "video": vinfo, "warnings": warnings, "strokes": [], "files": []}

    s = None
    if mode == "pose":
        result["pose_model"] = {"variant": model, "mediapipe": why}
        st_ = os.stat(input_path)
        key = "|".join(str(x) for x in (os.path.abspath(input_path), st_.st_size, int(st_.st_mtime), stride, first, last,
                                         model, player, max_side, round(real_fps, 3), __version__))
        cache = out / "pose_cache.npz"
        s = load_series(cache, key)
        if s is None:
            s = pose_mod.track(((i, analysed(fr)) for i, fr in video.iter_frames(input_path, stride, first, last)),
                               model_path, file_fps, slowmo, player=player)
            save_series(s, cache, key)
        jumps = clean_series(s)
        if jumps > max(3, 0.05 * len(s)):
            warnings.append(warn("TRACKING_JUMPS", "有 %d 帧的人体位置突然跳变（视频剪辑跳切、镜头晃动或跟错了人），这些帧已剔除。" % jumps))
        warnings.extend(series_warnings(s, hand))

    if task == "timeline":
        t0, t1 = float(timeline[0]), float(timeline[1])
        step = float(timeline[2]) if len(timeline) > 2 and timeline[2] else max(1.0 / file_fps, (t1 - t0) / 29.0)
        fr_ids = [f for f in timeline_frames(t0, t1, step, file_fps) if first <= f <= last]
        if len(fr_ids) > 90:
            warnings.append(warn("TIMELINE_CAPPED", "这个窗口要出 %d 帧，只出了前 90 帧；请缩小窗口或加大步长。" % len(fr_ids)))
            fr_ids = fr_ids[:90]
        result["timeline"] = {"start": t0, "end": t1, "step": step, "frames": len(fr_ids)}
        _render_timeline(input_path, fr_ids, s, s_a, hand, slowmo, file_fps, sub, "timeline", per_page=15, cols=3, tile_w=480)
    elif task == "at":
        result["at"] = []
        nf = _Frames(s)
        want = {}
        for k, t in enumerate(at_times, 1):
            fidx = int(round(t * file_fps))
            i = nf.nearest(fidx)
            entry = {"index": k, "t_file": t, "frame": fidx, "pose_found": bool(i is not None and s.present[i])}
            if entry["pose_found"]:
                entry["metrics"] = metrics.photo_metrics(s, hand, i)
            result["at"].append(entry)
            want[fidx] = (k, i)
        T_orig = metrics.torso_px(s) / s_a
        for fidx, fr in video.iter_frames(input_path, keep=set(want)):
            k, i = want[fidx]
            img = render.resize_w(fr, 720)
            if i is not None and s.present[i]:
                render.draw_pose(img, s.img[i], s.vis[i], hand, scale=(720.0 / fr.shape[1]) / s_a)
                box = render.hand_box(s.img[i] / s_a, s.vis[i], hand, T_orig, fr.shape)
                if box is not None:
                    cv2.imwrite(str(sub / ("at_%02d_zoom.jpg" % k)), render.crop(fr, box, 400), [cv2.IMWRITE_JPEG_QUALITY, 90])
            render.put_lines(img, ["AT %d  %s  frame %d" % (k, _tlabel(fidx / file_fps / slowmo, slowmo), fidx)])
            cv2.imwrite(str(sub / ("at_%02d.jpg" % k)), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    elif task == "strokes":
        result["strokes"] = measure_strokes(s, hand, given_strokes, slowmo=slowmo, side=side)
        _render_strokes(result["strokes"], s, input_path, sub, hand, s_a, vinfo, prefix="measured")
    else:
        if mode == "pose":
            result["strokes"], speed = _auto_strokes(result, s, hand)
            cv2.imwrite(str(sub / "overview.png"), render.speed_plot(s.t * slowmo, speed, [
                {"phases": {k: {"t": v["t_file"]} for k, v in st["phases"].items()}} for st in result["strokes"]]))
            _render_strokes(result["strokes"], s, input_path, sub, hand, s_a, vinfo, prefix="stroke")
        else:
            _motion_strokes(result, input_path, sub, stride, first, last, file_fps, slowmo)
        dur = (last - first + 1) / file_fps
        step = auto_timeline_step(dur)
        fr_ids = timeline_frames(first / file_fps, last / file_fps, step, file_fps)
        result["timeline"] = {"start": round(first / file_fps, 2), "end": round(last / file_fps, 2), "step": step,
                              "frames": len(fr_ids)}
        _render_timeline(input_path, [f for f in fr_ids if first <= f <= last], s, s_a, hand, slowmo, file_fps, sub, "timeline")

    result["video_metrics"] = metrics.video_metrics(result["strokes"])
    result["examples"] = report.pick_examples(result["strokes"], task=task) if result["strokes"] else []
    report.write(sub, result)
    result["files"] = sorted(p.name for p in sub.iterdir() if p.is_file())
    result["out_dir"] = str(sub)
    report.write(sub, result)
    return result


def _motion_strokes(result, input_path, out, stride, first, last, file_fps, slowmo):
    W = result["warnings"]
    energy, frames, prev = [], [], None
    for i, fr in video.iter_frames(input_path, stride, first, last):
        g = cv2.cvtColor(render.resize_w(fr, 160), cv2.COLOR_BGR2GRAY).astype(np.float32)
        energy.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
        frames.append(i)
        prev = g
    fps = result["video"]["analysis_fps"]
    e = signals.smooth(np.array(energy), _odd(round(0.05 * fps)))
    peaks = detect_motion_strokes(e, fps)[:MAX_STROKES]
    t = np.array(frames) / file_fps / slowmo
    strokes = []
    offs = {"ready": -0.40, "backswing_end": -0.15, "mid_forward": -0.07, "contact": 0.0, "finish": 0.20, "recovery": 0.45}
    for k, pk in enumerate(peaks):
        phases = {}
        for name, dt in offs.items():
            j = int(np.clip(pk + round(dt * fps), 0, len(frames) - 1))
            phases[name] = {"i": j, "frame": int(frames[j]), "t": round(float(t[j]), 3), "t_file": round(float(t[j]) * slowmo, 3)}
        strokes.append({"index": k + 1, "source": "auto", "t_contact_s": round(float(t[pk]), 3),
                        "t_contact_file_s": round(float(t[pk]) * slowmo, 3), "side_guess": "unclear",
                        "phases": phases, "metrics": {"quality": {"reliable": False, "notes": ["frames-only: no pose measurements"]}}})
    result["strokes"] = strokes
    if not strokes:
        W.append(warn("NO_STROKES", "没有找到明显的挥拍（只按画面变化判断）。请看 timeline_*.jpg。"))
    if frames:
        cv2.imwrite(str(out / "overview.png"), render.speed_plot(t * slowmo, e, [
            {"phases": {k: {"t": v["t_file"]} for k, v in st["phases"].items()}} for st in strokes], ylabel="image motion"))
    if strokes:
        needed = {}
        for st in strokes:
            for name in PHASES:
                needed.setdefault(st["phases"][name]["frame"], []).append((st["index"], name))
        tiles = {}
        for fidx, fr in video.iter_frames(input_path, keep=set(needed)):
            for k, name in needed[fidx]:
                tl = render.resize_w(fr, TILE_W)
                render.put_lines(tl, ["%s (approx.)  %s" % (PHASE_LABEL[name], _tlabel(strokes[k - 1]["phases"][name]["t"], slowmo))])
                tiles[(k, name)] = tl
        for st in strokes:
            k = st["index"]
            sheet = render.grid([tiles[(k, n)] for n in PHASES if (k, n) in tiles], 3)
            if sheet is not None:
                sheet = render.banner(sheet, "STROKE %d | frames-only (no pose) | motion burst %s" % (k, _tlabel(st["t_contact_s"], slowmo)))
                cv2.imwrite(str(out / ("stroke_%02d_sheet.jpg" % k)), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def run_image(path, out, hand="right", player="auto", model="full", frames_only=False, max_side=960):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("读不出这张图片：%s（HEIC 等格式请先另存为 JPG 或 PNG）" % path)
    h, w = img.shape[:2]
    s_a = video.scale_for(w, h, max_side)
    small = img if s_a == 1.0 else cv2.resize(img, (int(w * s_a), int(h * s_a)), interpolation=cv2.INTER_AREA)
    warnings = []
    mode, why = ("frames-only", "frames-only requested") if frames_only else ("pose", "")
    if mode == "pose":
        ok, why = pose_mod.available()
        if ok:
            try:
                model_path = pose_mod.ensure_model(model)
            except Exception as e:
                ok, why = False, "pose model unavailable: %s" % e
        if not ok:
            mode = "frames-only"
            warnings.append(warn("POSE_UNAVAILABLE", "没能启用骨架识别（%s），只能直接看照片。" % why))
    result = {"kind": "image", "task": "photo", "tool_version": __version__, "mode": "image-" + mode, "hand": hand,
              "player": player, "video": {"path": os.path.basename(str(path)), "width": w, "height": h},
              "warnings": warnings, "strokes": [], "files": []}
    annotated = small.copy()
    if mode == "pose":
        s = pose_mod.detect_image(small, model_path, player=player)
        if s is None:
            warnings.append(warn("NO_PERSON", "照片里没有认出人（人太小、被遮挡或光线太暗）。"))
        else:
            result["photo_metrics"] = metrics.photo_metrics(s, hand)
            render.draw_pose(annotated, s.img[0], s.vis[0], hand)
            T_orig = metrics.torso_px(s) / s_a
            box = render.hand_box(s.img[0] / s_a, s.vis[0], hand, T_orig, img.shape, size_torso=1.2)
            if box is not None:
                cv2.imwrite(str(Path(out) / "photo_hand_zoom.jpg"), render.crop(img, box, 480), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(Path(out) / "photo_annotated.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    report.write(out, result)
    result["files"] = sorted(p.name for p in Path(out).iterdir() if p.is_file())
    result["out_dir"] = str(out)
    report.write(out, result)
    return result

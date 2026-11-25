"""Per-stroke and per-video metrics from a PoseSeries.

Names match references/analysis/metrics-reference.md. 3-D values come from MediaPipe's
world landmarks (single-camera estimates); 2-D values from image pixels. Every value is a
float or None (never NaN) so it serialises to JSON.
"""
import math

import numpy as np

from . import geometry as g
from .landmarks import IDX, side_names

FOREHAND_LATERAL_M = 0.15   # wrist this far out to the playing side of the hip centre at contact -> forehand
BACKHAND_LATERAL_M = 0.08   # wrist closer than this (or across the body) -> backhand
MIN_FPS_FOR_SEQUENCE = 100.0
MAX_HAND_SPEED = 20.0  # torso lengths/s (about 10 m/s at the wrist): faster means the skeleton jumped


def num(x, nd=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


def _W(s, i, name):
    return s.world[i, IDX[name]]


def _P(s, i, name):
    return s.img[i, IDX[name]]


def _mid(a, b):
    return (np.asarray(a, float) + np.asarray(b, float)) / 2.0


def _hip_mid_w(s, i):
    return _mid(_W(s, i, "left_hip"), _W(s, i, "right_hip"))


def _sh_mid_w(s, i):
    return _mid(_W(s, i, "left_shoulder"), _W(s, i, "right_shoulder"))


def _hip_mid_p(s, i):
    return _mid(_P(s, i, "left_hip"), _P(s, i, "right_hip"))


def _sh_mid_p(s, i):
    return _mid(_P(s, i, "left_shoulder"), _P(s, i, "right_shoulder"))


VIS_MIN = 0.5


def seen(s, i, *names):
    """True when every named landmark is visible enough at frame i (MediaPipe still guesses hidden joints)."""
    return all(s.vis[i, IDX[n]] >= VIS_MIN and np.all(np.isfinite(s.img[i, IDX[n]])) for n in names)


HIPS = ("left_hip", "right_hip")
SHOULDERS = ("left_shoulder", "right_shoulder")


def torso_px(s):
    d = np.linalg.norm(_mid(s.img[:, IDX["left_shoulder"]], s.img[:, IDX["right_shoulder"]])
                       - _mid(s.img[:, IDX["left_hip"]], s.img[:, IDX["right_hip"]]), axis=1)
    return float(np.nanmedian(d)) if np.any(np.isfinite(d)) else float("nan")


def torso_m(s):
    d = np.linalg.norm(_mid(s.world[:, IDX["left_shoulder"]], s.world[:, IDX["right_shoulder"]])
                       - _mid(s.world[:, IDX["left_hip"]], s.world[:, IDX["right_hip"]]), axis=1)
    return float(np.nanmedian(d)) if np.any(np.isfinite(d)) else float("nan")


def height_level(s, i, y, hips_visible=True):
    """Image height of y as a level: 0 = hip line, 1 = shoulder line (above shoulder > 1).

    When the hips are hidden (e.g. behind the table) the hip line is estimated as the shoulder line
    plus the clip's median torso length.
    """
    yh, ys = _hip_mid_p(s, i)[1], _sh_mid_p(s, i)[1]
    if not hips_visible:
        yh = ys + torso_px(s)
    if not (np.isfinite(yh) and np.isfinite(ys) and np.isfinite(y)) or yh == ys:
        return float("nan")
    return (yh - y) / (yh - ys)


def median3(x):
    """Running median over 3 samples (removes single-frame landmark spikes); keeps NaN where all are NaN."""
    a = np.asarray(x, float)
    out = a.copy()
    for k in range(1, len(a) - 1):
        win = a[k - 1:k + 2]
        if np.any(np.isfinite(win)):
            out[k] = np.nanmedian(win, axis=0)
    return out


def hand_rel_shoulders_px(s, hand):
    """Playing wrist relative to the shoulder centre (pixels), spike-filtered.

    Shoulders are used rather than hips: hips are often hidden behind the table and jitter.
    """
    p, _ = side_names(hand)
    rel = s.img[:, IDX[p + "_wrist"]] - _mid(s.img[:, IDX["left_shoulder"]], s.img[:, IDX["right_shoulder"]])
    return median3(rel)


def central_velocity(x, fps, span_s=0.04):
    """Velocity by central differences over +-span_s (at least one frame); edges use one-sided steps."""
    x = np.asarray(x, float)
    n = len(x)
    h = max(1, int(round(span_s * fps)))
    v = np.gradient(x, axis=0) * fps if n >= 2 else np.full_like(x, np.nan)
    if n > 2 * h:
        v[h:n - h] = (x[2 * h:] - x[:n - 2 * h]) / (2.0 * h) * fps
    return v


def hand_speed(s, hand):
    """Speed of the playing wrist relative to the shoulder centre, in torso lengths per second (2-D)."""
    rel = hand_rel_shoulders_px(s, hand)
    T = torso_px(s)
    if len(s) < 2 or not np.isfinite(T) or T == 0:
        return np.full(len(s), np.nan)
    vel = central_velocity(rel, s.fps)
    return np.linalg.norm(vel, axis=1) / T


def _horiz_unit(v):
    v = np.asarray(v, float)
    h = np.array([v[0], v[2]])
    n = np.linalg.norm(h)
    return h / n if np.isfinite(n) and n > 0 else np.array([np.nan, np.nan])


def _dot_h(v, u2):
    v = np.asarray(v, float)
    return float(v[0] * u2[0] + v[2] * u2[1])


def side_guess(s, c, hand):
    """forehand / backhand / unclear from where the playing wrist is at contact."""
    p, o = side_names(hand)
    if not seen(s, c, p + "_wrist", *SHOULDERS):  # shoulders, not hips: hips are often hidden by the table
        return "unclear", None
    u = _horiz_unit(_W(s, c, p + "_shoulder") - _W(s, c, o + "_shoulder"))
    lat = _dot_h(_W(s, c, p + "_wrist") - _sh_mid_w(s, c), u)
    if not np.isfinite(lat):  # fall back to 2-D image coordinates
        T = torso_px(s)
        v = _P(s, c, p + "_shoulder") - _P(s, c, o + "_shoulder")
        nv = np.linalg.norm(v)
        if not (np.isfinite(nv) and nv > 0 and np.isfinite(T) and T > 0):
            return "unclear", None
        lat2 = float(np.dot(_P(s, c, p + "_wrist") - _sh_mid_p(s, c), v / nv)) / T
        if lat2 > 0.35:
            return "forehand", None
        if lat2 < 0.15:
            return "backhand", None
        return "unclear", None
    if lat > FOREHAND_LATERAL_M:
        return "forehand", lat
    if lat < BACKHAND_LATERAL_M:
        return "backhand", lat
    return "unclear", lat


def _weight_ratio(s, i, hand):
    """0 = hip centre over the playing-side foot, 1 = over the other foot (horizontal plane)."""
    p, o = side_names(hand)
    if not seen(s, i, *HIPS, "left_ankle", "right_ankle"):
        return float("nan")
    ap, ao, h = _W(s, i, p + "_ankle"), _W(s, i, o + "_ankle"), _hip_mid_w(s, i)
    d = np.array([ao[0] - ap[0], ao[2] - ap[2]])
    dd = float(np.dot(d, d))
    if not np.isfinite(dd) or dd < 1e-6:
        return float("nan")
    return float(np.dot(np.array([h[0] - ap[0], h[2] - ap[2]]), d) / dd)


def _yaw_series(s, a, b, lo, hi):
    y = np.array([g.yaw_deg(_W(s, i, a), _W(s, i, b)) for i in range(lo, hi + 1)])
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return None
    y[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), y[ok]) if (~ok).any() else y[~ok]
    return np.degrees(np.unwrap(np.radians(y)))


def _peak_time_ms(rate, lo, c, fps):
    if rate is None or not np.any(np.isfinite(rate)):
        return None
    k = int(np.nanargmax(np.abs(rate)))
    return (lo + k - c) / fps * 1000.0


KEY_PARTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


def _quality(s, r, rc, hand, notes):
    p, o = side_names(hand)
    sl = slice(r, rc + 1)
    coverage = float(np.mean(s.present[sl])) if rc >= r else 0.0
    names = [p + "_shoulder", p + "_elbow", p + "_wrist", o + "_shoulder", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle", "nose"]
    vis = {n: float(np.mean(s.vis[sl, IDX[n]])) for n in names}
    low = [n for n, v in vis.items() if v < 0.5]
    mean_vis = float(np.mean(list(vis.values())))
    # judged on the playing arm and shoulders; hidden legs (behind the table) only blank the leg metrics
    arm = [vis[n] for n in (p + "_shoulder", p + "_elbow", p + "_wrist", o + "_shoulder")]
    reliable = coverage >= 0.8 and float(np.mean(arm)) >= 0.6 and not any(
        n in low for n in (p + "_shoulder", p + "_elbow", p + "_wrist"))
    return {"pose_coverage": num(coverage, 2), "mean_visibility": num(mean_vis, 2),
            "low_visibility_parts": low, "reliable": bool(reliable), "notes": notes}


def stroke_metrics(s, ph, hand):
    """All per-stroke metrics for one stroke (phases ph) of a `hand`-handed player."""
    p, o = side_names(hand)
    fps = float(s.fps)
    r, b, m, c, f, rc = ph.ready, ph.backswing_end, ph.mid_forward, ph.contact, ph.finish, ph.recovery
    notes = []
    T, Tm = torso_px(s), torso_m(s)
    core = slice(b, f + 1)  # the swing itself; ready/recovery frames may lack a pose without harming it
    hidden = {n for n in ("left_knee", "right_knee", "left_ankle", "right_ankle", "left_hip", "right_hip")
              if f >= b and float(np.mean(s.vis[core, IDX[n]])) < VIS_MIN}

    def ms(a, z):
        return num((z - a) / fps * 1000.0, 0)

    def elbow(i):
        if not seen(s, i, p + "_shoulder", p + "_elbow", p + "_wrist"):
            return float("nan")
        return g.angle_3pt(_W(s, i, p + "_shoulder"), _W(s, i, p + "_elbow"), _W(s, i, p + "_wrist"))

    def elbow2d(i):
        if not seen(s, i, p + "_shoulder", p + "_elbow", p + "_wrist"):
            return float("nan")
        return g.angle_3pt(_P(s, i, p + "_shoulder"), _P(s, i, p + "_elbow"), _P(s, i, p + "_wrist"))

    def knee(i, side):
        if side + "_knee" in hidden or side + "_ankle" in hidden:
            return float("nan")  # mostly hidden during the swing: single "visible" frames are often the table edge
        if not seen(s, i, side + "_hip", side + "_knee", side + "_ankle"):
            return float("nan")
        return g.angle_3pt(_W(s, i, side + "_hip"), _W(s, i, side + "_knee"), _W(s, i, side + "_ankle"))

    def lean(i):
        if not seen(s, i, *HIPS, *SHOULDERS):
            return float("nan")
        return g.lean_from_vertical_deg(_hip_mid_w(s, i), _sh_mid_w(s, i))

    def lvl(i, name):
        if not seen(s, i, name, *SHOULDERS):
            return float("nan")
        hv = seen(s, i, *HIPS)
        if not hv and "hip line estimated from shoulders (hips hidden)" not in notes:
            notes.append("hip line estimated from shoulders (hips hidden)")
        return height_level(s, i, _P(s, i, name)[1], hips_visible=hv)

    sg, lateral = side_guess(s, c, hand)
    out = {"side_guess": sg,
           "backswing_ms": ms(r, b), "forward_ms": ms(b, c), "follow_ms": ms(c, f)}

    e3, e2 = elbow(c), elbow2d(c)
    if np.isfinite(e3) and np.isfinite(e2) and abs(e3 - e2) > 30:
        notes.append("contact elbow angle: 3-D %.0f vs 2-D %.0f differ a lot (arm pointing at/away from camera)" % (e3, e2))
    out["elbow_angle_deg"] = {"backswing_end": num(elbow(b), 1), "contact": num(e3, 1), "finish": num(elbow(f), 1)}

    out["knee_angle_deg"] = {side_key: {"ready": num(knee(r, sd), 1), "backswing_end": num(knee(b, sd), 1),
                                        "contact": num(knee(c, sd), 1)}
                             for side_key, sd in (("playing", p), ("other", o))}
    out["knee_min_deg"] = {}
    for side_key, sd in (("playing", p), ("other", o)):
        vals = [knee(i, sd) for i in range(b, c + 1)]
        vals = [v for v in vals if np.isfinite(v)]
        out["knee_min_deg"][side_key] = num(min(vals), 1) if vals else None

    out["trunk_lean_deg"] = {"backswing_end": num(lean(b), 1), "contact": num(lean(c), 1)}

    fb = g.facing_vector(_W(s, b, "left_shoulder"), _W(s, b, "right_shoulder"), _W(s, b, "nose"))
    trunk_b = seen(s, b, *SHOULDERS, "nose")
    out["shoulder_turn_deg"] = num(g.horizontal_turn_deg(_W(s, b, p + "_shoulder") - _sh_mid_w(s, b),
                                                         _W(s, f, p + "_shoulder") - _sh_mid_w(s, f), fb), 1) \
        if trunk_b and seen(s, f, *SHOULDERS) else None
    out["hip_turn_deg"] = num(g.horizontal_turn_deg(_W(s, b, p + "_hip") - _hip_mid_w(s, b),
                                                    _W(s, f, p + "_hip") - _hip_mid_w(s, f), fb), 1) \
        if trunk_b and seen(s, b, *HIPS) and seen(s, f, *HIPS) else None
    ysh = g.yaw_deg(_W(s, b, "left_shoulder"), _W(s, b, "right_shoulder"))
    yhp = g.yaw_deg(_W(s, b, "left_hip"), _W(s, b, "right_hip"))
    out["x_factor_deg"] = num(abs(g.wrap_deg(ysh - yhp)), 1) \
        if np.isfinite(ysh) and np.isfinite(yhp) and seen(s, b, *SHOULDERS, *HIPS) else None

    def swr(i):
        if not seen(s, i, *SHOULDERS):
            return float("nan")
        d = np.linalg.norm(_P(s, i, "left_shoulder") - _P(s, i, "right_shoulder"))
        return d / T if np.isfinite(T) and T > 0 else float("nan")
    out["shoulder_width_ratio"] = {"ready": num(swr(r), 2), "backswing_end": num(swr(b), 2),
                                   "contact": num(swr(c), 2), "finish": num(swr(f), 2)}

    feet_hidden = "left_ankle" in hidden or "right_ankle" in hidden
    wb, wf = (float("nan"), float("nan")) if feet_hidden else (_weight_ratio(s, b, hand), _weight_ratio(s, f, hand))
    out["weight_shift"] = {"backswing_end": num(wb, 2), "finish": num(wf, 2), "delta": num(wf - wb, 2)}

    out["contact_height_level"] = num(lvl(c, p + "_wrist"), 2)
    fc = g.facing_vector(_W(s, c, "left_shoulder"), _W(s, c, "right_shoulder"), _W(s, c, "nose"))
    rel_c = _W(s, c, p + "_wrist") - _hip_mid_w(s, c)
    body_c = seen(s, c, *HIPS, *SHOULDERS)
    out["contact_forward_m"] = num(_dot_h(rel_c, _horiz_unit(fc)), 2) \
        if body_c and seen(s, c, p + "_wrist", "nose") else None
    out["contact_lateral_m"] = num(lateral, 2)
    out["elbow_body_gap_m"] = num(g.point_line_distance(_W(s, c, p + "_elbow"), _hip_mid_w(s, c), _sh_mid_w(s, c)), 2) \
        if body_c and seen(s, c, p + "_elbow") else None

    dv = (_W(s, f, p + "_wrist") - _hip_mid_w(s, f)) - (_W(s, b, p + "_wrist") - _hip_mid_w(s, b))
    if np.all(np.isfinite(dv)) and np.isfinite(Tm) and Tm > 0 and seen(s, b, p + "_wrist", *HIPS) \
            and seen(s, f, p + "_wrist", *HIPS):
        up, horiz = -dv[1] / Tm, math.hypot(dv[0], dv[2]) / Tm
        out["swing_path"] = {"up": num(up, 2), "horiz": num(horiz, 2), "angle_deg": num(math.degrees(math.atan2(up, horiz)), 1)}
    else:
        out["swing_path"] = {"up": None, "horiz": None, "angle_deg": None}

    out["finish_height_level"] = num(lvl(f, p + "_wrist"), 2)
    out["head_level_finish"] = num(lvl(f, "nose"), 2)

    nb = _P(s, b, "nose")
    hm = [np.linalg.norm(_P(s, i, "nose") - nb) for i in range(b, f + 1) if seen(s, i, "nose") and seen(s, b, "nose")]
    hm = [v for v in hm if np.isfinite(v)]
    out["head_motion"] = num(max(hm) / T, 2) if hm and np.isfinite(T) and T > 0 else None

    out["free_arm_height_level"] = num(lvl(c, o + "_wrist"), 2)

    sp = hand_speed(s, hand)
    win = sp[b:f + 1] if f >= b else sp[c:c + 1]
    out["peak_hand_speed"] = num(np.nanmax(win), 1) if np.any(np.isfinite(win)) else None

    seq = {"hip": None, "shoulder": None, "elbow": None}
    order_ok = None
    if fps < MIN_FPS_FOR_SEQUENCE:
        notes.append("sequence timing not judged: %.0f fps is below %.0f" % (fps, MIN_FPS_FOR_SEQUENCE))
    else:
        lo, hi = b, min(f, c + max(1, int(round(0.05 * fps))))
        if hi - lo >= 3:
            hy = _yaw_series(s, "left_hip", "right_hip", lo, hi)
            sy = _yaw_series(s, "left_shoulder", "right_shoulder", lo, hi)
            ev = np.array([elbow(i) for i in range(lo, hi + 1)])
            seq["hip"] = num(_peak_time_ms(None if hy is None else np.gradient(hy), lo, c, fps), 0)
            seq["shoulder"] = num(_peak_time_ms(None if sy is None else np.gradient(sy), lo, c, fps), 0)
            seq["elbow"] = num(_peak_time_ms(np.gradient(ev) if np.isfinite(ev).sum() >= 3 else None, lo, c, fps), 0)
            if None not in seq.values():
                tol = 1000.0 / fps
                order_ok = bool(seq["hip"] <= seq["shoulder"] + tol and seq["shoulder"] <= seq["elbow"] + tol
                                and seq["elbow"] <= tol)
    out["sequence_ms"] = seq
    out["sequence_order_ok"] = order_ok

    ls, rs = _W(s, r, "left_shoulder"), _W(s, r, "right_shoulder")
    la, ra = _W(s, r, "left_ankle"), _W(s, r, "right_ankle")
    shw = np.linalg.norm(ls - rs)
    out["stance_width_ratio"] = num(np.linalg.norm(la - ra) / shw, 2) \
        if np.isfinite(shw) and shw > 0 and not feet_hidden and seen(s, r, *SHOULDERS, "left_ankle", "right_ankle") else None

    def rel_p(i):
        return _P(s, i, p + "_wrist") - _hip_mid_p(s, i)
    rd = np.linalg.norm(rel_p(rc) - rel_p(r)) if seen(s, r, p + "_wrist", *HIPS) and seen(s, rc, p + "_wrist", *HIPS) \
        else float("nan")
    out["recovery_dist"] = num(rd / T, 2) if np.isfinite(rd) and np.isfinite(T) and T > 0 else None

    implausible = []
    for key, lim in (("shoulder_turn_deg", 120), ("hip_turn_deg", 120)):
        if out.get(key) is not None and abs(out[key]) > lim:
            implausible.append("%s %.0f" % (key, out[key]))
    if out.get("contact_height_level") is not None and not (-1.0 <= out["contact_height_level"] <= 2.5):
        implausible.append("contact_height_level %.2f" % out["contact_height_level"])
    if (out.get("trunk_lean_deg") or {}).get("contact") is not None and out["trunk_lean_deg"]["contact"] > 60:
        implausible.append("trunk_lean_deg.contact %.0f" % out["trunk_lean_deg"]["contact"])
    if out.get("peak_hand_speed") is not None and out["peak_hand_speed"] > MAX_HAND_SPEED:
        implausible.append("peak_hand_speed %.1f" % out["peak_hand_speed"])
    if np.isfinite(e3) and np.isfinite(e2) and abs(e3 - e2) > 45:
        implausible.append("elbow 3-D %.0f vs 2-D %.0f" % (e3, e2))
    if implausible:
        notes.append("implausible values (skeleton probably wrong in some frames, check the images): " + ", ".join(implausible))
    out["quality"] = _quality(s, b, f, hand, notes)
    if implausible:
        out["quality"]["reliable"] = False
    return out


def photo_metrics(s, hand, i=0):
    """Static angles for one frame (a photo of a stance, grip position or stroke moment)."""
    p, o = side_names(hand)

    def ang(side, a, b, c):
        if not seen(s, i, side + a, side + b, side + c):
            return None
        return num(g.angle_3pt(_W(s, i, side + a), _W(s, i, side + b), _W(s, i, side + c)), 1)

    def lvl(name):
        if not seen(s, i, name, *SHOULDERS):
            return None
        return num(height_level(s, i, _P(s, i, name)[1], hips_visible=seen(s, i, *HIPS)), 2)

    shw = np.linalg.norm(_W(s, i, "left_shoulder") - _W(s, i, "right_shoulder"))
    stw = np.linalg.norm(_W(s, i, "left_ankle") - _W(s, i, "right_ankle"))
    feet_ok = seen(s, i, "left_ankle", "right_ankle", *SHOULDERS)
    T = torso_px(s)
    swd = np.linalg.norm(_P(s, i, "left_shoulder") - _P(s, i, "right_shoulder"))
    return {
        "shoulder_width_ratio": num(swd / T, 2) if seen(s, i, *SHOULDERS) and np.isfinite(T) and T > 0 else None,
        "elbow_angle_deg": {"playing": ang(p, "_shoulder", "_elbow", "_wrist"),
                            "other": ang(o, "_shoulder", "_elbow", "_wrist")},
        "knee_angle_deg": {"playing": ang(p, "_hip", "_knee", "_ankle"), "other": ang(o, "_hip", "_knee", "_ankle")},
        "trunk_lean_deg": num(g.lean_from_vertical_deg(_hip_mid_w(s, i), _sh_mid_w(s, i)), 1)
        if seen(s, i, *HIPS, *SHOULDERS) else None,
        "stance_width_ratio": num(stw / shw, 2) if feet_ok and np.isfinite(shw) and shw > 0 else None,
        "wrist_height_level": lvl(p + "_wrist"),
        "free_arm_height_level": lvl(o + "_wrist"),
        "quality": {"mean_visibility": num(float(np.mean(s.vis[i])), 2),
                    "low_visibility_parts": [n for n in ("%s_wrist" % p, "%s_elbow" % p, "%s_shoulder" % p,
                                                         "left_knee", "right_knee", "left_ankle", "right_ankle")
                                             if s.vis[i, IDX[n]] < 0.5]},
    }


CONSISTENCY_KEYS = ("elbow_angle_deg.contact", "knee_min_deg.playing", "shoulder_turn_deg", "hip_turn_deg",
                    "contact_height_level", "contact_forward_m", "swing_path.angle_deg", "forward_ms",
                    "weight_shift.delta", "finish_height_level")


def _get(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def video_metrics(strokes):
    """Rhythm (time between contacts) and per-side consistency (mean/std/n) across strokes."""
    ts = sorted(st["t_contact_s"] for st in strokes if st.get("t_contact_s") is not None)
    iv = [round(b - a, 3) for a, b in zip(ts, ts[1:])]
    rhythm = {"intervals_s": iv,
              "mean_s": num(np.mean(iv), 3) if iv else None,
              "cv": num(np.std(iv) / np.mean(iv), 2) if len(iv) >= 2 and np.mean(iv) > 0 else None}
    cons = {}
    for side in ("forehand", "backhand", "unclear"):
        group = [st for st in strokes if st.get("side_guess") == side
                 and ((st.get("metrics") or {}).get("quality") or {}).get("reliable", True)]
        if not group:
            continue
        cons[side] = {}
        for key in CONSISTENCY_KEYS:
            vals = [num(_get(st.get("metrics", {}), key)) for st in group]
            vals = [v for v in vals if v is not None]
            if vals:
                cons[side][key] = {"mean": num(np.mean(vals), 2), "std": num(np.std(vals), 2), "n": len(vals)}
    return {"rhythm": rhythm, "consistency": cons}

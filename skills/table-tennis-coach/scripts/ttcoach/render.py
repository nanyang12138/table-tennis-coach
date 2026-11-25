"""Drawing helpers: skeleton overlay, labelled tiles, grids, hand zooms and the speed overview.

Text drawn into images is ASCII only (OpenCV fonts have no CJK glyphs).
"""
import cv2
import numpy as np

from .landmarks import EDGES, IDX

PLAY_ARM = (0, 200, 255)   # BGR orange: playing arm
TRUNK = (255, 255, 255)
LEGS = (255, 190, 60)
OTHER_ARM = (180, 180, 180)


def _ok(p, v, vmin=0.5):  # same threshold as the metrics: do not draw guessed joints
    return p is not None and np.all(np.isfinite(p)) and v >= vmin


def draw_pose(img, xy, vis, hand, scale=1.0, thickness=2):
    """Draw the skeleton (xy in analysed pixels times `scale`) onto img in place."""
    if xy is None or not np.any(np.isfinite(xy)):
        return img
    pts = xy * scale
    for a, b in EDGES:
        ia, ib = IDX[a], IDX[b]
        if not (_ok(pts[ia], vis[ia]) and _ok(pts[ib], vis[ib])):
            continue
        if a.startswith(hand + "_") and b.startswith(hand + "_") and ("elbow" in a + b or "wrist" in a + b or "index" in b):
            col, th = PLAY_ARM, thickness + 2
        elif "knee" in a + b or "ankle" in a + b or "heel" in a + b or "foot" in a + b:
            col, th = LEGS, thickness
        elif "elbow" in a + b or "wrist" in a + b or "index" in a + b:
            col, th = OTHER_ARM, thickness
        else:
            col, th = TRUNK, thickness
        cv2.line(img, tuple(int(v) for v in pts[ia]), tuple(int(v) for v in pts[ib]), col, th, cv2.LINE_AA)
    for name in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        for side in ("left", "right"):
            j = IDX[side + "_" + name]
            if _ok(pts[j], vis[j]):
                cv2.circle(img, tuple(int(v) for v in pts[j]), thickness + 2,
                           PLAY_ARM if side == hand else TRUNK, -1, cv2.LINE_AA)
    return img


def put_lines(img, lines, origin=(6, 6), scale=0.5, color=(0, 255, 255)):
    """Text block with a dark background in the top-left corner."""
    if not lines:
        return img
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = 1 if scale < 0.7 else 2
    sizes = [cv2.getTextSize(str(t), font, scale, th)[0] for t in lines]
    lh = max(s[1] for s in sizes) + 6
    w = max(s[0] for s in sizes) + 10
    x0, y0 = origin
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + lh * len(lines) + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for k, t in enumerate(lines):
        cv2.putText(img, str(t), (x0 + 5, y0 + lh * (k + 1) - 2), font, scale, color, th, cv2.LINE_AA)
    return img


def resize_w(img, w):
    h0, w0 = img.shape[:2]
    return cv2.resize(img, (int(w), max(1, int(round(h0 * w / float(w0))))), interpolation=cv2.INTER_AREA)


def grid(tiles, cols, pad=4, bg=(30, 30, 30)):
    if not tiles:
        return None
    H = max(t.shape[0] for t in tiles)
    W = max(t.shape[1] for t in tiles)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        row = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, W - t.shape[1], cv2.BORDER_CONSTANT, value=bg) for t in row]
        row += [np.full((H, W, 3), bg, np.uint8)] * (cols - len(row))
        rows.append(np.hstack([cv2.copyMakeBorder(t, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg) for t in row]))
    return np.vstack(rows)


def banner(img, text, height=34):
    bar = np.full((height, img.shape[1], 3), (20, 20, 20), np.uint8)
    cv2.putText(bar, text, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def hand_box(xy_orig, vis, hand, torso_orig, frame_shape, size_torso=1.9):
    """Square box (x0, y0, x1, y1) in original pixels around the playing hand and racket.

    The centre is pushed along the forearm (elbow -> wrist) so an extended arm does not cut off the racket head.
    """
    js = [IDX[hand + "_" + n] for n in ("wrist", "index", "pinky", "thumb")]
    pts = np.array([xy_orig[j] for j in js if np.all(np.isfinite(xy_orig[j]))])
    if len(pts) == 0 or not np.isfinite(torso_orig) or torso_orig <= 0:
        return None
    cx, cy = pts.mean(axis=0)
    el, wr = xy_orig[IDX[hand + "_elbow"]], xy_orig[IDX[hand + "_wrist"]]
    if np.all(np.isfinite(el)) and np.all(np.isfinite(wr)):
        d = wr - el
        n = np.linalg.norm(d)
        if n > 0:
            cx, cy = cx + 0.45 * torso_orig * d[0] / n, cy + 0.45 * torso_orig * d[1] / n
    half = 0.5 * size_torso * torso_orig
    h, w = frame_shape[:2]
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    return x0, y0, x1, y1


def crop(img, box, out=300):
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    pad = [max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w)]
    c = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    c = cv2.copyMakeBorder(c, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=(40, 40, 40))
    return cv2.resize(c, (out, out), interpolation=cv2.INTER_CUBIC)


def speed_plot(t, speed, strokes, width=1200, height=280, ylabel="hand speed (torso/s)"):
    """Line plot of a 1-D signal with stroke phase markers (strokes: list of dicts with 't' per phase)."""
    img = np.full((height, width, 3), 250, np.uint8)
    L, R, Tm, B = 60, 15, 30, 35
    t = np.asarray(t, float)
    y = np.nan_to_num(np.asarray(speed, float), nan=0.0)
    if len(t) < 2:
        return img
    t0, t1 = float(t[0]), float(t[-1]) if t[-1] > t[0] else float(t[0]) + 1
    ymax = float(np.max(y)) * 1.1 if np.max(y) > 0 else 1.0

    def X(v):
        return int(L + (v - t0) / (t1 - t0) * (width - L - R))

    def Y(v):
        return int(height - B - v / ymax * (height - Tm - B))
    cv2.line(img, (L, height - B), (width - R, height - B), (0, 0, 0), 1)
    cv2.line(img, (L, Tm), (L, height - B), (0, 0, 0), 1)
    for k in range(6):
        tv = t0 + (t1 - t0) * k / 5.0
        cv2.putText(img, "%.1fs" % tv, (X(tv) - 15, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img, ylabel, (L + 5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    pts = np.array([[X(a), Y(b)] for a, b in zip(t, y)], np.int32)
    cv2.polylines(img, [pts], False, (180, 90, 0), 1, cv2.LINE_AA)
    for k, st in enumerate(strokes, 1):
        ph = st.get("phases", {})
        for name, col in (("backswing_end", (200, 120, 0)), ("contact", (0, 0, 220)), ("finish", (0, 150, 0))):
            if name in ph and ph[name].get("t") is not None:
                xx = X(ph[name]["t"])
                cv2.line(img, (xx, Tm), (xx, height - B), col, 2 if name == "contact" else 1)
        if "contact" in ph:
            cv2.putText(img, str(k), (X(ph["contact"]["t"]) + 3, Tm + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
    return img

"""Small geometry helpers (numpy only). Angles in degrees.

World convention (MediaPipe pose_world_landmarks): metres, hip centre at the origin,
x to image right, y down, z away from the camera. "Horizontal" means the x-z plane.
"""
import math

import numpy as np


def _arr(p):
    return np.asarray(p, dtype=float)


def angle_3pt(a, b, c):
    """Angle at b formed by a-b-c (2D or 3D). NaN if a point is missing or a segment has zero length."""
    a, b, c = _arr(a), _arr(b), _arr(c)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if not (np.isfinite(n1) and np.isfinite(n2)) or n1 == 0 or n2 == 0:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def wrap_deg(d):
    """Wrap an angle to (-180, 180]."""
    r = (float(d) + 180.0) % 360.0 - 180.0
    return 180.0 if r == -180.0 else r


def yaw_deg(p_from, p_to):
    """Direction of the p_from->p_to line in the horizontal (x-z) plane; 0 = +x, 90 = +z."""
    a, b = _arr(p_from), _arr(p_to)
    dx, dz = b[0] - a[0], b[2] - a[2]
    if not (np.isfinite(dx) and np.isfinite(dz)) or (dx == 0 and dz == 0):
        return float("nan")
    return math.degrees(math.atan2(dz, dx))


def lean_from_vertical_deg(lower, upper):
    """Angle between the lower->upper segment and straight up (-y)."""
    v = _arr(upper) - _arr(lower)
    return angle_3pt(v, (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)) if len(v) == 3 else \
        angle_3pt(v, (0.0, 0.0), (0.0, -1.0))


def point_line_distance(p, a, b):
    """Distance from point p to the infinite line through a and b."""
    p, a, b = _arr(p), _arr(a), _arr(b)
    d = b - a
    n = np.linalg.norm(d)
    if not np.isfinite(n) or n == 0:
        return float("nan")
    return float(np.linalg.norm(np.cross(p - a, d)) / n) if len(p) == 3 else \
        float(abs(d[0] * (p - a)[1] - d[1] * (p - a)[0]) / n)


def _horiz(v):
    v = _arr(v)
    return np.array([v[0], v[2]])


def horizontal_turn_deg(start, end, forward):
    """Signed rotation of a body-relative vector in the horizontal plane.

    Magnitude: angle between start and end projected on x-z.
    Sign: + when the end vector points further along `forward` than the start vector
    (e.g. the playing-side shoulder comes forward), - otherwise.
    """
    s, e, f = _horiz(start), _horiz(end), _horiz(forward)
    ns, ne = np.linalg.norm(s), np.linalg.norm(e)
    if not (np.isfinite(ns) and np.isfinite(ne)) or ns == 0 or ne == 0:
        return float("nan")
    mag = math.degrees(math.acos(float(np.clip(np.dot(s, e) / (ns * ne), -1.0, 1.0))))
    nf = np.linalg.norm(f)
    if not np.isfinite(nf) or nf == 0:
        return float("nan")
    f = f / nf
    return mag if np.dot(e / ne, f) >= np.dot(s / ns, f) else -mag


def facing_vector(left_shoulder, right_shoulder, nose):
    """Horizontal unit vector the trunk faces: perpendicular to the shoulder line, on the nose side."""
    ls, rs, no = _arr(left_shoulder), _arr(right_shoulder), _arr(nose)
    line = _horiz(rs - ls)
    n = np.linalg.norm(line)
    if not np.isfinite(n) or n == 0:
        return np.array([np.nan, 0.0, np.nan])
    perp = np.array([-line[1], line[0]]) / n
    to_nose = _horiz(no - (ls + rs) / 2.0)
    if np.all(np.isfinite(to_nose)) and np.dot(perp, to_nose) < 0:
        perp = -perp
    return np.array([perp[0], 0.0, perp[1]])

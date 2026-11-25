"""MediaPipe Tasks PoseLandmarker wrapper (mediapipe 0.10.x and 1.x; the old `mp.solutions` API is not used)."""
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .series import PoseSeries

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")


def cache_dir():
    env = os.environ.get("TTCOACH_CACHE")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "table-tennis-coach"
    return Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "table-tennis-coach"


def available():
    """(True, version) if the MediaPipe Tasks pose API can be imported, else (False, reason)."""
    try:
        import mediapipe  # noqa: F401
        from mediapipe.tasks.python import vision  # noqa: F401
        return True, getattr(mediapipe, "__version__", "?")
    except Exception as e:  # ImportError, missing native libs, ...
        return False, "%s: %s" % (type(e).__name__, e)


def ensure_model(variant="full"):
    """Path to the pose model; downloads it once into the cache (or use TTCOACH_MODEL_PATH)."""
    env = os.environ.get("TTCOACH_MODEL_PATH")
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError("TTCOACH_MODEL_PATH points to a missing file: %s" % p)
        return p
    if variant not in ("lite", "full", "heavy"):
        raise ValueError("model must be lite, full or heavy")
    d = cache_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    f = d / ("pose_landmarker_%s.task" % variant)
    if f.exists() and f.stat().st_size > 1_000_000:
        return f
    tmp = f.with_suffix(".part")
    req = urllib.request.Request(MODEL_URL.format(v=variant), headers={"User-Agent": "table-tennis-coach"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    if tmp.stat().st_size < 1_000_000:
        raise IOError("model download looks broken (%d bytes)" % tmp.stat().st_size)
    tmp.replace(f)
    return f


class Landmarker:
    def __init__(self, model_path, video=True, num_poses=3):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        self.video = video
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=num_poses, min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)
        self._lm = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, frame_bgr, ts_ms=0):
        """List of people: (xy px (33,2), visibility (33,), world metres (33,3))."""
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(image, int(ts_ms)) if self.video else self._lm.detect(image)
        people = []
        for k, lms in enumerate(res.pose_landmarks):
            xy = np.array([[l.x * w, l.y * h] for l in lms], float)
            vis = np.array([1.0 if getattr(l, "visibility", None) is None else float(l.visibility) for l in lms])
            wl = res.pose_world_landmarks[k] if k < len(res.pose_world_landmarks) else None
            world = np.array([[l.x, l.y, l.z] for l in wl], float) if wl else np.full((33, 3), np.nan)
            people.append((xy, vis, world))
        return people

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass


def _hip_center(xy):
    return (xy[23] + xy[24]) / 2.0


def _torso(xy):
    return float(np.linalg.norm((xy[11] + xy[12]) / 2.0 - _hip_center(xy)))


def pick_person(people, prev_center, prev_torso, player, frame_w, frame_h):
    """Index of the tracked player among detected people, or None."""
    if not people:
        return None
    if prev_center is not None:
        d = [np.linalg.norm(_hip_center(p[0]) - prev_center) for p in people]
        k = int(np.argmin(d))
        limit = 1.5 * prev_torso if prev_torso and np.isfinite(prev_torso) else 0.25 * max(frame_w, frame_h)
        return k if d[k] <= limit else None
    hips = [_hip_center(p[0]) for p in people]
    if player == "left":
        return int(np.argmin([h[0] for h in hips]))
    if player == "right":
        return int(np.argmax([h[0] for h in hips]))
    if player == "center":
        return int(np.argmin([abs(h[0] - frame_w / 2.0) for h in hips]))
    return int(np.argmax([_torso(p[0]) for p in people]))  # auto / largest


def track(frames, model_path, file_fps, slowmo, player="auto", num_poses=3, reacquire_s=0.5):
    """Run the landmarker over (index, frame) pairs and follow one player. Returns PoseSeries.

    frames: iterable of (source_frame_index, analysed_frame_bgr).
    Real time of frame i = i / file_fps / slowmo.
    """
    lm = Landmarker(model_path, video=True, num_poses=num_poses)
    T, F, IMG, VIS, WLD, PRES, NP = [], [], [], [], [], [], []
    prev_c, prev_t, missing_since = None, None, None
    w = h = 0
    try:
        for i, fr in frames:
            h, w = fr.shape[:2]
            t_file = i / float(file_fps)
            people = lm.detect(fr, ts_ms=int(round(t_file * 1000)))
            t_real = t_file / float(slowmo)
            if prev_c is not None and missing_since is not None and t_real - missing_since > reacquire_s:
                prev_c, prev_t = None, None  # lost for a while: pick again by the player rule
            k = pick_person(people, prev_c, prev_t, player, w, h)
            T.append(t_real)
            F.append(i)
            NP.append(len(people))
            if k is None:
                IMG.append(np.full((33, 2), np.nan))
                VIS.append(np.zeros(33))
                WLD.append(np.full((33, 3), np.nan))
                PRES.append(False)
                if missing_since is None:
                    missing_since = t_real
            else:
                xy, vis, world = people[k]
                IMG.append(xy)
                VIS.append(vis)
                WLD.append(world)
                PRES.append(True)
                prev_c, prev_t, missing_since = _hip_center(xy), _torso(xy), None
    finally:
        lm.close()
    n = len(T)
    fps = (np.median(np.diff(T)) ** -1) if n >= 2 else float(file_fps) * float(slowmo)
    return PoseSeries(t=np.array(T), frame_idx=np.array(F, int),
                      img=np.array(IMG).reshape(n, 33, 2), vis=np.array(VIS).reshape(n, 33),
                      world=np.array(WLD).reshape(n, 33, 3), present=np.array(PRES, bool),
                      n_people=np.array(NP, int), fps=float(fps), width=int(w), height=int(h))


def detect_image(frame_bgr, model_path, player="auto"):
    """Single photo: returns a 1-frame PoseSeries (fps=1) or None if nobody is found."""
    lm = Landmarker(model_path, video=False, num_poses=3)
    try:
        people = lm.detect(frame_bgr)
    finally:
        lm.close()
    h, w = frame_bgr.shape[:2]
    k = pick_person(people, None, None, player, w, h)
    if k is None:
        return None
    xy, vis, world = people[k]
    return PoseSeries(t=np.zeros(1), frame_idx=np.zeros(1, int), img=xy[None], vis=vis[None], world=world[None],
                      present=np.ones(1, bool), n_people=np.array([len(people)]), fps=1.0, width=w, height=h)

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ttcoach.landmarks import IDX, NAMES  # noqa: E402
from ttcoach.series import PoseSeries  # noqa: E402

# A right-handed player facing the camera, MediaPipe world convention:
# metres, hip centre at the origin, x to image right, y down, z away from the camera.
# Facing the camera means the player's right side is at -x and the nose is at -z.
BASE = {
    "nose": (0.0, -0.75, -0.10),
    "left_eye_inner": (0.02, -0.77, -0.09), "left_eye": (0.035, -0.77, -0.09), "left_eye_outer": (0.05, -0.77, -0.08),
    "right_eye_inner": (-0.02, -0.77, -0.09), "right_eye": (-0.035, -0.77, -0.09), "right_eye_outer": (-0.05, -0.77, -0.08),
    "left_ear": (0.08, -0.75, -0.02), "right_ear": (-0.08, -0.75, -0.02),
    "mouth_left": (0.025, -0.71, -0.09), "mouth_right": (-0.025, -0.71, -0.09),
    "left_shoulder": (0.20, -0.50, 0.0), "right_shoulder": (-0.20, -0.50, 0.0),
    "left_elbow": (0.25, -0.25, 0.0), "right_elbow": (-0.25, -0.25, 0.0),
    "left_wrist": (0.25, -0.05, -0.10), "right_wrist": (-0.30, -0.05, -0.15),
    "left_pinky": (0.26, -0.02, -0.11), "right_pinky": (-0.31, -0.02, -0.16),
    "left_index": (0.25, -0.01, -0.13), "right_index": (-0.30, -0.01, -0.18),
    "left_thumb": (0.24, -0.03, -0.12), "right_thumb": (-0.29, -0.03, -0.17),
    "left_hip": (0.12, 0.0, 0.0), "right_hip": (-0.12, 0.0, 0.0),
    "left_knee": (0.15, 0.45, -0.05), "right_knee": (-0.15, 0.45, -0.05),
    "left_ankle": (0.18, 0.90, 0.0), "right_ankle": (-0.18, 0.90, 0.0),
    "left_heel": (0.18, 0.93, 0.04), "right_heel": (-0.18, 0.93, 0.04),
    "left_foot_index": (0.19, 0.95, -0.12), "right_foot_index": (-0.19, 0.95, -0.12),
}

PX_PER_M = 400.0
CX, CY = 640.0, 500.0


def mirror(pose):
    """Mirror a pose left<->right (x -> -x, swap left/right names): a left-hander."""
    out = {}
    for name, (x, y, z) in pose.items():
        if name.startswith("left_"):
            other = "right_" + name[5:]
        elif name.startswith("right_"):
            other = "left_" + name[6:]
        else:
            other = name
        out[other] = (-x, y, z)
    return out


def build_series(poses, fps=100.0, vis=None):
    """poses: list of dicts name->(x,y,z) world metres (missing names fall back to BASE)."""
    n = len(poses)
    world = np.full((n, 33, 3), np.nan)
    img = np.full((n, 33, 2), np.nan)
    visib = np.full((n, 33), 0.99)
    for i, p in enumerate(poses):
        full = dict(BASE)
        full.update(p)
        for name, (x, y, z) in full.items():
            j = IDX[name]
            world[i, j] = (x, y, z)
            img[i, j] = (CX + PX_PER_M * x, CY + PX_PER_M * y)
    if vis:
        for name, v in vis.items():
            visib[:, IDX[name]] = v
    return PoseSeries(
        t=np.arange(n) / fps, frame_idx=np.arange(n), img=img, vis=visib, world=world,
        present=np.ones(n, bool), n_people=np.ones(n, int), fps=fps, width=1280, height=1000)


@pytest.fixture
def base_pose():
    return dict(BASE)


@pytest.fixture
def names():
    return NAMES

"""Container for a tracked person's pose over time (no MediaPipe import here)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class PoseSeries:
    t: np.ndarray          # (N,) real-time seconds of each processed frame
    frame_idx: np.ndarray  # (N,) index of the source frame in the video
    img: np.ndarray        # (N, 33, 2) pixel coordinates (x right, y down) in the analysed frame size; NaN = missing
    vis: np.ndarray        # (N, 33) visibility 0..1 (0 when missing)
    world: np.ndarray      # (N, 33, 3) metres, hip-centred, x right, y down, z away from camera; NaN = missing
    present: np.ndarray    # (N,) bool, a pose was found for the tracked player
    n_people: np.ndarray   # (N,) number of people detected in the frame
    fps: float             # real-time samples per second of this series
    width: int             # analysed frame width (px)
    height: int            # analysed frame height (px)

    def __len__(self):
        return len(self.t)

"""MediaPipe Pose landmark names (33 points) in model order."""

NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
IDX = {n: i for i, n in enumerate(NAMES)}

# Skeleton edges used for drawing.
EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_wrist", "left_index"), ("right_wrist", "right_index"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_foot_index"),
    ("left_ankle", "left_heel"), ("right_ankle", "right_heel"),
]


def side_names(hand):
    """Return (playing, other) prefixes for a 'right' or 'left' handed player."""
    if hand not in ("right", "left"):
        raise ValueError("hand must be 'right' or 'left'")
    return (hand, "left" if hand == "right" else "right")

"""Video probing and frame iteration (OpenCV only)."""
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


def warn(code, message):
    return {"code": code, "message": message}


@dataclass
class VideoInfo:
    path: str
    file_fps: float        # frames per second of file time
    frames: int            # frames actually decoded
    width: int
    height: int
    duration_s: float      # file time
    fps_source: str        # timestamps | metadata | assumed
    meta_fps: float
    meta_frames: int
    warnings: list = field(default_factory=list)


def _plausible(fps):
    return fps is not None and np.isfinite(fps) and 5.0 <= fps <= 1000.0


def open_capture(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("OpenCV 打不开这个视频：%s。可以先用手机或电脑把它转成 H.264 编码的 MP4 再试。" % path)
    return cap


def probe(path):
    """Decode the whole file once: real frame count, frame size, and fps from frame timestamps."""
    cap = open_capture(path)
    meta_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    meta_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ts, n, w, h = [], 0, 0, 0
    while cap.grab():
        if n == 0:
            ok, fr = cap.retrieve()
            if ok:
                h, w = fr.shape[:2]
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        n += 1
    cap.release()
    if n == 0:
        raise ValueError("视频里一帧都解不出来：%s。可以先转成 H.264 编码的 MP4 再试。" % path)

    warnings = []
    d = np.diff(np.asarray(ts, float))
    d = d[np.isfinite(d) & (d > 0)]
    ts_fps = 1000.0 / float(np.median(d)) if len(d) >= 5 else float("nan")
    if _plausible(ts_fps):
        fps, src = ts_fps, "timestamps"
    elif _plausible(meta_fps):
        fps, src = meta_fps, "metadata"
    else:
        fps, src = 30.0, "assumed"
        warnings.append(warn("FPS_UNKNOWN", "读不出这个视频的帧率，暂按每秒 30 帧计算；时间和速度类数字可能不准。如果知道真实帧率，请用 --capture-fps 告诉脚本。"))
    if not _plausible(meta_fps) or (_plausible(ts_fps) and abs(meta_fps - ts_fps) / ts_fps > 0.2):
        warnings.append(warn("FPS_METADATA_BAD", "视频文件里记录的帧率（%.1f）不可信，已改用帧时间戳算出的帧率 %.1f。" % (meta_fps, fps)))
    # only trust the container's frame count when its fps is plausible too (bogus metadata often has both wrong)
    if _plausible(meta_fps) and meta_frames > 0 and n < 0.9 * meta_frames and meta_frames - n > 10:
        warnings.append(warn("DECODE_STOPPED_EARLY", "文件声称有 %d 帧（约 %.0f 秒），但只解出 %d 帧（约 %.0f 秒）：视频可能损坏或编码特殊，后面的部分没有分析。" % (
            meta_frames, meta_frames / meta_fps, n, n / fps)))
    return VideoInfo(path=path, file_fps=float(fps), frames=n, width=int(w), height=int(h),
                     duration_s=n / float(fps), fps_source=src, meta_fps=meta_fps, meta_frames=meta_frames,
                     warnings=warnings)


def scale_for(width, height, max_side):
    if not max_side or max(width, height) <= max_side:
        return 1.0
    return float(max_side) / float(max(width, height))


def iter_frames(path, stride=1, first=0, last=None, keep=None):
    """Yield (index, frame_bgr) for frames first..last (inclusive) every `stride` frames.

    If `keep` (a set of indices) is given, yield exactly those frames instead.
    """
    cap = open_capture(path)
    i = 0
    try:
        while cap.grab():
            if last is not None and i > last:
                break
            take = (i in keep) if keep is not None else (i >= first and (i - first) % stride == 0)
            if take:
                ok, fr = cap.retrieve()
                if ok:
                    yield i, fr
            i += 1
    finally:
        cap.release()

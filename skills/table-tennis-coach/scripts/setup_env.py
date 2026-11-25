#!/usr/bin/env python3
"""Create a private Python environment for analyze_video.py and install what it needs.

  python3 setup_env.py           # create or update; prints TTCOACH_PYTHON=<python to use>
  python3 setup_env.py --check   # only report what is installed

Uses only the standard library, so any Python 3.9+ can run it. The environment lives in the
user cache folder (~/.cache/table-tennis-coach/venv, or %LOCALAPPDATA%\\table-tennis-coach\\venv),
not inside the skill folder, so plugin updates do not wipe it.
Install order: mediapipe (pose) -> mediapipe<1 (older platforms, e.g. Intel Macs) -> OpenCV only
(frames-only mode: key frames without skeleton angles).
"""
import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cache_dir():
    env = os.environ.get("TTCOACH_CACHE")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "table-tennis-coach"
    return Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "table-tennis-coach"


def venv_python(d):
    return d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd, quiet=True):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if not quiet and p.returncode != 0:
        sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:])
    return p


def probe(py):
    cv = run([str(py), "-I", "-c", "import cv2, numpy; print(cv2.__version__)"])
    mp = run([str(py), "-I", "-c", "import mediapipe; from mediapipe.tasks.python import vision; print(mediapipe.__version__)"])
    return (cv.stdout.strip() if cv.returncode == 0 else None), (mp.stdout.strip() if mp.returncode == 0 else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-model", action="store_true", help="do not pre-download the pose model")
    a = ap.parse_args(argv)
    if sys.version_info < (3, 9):
        print("Python 3.9 or newer is needed (this is %s)." % sys.version.split()[0])
        return 2
    d = cache_dir() / "venv"
    py = venv_python(d)
    if a.check:
        if not py.exists():
            print("not installed: run setup_env.py")
            return 1
        cv, mp = probe(py)
        print("opencv=%s mediapipe=%s" % (cv, mp))
        print("TTCOACH_PYTHON=%s" % py)
        return 0 if cv else 1

    if not py.exists():
        print("creating %s ..." % d)
        venv.EnvBuilder(with_pip=True, clear=False).create(str(d))
    pip = [str(py), "-m", "pip", "install", "--disable-pip-version-check", "-q"]
    run(pip + ["--upgrade", "pip"])
    mode = "pose"
    if run(pip + ["mediapipe", "numpy"]).returncode != 0:
        print("latest mediapipe has no package for this computer; trying mediapipe<1 ...")
        if run(pip + ["mediapipe<1", "numpy"]).returncode != 0:
            mode = "frames-only"
            print("mediapipe is not available here; installing OpenCV only (frames-only mode).")
            run(pip + ["opencv-python-headless", "numpy"], quiet=False)
    cv, mp = probe(py)
    if not cv:
        print("OpenCV could not be imported; see the pip output above.")
        return 1
    if mode == "pose" and not mp:
        mode = "frames-only"
    if mode == "pose" and not a.no_model:
        r = run([str(py), "-I", "-c", "import sys; sys.path.insert(0, %r); from ttcoach import pose; print(pose.ensure_model('full'))" % str(HERE)])
        print("pose model: %s" % (r.stdout.strip() or "download failed (it will be retried on first use)"))
    print("opencv=%s mediapipe=%s mode=%s" % (cv, mp, mode))
    print("TTCOACH_PYTHON=%s" % py)
    return 0


if __name__ == "__main__":
    sys.exit(main())

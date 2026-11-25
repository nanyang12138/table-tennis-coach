#!/usr/bin/env python3
"""Table tennis technique analyser: video or photo -> key frames, skeleton, angles, summary.md.

Run it with the Python that scripts/setup_env.py prints (TTCOACH_PYTHON). Typical use:

  analyze_video.py my_forehand.mp4 --hand right                 # auto: candidates + timeline pages
  analyze_video.py my_forehand.mp4 --timeline 3.0 5.0 0.04      # dense frames of one window
  analyze_video.py my_forehand.mp4 --strokes "3.10,3.34,3.52,3.70,4.05"   # measure phases you chose
  analyze_video.py my_forehand.mp4 --at 3.52,3.70                # angles at single moments
  analyze_video.py grip.jpg --hand right                          # a photo (stance, grip, posture)

All times are in seconds of the video file, the same numbers printed as t=... on the images.
For slow-motion files exported at normal speed add --capture-fps 120 (or 240).
Outputs go to --out (default ./ttcoach_<video name>); follow-up runs with the same --out reuse
the cached pose tracking.
"""
import argparse
import contextlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@contextlib.contextmanager
def native_stderr_to(path):
    """MediaPipe's C++ layer prints a lot to stderr; send it to a log file instead."""
    try:
        sys.stderr.flush()
        saved = os.dup(2)
    except (OSError, ValueError):
        yield
        return
    with open(path, "ab") as log:
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)


def default_out(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem) or "video"
    return os.path.join(os.getcwd(), "ttcoach_" + safe)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Table tennis technique analyser (see SKILL.md).")
    ap.add_argument("input", help="video (mp4/mov/webm/...) or photo (jpg/png/webp)")
    ap.add_argument("--hand", choices=["right", "left"], default="right", help="playing hand (default right)")
    ap.add_argument("--out", help="output folder (default ./ttcoach_<name>)")
    ap.add_argument("--capture-fps", type=float, help="real recording fps of a slow-motion file exported at normal speed")
    ap.add_argument("--start", type=float, help="analyse from this file time (s)")
    ap.add_argument("--end", type=float, help="analyse up to this file time (s)")
    ap.add_argument("--player", choices=["auto", "largest", "left", "right", "center"], default="auto",
                    help="which person to follow when several are in the frame (default: largest)")
    ap.add_argument("--model", choices=["lite", "full", "heavy"], default="full", help="pose model size")
    ap.add_argument("--frames-only", action="store_true", help="skip pose estimation (frames and timeline only)")
    ap.add_argument("--max-side", type=int, default=960, help="analysis resolution (longest side, px)")
    ap.add_argument("--timeline", nargs="+", type=float, metavar="T",
                    help="START END [STEP]: dense frames for one window (file seconds)")
    ap.add_argument("--strokes", help='phase times per stroke: "ready,backswing_end,contact,finish,recovery; ..."')
    ap.add_argument("--at", help="comma-separated times to measure single moments")
    a = ap.parse_args(argv)
    if a.timeline is not None and len(a.timeline) not in (2, 3):
        ap.error("--timeline needs START END [STEP]")
    if sum(x is not None for x in (a.timeline, a.strokes, a.at)) > 1:
        ap.error("use only one of --timeline, --strokes, --at per run")

    out = os.path.abspath(a.out or default_out(a.input))
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "run.log")
    try:
        from ttcoach import pipeline
        with native_stderr_to(log):
            res = pipeline.run(a.input, out, hand=a.hand, capture_fps=a.capture_fps, start=a.start, end=a.end,
                               player=a.player, model=a.model, frames_only=a.frames_only, max_side=a.max_side,
                               timeline=a.timeline, strokes=a.strokes, at=a.at)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    except ImportError as e:
        print("ERROR: missing dependency (%s). Run scripts/setup_env.py and use the python it prints." % e,
              file=sys.stderr)
        return 2
    except Exception:
        with open(log, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        return 3

    print("out_dir: %s" % res["out_dir"])
    print("summary: %s" % os.path.join(res["out_dir"], "summary.md"))
    print("task: %s | mode: %s | hand: %s" % (res.get("task"), res["mode"], res["hand"]))
    if res.get("kind") == "video":
        v = res["video"]
        print("video: %.1f fps real (file %.1f, slowmo x%.1f), %d frames, %sx%s" % (
            v["real_fps"], v["file_fps"], v["slowmo_factor"], v["frames"], v["width"], v["height"]))
    if res.get("strokes"):
        print("strokes: %d (%s)" % (len(res["strokes"]), ", ".join(
            "%d@%.2fs %s" % (s["index"], s.get("t_contact_file_s", s["t_contact_s"]), s["side_guess"]) for s in res["strokes"])))
    for w in res.get("warnings", []):
        print("WARNING %s: %s" % (w["code"], w["message"]))
    print("files: %s" % ", ".join(res.get("files", [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Write analysis.json and a short Chinese summary.md for Claude to read first."""
import json
import math
from pathlib import Path

import numpy as np

SIDE_CN = {"forehand": "正手", "backhand": "反手", "unclear": "不确定"}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _f(v, fmt="%.0f"):
    return "-" if v is None else fmt % v


def pick_examples(strokes, task=None):
    """Per side: the most typical stroke and the one most unlike the others (by z-score of key metrics).

    Auto-detected candidates are not verified yet, so nothing is recommended for them (task='auto')."""
    if task == "auto":
        return []
    keys = [("elbow_angle_deg", "contact"), ("knee_min_deg", "playing"), ("shoulder_turn_deg", None),
            ("contact_height_level", None), ("forward_ms", None)]
    picks = []
    for side in ("forehand", "backhand", "unclear"):
        group = [s for s in strokes if s["side_guess"] == side and s["metrics"]["quality"]["reliable"]]
        if not group:
            group = [s for s in strokes if s["side_guess"] == side]
        if not group:
            continue
        if len(group) < 3:
            picks.append((group[0]["index"], "%s示例" % SIDE_CN[side]))
            continue
        cols = []
        for k, sub in keys:
            vals = [s["metrics"].get(k) if sub is None else (s["metrics"].get(k) or {}).get(sub) for s in group]
            if all(v is not None for v in vals):
                arr = np.array(vals, float)
                sd = arr.std() or 1.0
                cols.append(np.abs(arr - np.median(arr)) / sd)
        if not cols:
            picks.append((group[0]["index"], "%s示例" % SIDE_CN[side]))
            continue
        score = np.sum(cols, axis=0)
        picks.append((group[int(np.argmin(score))]["index"], "%s里最典型的一板" % SIDE_CN[side]))
        far = group[int(np.argmax(score))]["index"]
        if far != picks[-1][0]:
            picks.append((far, "%s里和其他板差别最大的一板" % SIDE_CN[side]))
    return picks


def _static_lines(m):
    """Readable lines for one-frame measurements (photo or --at); None means hidden / not measurable."""
    def f(v, fmt="%.0f"):
        return "看不清" if v is None else fmt % v
    e, k = m.get("elbow_angle_deg") or {}, m.get("knee_angle_deg") or {}
    q = m.get("quality") or {}
    return ["- 肘角（持拍臂 / 另一臂）：%s / %s 度" % (f(e.get("playing")), f(e.get("other"))),
            "- 膝角（持拍侧 / 另一侧）：%s / %s 度" % (f(k.get("playing")), f(k.get("other"))),
            "- 躯干偏离竖直：%s 度；站位宽度（两踝 ÷ 两肩）：%s" % (f(m.get("trunk_lean_deg")), f(m.get("stance_width_ratio"), "%.2f")),
            "- 持拍手腕高度：%s；非执拍手腕高度：%s（0 = 髋，1 = 肩）" % (
                f(m.get("wrist_height_level"), "%.2f"), f(m.get("free_arm_height_level"), "%.2f")),
            "- 看不清的关节：%s" % ("、".join(q.get("low_visibility_parts") or []) or "无")]


def write(out_dir, result):
    out = Path(out_dir)
    prefix = "measured" if result.get("task") == "strokes" else "stroke"
    data = _clean(result)
    (out / "analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    v = data.get("video", {})
    lines = ["# 视频分析摘要（脚本自动生成，判断由 Claude 对照技术卡完成）", ""]
    if data.get("kind") == "image":
        lines.append("- 照片：%s｜尺寸 %s×%s｜模式：%s｜持拍手：%s" % (
            v.get("path"), v.get("width"), v.get("height"), data.get("mode"), data.get("hand")))
    else:
        lines.append("- 文件：%s｜时长 %.1f 秒（文件时间）｜文件帧率 %.1f（来源：%s）" % (
            v.get("path"), v.get("duration_s") or 0, v.get("file_fps") or 0, v.get("fps_source")))
        lines.append("- 真实拍摄帧率 %.0f（慢放倍数 %.1f）｜分析用 %.0f 帧/秒（每 %d 帧取 1 帧）｜分辨率 %s×%s" % (
            v.get("real_fps") or 0, v.get("slowmo_factor") or 1, v.get("analysis_fps") or 0, v.get("stride") or 1,
            v.get("width"), v.get("height")))
        lines.append("- 模式：%s｜持拍手：%s｜跟踪对象：%s" % (data.get("mode"), data.get("hand"), data.get("player")))
    strokes = data.get("strokes", [])
    if data.get("kind") != "image" and data.get("task") in ("auto", "strokes", None):
        n_side = {k: sum(1 for s in strokes if s["side_guess"] == k) for k in SIDE_CN}
        how = "按你给的时刻测量了" if data.get("task") == "strokes" else "自动找到候选"
        lines.append("- %s %d 板：正手 %d，反手 %d，不确定 %d（正反手是按击球时手的位置粗判的，以图片为准）" % (
            how, len(strokes), n_side["forehand"], n_side["backhand"], n_side["unclear"]))
        if data.get("task") != "strokes" and strokes:
            lines.append("- 自动候选可能把引拍或收拍当成一板，也可能漏板：先看 timeline_*.jpg 和每个 stroke_NN_strip.jpg 核对，再用 --strokes 按你定的时刻测量。")
    lines.append("")
    if data.get("warnings"):
        lines.append("## 警告（先读这一节）")
        for w in data["warnings"]:
            lines.append("- [%s] %s" % (w["code"], w["message"]))
        lines.append("")
    if strokes:
        lines.append("## 每一板的关键数字")
        lines.append("| # | 击球时刻 s | 粗判 | 引拍/前挥/随挥 ms | 击球肘角 | 持拍侧膝最小角 | 转肩 | 转髋 | 重心转移 | 击球高度 | 挥拍仰角 | 可靠 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in strokes:
            m = s["metrics"]
            lines.append("| %d | %.2f | %s | %s/%s/%s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                s["index"], s["t_contact_s"], SIDE_CN.get(s["side_guess"], s["side_guess"]),
                _f(m.get("backswing_ms")), _f(m.get("forward_ms")), _f(m.get("follow_ms")),
                _f((m.get("elbow_angle_deg") or {}).get("contact")), _f((m.get("knee_min_deg") or {}).get("playing")),
                _f(m.get("shoulder_turn_deg")), _f(m.get("hip_turn_deg")),
                _f((m.get("weight_shift") or {}).get("delta"), "%.2f"), _f(m.get("contact_height_level"), "%.2f"),
                _f((m.get("swing_path") or {}).get("angle_deg")),
                "是" if m["quality"]["reliable"] else "否"))
        lines.append("")
        lines.append("角度单位：度；击球高度 0 = 髋、1 = 肩；重心转移 0 = 持拍侧脚、1 = 另一只脚（看 delta 的正负）。各指标的含义和可信度见 references/analysis/metrics-reference.md。")
        lines.append("")
        lines.append("## 建议先看的图")
        if data.get("task") in ("auto", None):
            lines.append("- 先看 timeline_01.jpg 起的各页：弄清整段有几板、各在什么时间、是什么技术。")
            lines.append("- 再逐个看 stroke_NN_strip.jpg：核对这个候选是不是真的一板、标的击球时刻对不对。")
        for idx, why in data.get("examples", []):
            lines.append("- %s_%02d_sheet.jpg、%s_%02d_zoom.jpg、%s_%02d_strip.jpg：%s" % (prefix, idx, prefix, idx, prefix, idx, why))
        lines.append("")
    if data.get("photo_metrics"):
        lines.append("## 照片里的静态测量")
        lines.extend(_static_lines(data["photo_metrics"]))
        lines.append("")
        lines.append("图：photo_annotated.jpg（带骨架）、photo_hand_zoom.jpg（持拍手放大，用来看握拍和拍面）。")
        lines.append("")
    if data.get("task") == "at":
        lines.append("## 指定时刻的测量")
        for a in data.get("at", []):
            lines.append("### 时刻 %d：t=%.2fs（第 %d 帧）→ at_%02d.jpg、at_%02d_zoom.jpg" % (
                a["index"], a["t_file"], a["frame"], a["index"], a["index"]))
            if not a.get("pose_found"):
                lines.append("- 这一帧没认出人（被遮挡、出画或跟踪丢失）。")
            else:
                lines.extend(_static_lines(a["metrics"]))
        lines.append("")
    if data.get("task") == "timeline":
        tl = data.get("timeline", {})
        lines.append("## 时间轴画面")
        lines.append("- 范围 %.2f–%.2f 秒，步长 %.3f 秒，共 %s 帧：见 timeline_XX.jpg（每张 20 帧，带骨架和 t= 时间）。" % (
            tl.get("start", 0), tl.get("end", 0), tl.get("step", 0), tl.get("frames", "?")))
        lines.append("")
    files = data.get("files", [])
    if files:
        lines.append("## 生成的文件")
        for f in files:
            lines.append("- %s" % f)
        lines.append("")
    lines.append("## 提醒")
    lines.append("- 这些数字来自单机位视频的骨架估计，受拍摄角度影响；拍面角度、握拍和击球点要看图判断。")
    lines.append("- 先看警告，再看图，最后看数字；图和数字对不上时，以图为准，并说明原因。")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / "summary.md"

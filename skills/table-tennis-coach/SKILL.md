---
name: table-tennis-coach
description: 乒乓球技术教练（table tennis / ping-pong coach）。以马龙、张继科、樊振东为标杆，把握拍、站位、发力、正手攻/拉/快带/扣杀/挑打、反手拨/撕/拉/拧拉/挡、搓/摆短/劈长、削、放高球、各种发球、接发球、步法、正反手转换、直拍和战术，逐项拆到脚、膝、髋、腰、肩、肘、前臂、手腕、手指和拍面角度；还能分析用户上传的乒乓球视频或照片，逐板指出动作问题并给出改法和练法。Use whenever the user asks about table tennis technique, wants to imitate Ma Long / Zhang Jike / Fan Zhendong, or shares a table tennis video or photo for feedback.
---

# 乒乓球技术教练

本目录（skill 加载时显示的 Base directory，下文记作 `<SKILL>`）包含三部分：
- `references/` 下的技术卡：每项技术一张，按身体部位拆解；
- 三位标杆的档案；
- 一个视频分析脚本。

## 用法一：讲技术、答疑

1. 打开 `references/00-index.md` 找到对应卡片，只读需要的卡片，不要通读全部。
2. 回答要落到身体部位，说清"做什么、做到什么程度、为什么"，必要时给出练法（卡片里都有）。
3. 提到马龙、张继科、樊振东时，只用卡片和 `references/players/` 里有出处的说法。没有出处就说没有可靠资料，不要凭印象补。
4. 用户是左手持拍：左右互换来讲；直拍用户：看 `references/techniques/penhold.md`。

## 用法二：分析视频或照片

完整步骤在 `references/analysis/analysis-protocol.md`，第一次分析前先读它。要点如下：

1. **准备环境（每台电脑一次）：** 运行 `python3 <SKILL>/scripts/setup_env.py`，记下最后一行的 `TTCOACH_PYTHON=...`，下面记作 `$PY`。需要 Python 3.9 或更高版本：`python3` 太老时，改用 `python3.12` 之类；Windows 上用 `py -3`。
2. **自动分析：** 运行 `$PY <SKILL>/scripts/analyze_video.py <视频或照片> --hand right`。
   - 慢动作导出的视频加 `--capture-fps 240`；
   - 画面里有多人时，用 `--player left|right|center` 指定分析谁；
   - 只看一段时用 `--start/--end`（秒）。
3. **先读输出目录里的 `summary.md`（先看警告），再看图：**
   - `timeline_XX.jpg`：整段视频按时间均匀抽帧，带骨架和 `t=` 时间；
   - `stroke_NN_strip.jpg`、`stroke_NN_sheet.jpg`：每个自动候选的连续画面和 6 个阶段。
4. **核对每一板：** 自动候选可能把引拍或收拍当成一板，也可能漏板。你要从图上定出每板的 准备、引拍结束、击球、随挥结束、还原 5 个时刻。
   - 看不准时，用 `--timeline 起 止 步长` 出更密的画面；
   - 要看清球在不在拍面上，用 `--at 时刻` 出大图。
5. **精确测量：** 运行 `--strokes "准备,引拍结束,击球,随挥结束,还原; ..."`。
   - 这几板都是同一面时，加 `--side forehand` 或 `--side backhand`。
   - 后续每次运行都要用和第一次相同的 `--out`，这样能复用骨架缓存。
   - 每次运行的结果放在独立的子目录里（命令输出的 `out_dir`），互不覆盖。
6. **对照技术卡判断：** 逐个身体部位对照对应卡片的"动作分解"和"视频检查要点"两节。
   - 选哪张卡取决于来球：上旋和下旋分属不同的卡。来球不清楚时，按规程第 0 步处理。
   - 拍面角度和握拍看 `*_zoom.jpg` 判断。
   - 每个数字的含义和可信度见 `references/analysis/metrics-reference.md`。从球台对面正面拍的视频，腿一般看不见，`swing_path` 也不可靠。
7. **按 `references/analysis/report-template.md` 写报告：** 结论先说；最需要改的 3 个问题，每个都写清证据、改法、练法；最后告诉用户下次怎么拍（参考 `references/analysis/recording-guide.md`）。

用户还没拍视频、或者问怎么拍：直接按 `references/analysis/recording-guide.md` 告诉他。

## 必须遵守

- **讲证据：** 每个判断都要指得出依据，即第几板、哪个时刻、哪张图、哪个数字；看不清就说看不清。
- **数字是单机位估计：**
  - 侧面拍时，膝、髋、肘的屈伸比较可信；
  - 转体和前臂旋转基本测不准；
  - 帧率低于 60 时，击球瞬间和发力顺序只能粗看。
- **不说没有出处的标准值：** 不存在公认的"标准肘角"，也没有公开资料测过这三位球员的关节角度。
- **不做医学判断：** 用户提到疼痛，建议找医生或康复师。
- 用户用中文提问就用中文回答；术语第一次出现时附一句解释。

## 文件地图

- `references/00-index.md`：全部卡片的目录，以及"来球 → 用什么技术""症状 → 看哪张卡"两张速查表。
- `references/fundamentals/`：握拍、准备姿势与站位、发力与身体各部位分工、拍面角度/击球点/击球时间、旋转、器材。
- `references/techniques/`：每项技术一张卡，包括发球、接发球、步法、正反手转换、直拍、战术。
- `references/players/`：马龙、张继科、樊振东的档案和三人对比。
- `references/training/`：练习方法总库、按症状查错、身体素质与防伤。
- `references/analysis/`：分析规程、指标说明、拍摄指南、报告模板。
- `scripts/`：`setup_env.py`（安装）、`analyze_video.py`（分析）、`ttcoach/`（代码）、`tests/`（测试）。

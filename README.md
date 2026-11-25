# table-tennis-coach · 乒乓球技术教练（Claude Code Skill）

以**马龙、张继科、樊振东**为标杆的乒乓球技术 skill，主要做两件事：

1. **讲技术。** 每项技术一张卡，拆到脚、膝、髋、腰、肩、肘、前臂、手腕、手指和拍面角度。范围覆盖握拍、站位、发力、正手、反手、台内、防守、发球、接发球、步法、转换、直拍、战术和双打。每张卡都有常见错误、练习方法和视频检查要点。
2. **看视频。** 你给一段挥拍视频（或一张照片），Claude 会：
   - 把视频拆成帧，画出人体骨架；
   - 量出肘、膝的角度，以及重心转移、击球点高度、挥拍方向；
   - 对照技术卡逐个身体部位判断；
   - 指出最需要改的 3 个问题，告诉你怎么改、怎么练。

> 英文简介：A Claude Code skill that teaches every table tennis technique in body-part detail (benchmarking Ma Long, Zhang Jike and Fan Zhendong) and analyses your stroke videos with pose estimation (MediaPipe) plus Claude's own reading of the frames.

## 安装

需要 [Claude Code](https://claude.com/claude-code)（命令行、桌面应用或 IDE 插件都可以）。

**方法一：作为插件安装（推荐）。** 在 Claude Code 里输入：

```
/plugin marketplace add nanyang12138/table-tennis-coach
/plugin install table-tennis-coach@table-tennis-coach
```

**方法二：手动拷贝。**

```bash
git clone https://github.com/nanyang12138/table-tennis-coach.git
mkdir -p ~/.claude/skills
cp -r table-tennis-coach/skills/table-tennis-coach ~/.claude/skills/
```

视频分析需要电脑上有 Python 3.9 或更高版本。第一次分析视频时，Claude 会运行 `scripts/setup_env.py`，自动建立一个独立的 Python 环境，并安装 MediaPipe 和 OpenCV（需要联网，约几百 MB）。
- 如果你的电脑没有 MediaPipe 的安装包（例如 Intel 芯片的 Mac 装不上新版），会自动退回到"只看图"模式：照样抽帧给 Claude 看，但没有角度数据。

## 怎么用

直接用中文问就行，例如：

- "正手拉下旋的动作要领是什么？从脚到手腕都讲一下。"
- "我想学张继科的反手拧拉，怎么练？"
- "樊振东的反手和马龙的反手有什么区别？"
- "帮我看看这个视频：~/Movies/forehand.mov。我右手横拍，打了两年，用 iPhone 240 帧慢动作从侧面拍的，拉球老下网。"
- "这是我的握拍照片，对不对？"

分析视频时，Claude 会：
1. 先跑自动分析：抽帧、画骨架、找出候选的每一板；
2. 看图确认每一板的引拍、击球、收拍各在哪个时刻，再让脚本在这些时刻精确测量；
3. 对照技术卡写报告，包括结论、分部位评估、最需要改的 3 个问题、练习计划，以及下次怎么拍。

## 怎么拍视频，分析才准

详见 `skills/table-tennis-coach/references/analysis/recording-guide.md`。最重要的三条：

1. **用慢动作拍（120 或 240 帧/秒）。** 普通 30 帧视频抓不到击球那一下。
2. **全身入画，用三脚架固定，不要跟拍、不要变焦。**
3. **选对机位：** 正手拉、搓球、发球从侧面拍，反手从正面或斜前方 45° 拍。

## 能做什么、不能做什么

- 单机位视频只能**估计**角度：侧面拍时，膝、髋、肘的屈伸比较可信；躯干转动、前臂旋转单机位测不准，报告里会说明。
- 拍面角度、握拍由 Claude 看手部放大图判断，不是由程序测量的。
- 三位球员的技术特点都附有出处。公开资料里没有这三位的关节角度测量数据，所以这个 skill 不会说"马龙击球时肘角多少度"之类的话。
- 不做医学判断：有疼痛请找医生或康复师。

## 目录结构

```
.claude-plugin/              插件与 marketplace 配置
skills/table-tennis-coach/
  SKILL.md                   入口：Claude 怎么用这个 skill
  references/
    00-index.md              全部卡片目录 + 速查表
    fundamentals/            握拍、站位、发力、拍面/击球点/击球时间、旋转、器材
    techniques/              每项技术一张卡（含发球、接发球、步法、转换、直拍、战术、双打）
    players/                 马龙、张继科、樊振东档案与对比
    training/                练习方法、按症状查错、身体素质与防伤
    analysis/                分析规程、指标说明、拍摄指南、报告模板
  scripts/
    setup_env.py             一键安装分析环境
    analyze_video.py         视频/照片分析
    ttcoach/                 分析代码
    tests/                   测试
```

## 开发者：运行测试

```bash
cd skills/table-tennis-coach/scripts
python -m pytest tests            # 单元测试（不需要视频）
TTCOACH_TEST_VIDEOS=<放测试视频的目录> python -m pytest tests   # 加上真实视频的集成测试
```

集成测试用到的是 Wikimedia Commons 上公开授权的乒乓球视频，没有放进仓库：
- Topspin Drive.ogv、Reves topspin.ogv、Bloopers Tenis de mesa.ogv（Blasco88，CC BY-SA 3.0）；
- Table tennis.webm（Evworo，CC BY-SA 3.0）。

## 资料来源

每张技术卡和球员档案的末尾都列有参考来源，主要包括：
- 同行评审的生物力学论文；
- 新华社、央视等媒体报道；
- ITTF 规则；
- 中英文教学资料。

查不到可靠出处的说法没有写进卡片。

## 许可证

MIT

# 指标说明（analyze_video.py 输出的每个数字是什么、能信几分）

> 本页和脚本一一对应：`analysis.json` 里每板的 `metrics` 字段、`summary.md` 表格里的列，都在这里解释。
> 读数前先记住三条：
> 1. **单机位视频只能估计。** 侧面拍（镜头对着球员身侧）时，膝、髋、肘的屈伸最可信；正面拍时这些角度误差大。躯干/骨盆转动、肩内外旋、前臂旋前旋后属于"离开画面平面的转动"，单机位基本测不准（研究：单机位 MediaPipe 测前臂旋前旋后 RMSE>30°，见文末 [Shrestha2026]）。
> 2. **快速挥拍时误差最大。** 双摄像头的研究里，正手拉前挥阶段肘角平均误差约 2.6°–7.1°，引拍阶段约 1.1°–1.8°；单机位只会更差 [Cao2026]。
> 3. **不要拿这些数字和论文里的数字直接比大小。** 论文多用惯性传感器或多摄像头 3D 系统，角度定义也不同（见每项说明）。最可靠的用法是：**同一个人、同一机位，前后对比**；或者同一段视频里各板之间对比（看稳定性）。

## 一、时间与阶段

| 字段 | 含义 | 怎么算 | 可信度 |
|---|---|---|---|
| `phases.*.t_file` | 六个阶段的时刻：ready 准备、backswing_end 引拍结束、mid_forward 前挥中段、contact 击球（估计）、finish 随挥结束、recovery 还原 | 自动模式：按手相对肩部的速度切分；`--strokes` 模式：用你看图定下的时刻 | 自动模式只是候选，必须看图核对；`--strokes` 模式以你定的时刻为准 |
| `backswing_ms` | 准备 → 引拍结束 | 帧差 ÷ 真实帧率 | 帧率 ≥120 才有意义；25–30 帧时误差 ±40 ms 量级 |
| `forward_ms` | 引拍结束 → 击球 | 同上 | 同上。研究参考：一项 7 名高水平选手的惯性传感器研究中，"引拍结束→击球"约 0.1 s，随挥约 0.2 s，回到准备约 0.4 s [B2020b]。各研究分期定义不同，不能当标准值 |
| `follow_ms` | 击球 → 随挥结束 | 同上 | 同上 |
| `t_contact_s` / `t_contact_file_s` | 击球时刻（真实时间 / 文件时间） | 自动模式取手速峰值；手速峰值与真正触球可能差 1–2 帧 | 以手部放大图为准 |

## 二、关节角度（单位：度）

**本工具的角度都是"夹角"：180° = 完全伸直，越小越弯。** 很多论文用"屈曲角"（0° = 伸直），换算大约是：夹角 ≈ 180° − 屈曲角。

| 字段 | 含义 | 机位 | 研究里观察到的范围（仅供对照，样本小、方法不同） |
|---|---|---|---|
| `elbow_angle_deg.backswing_end/contact/finish` | 持拍臂肩-肘-腕夹角 | 侧面最好；手臂朝向/背向镜头时不准（脚本会在 quality.notes 里提示 3D 与 2D 差异过大） | 正手拉（7 名波兰国际水平男子，惯性传感器）：拉上旋来球时肘屈曲 引拍结束约 44°、击球约 47°、随挥结束约 72°，即夹角约 136° → 133° → 108°；拉下旋时击球屈曲约 26°（夹角约 154°）。组内标准差 30°–50°，**个人差异极大，不存在统一"标准肘角"** [B2020a]。反手拉（8 名精英女子）：重心最低时肘屈曲约 103°–105°、击球约 75°–78°，即前挥中伸肘约 27° [Xing2022] |
| `knee_angle_deg.playing/other.*` | 髋-膝-踝夹角（持拍侧 / 另一侧） | 侧面最好；腿被球台挡住时脚本给 null | 正手拉：持拍侧膝屈曲 引拍结束约 50°（拉上旋）/约 63°（拉下旋），击球时约 50°/66°（夹角约 130°/114°）；膝是变异最小的部位之一 [B2020a] |
| `knee_min_deg.playing/other` | 引拍结束到击球之间膝夹角的最小值（蹲得最低时） | 同上 | 同上 |
| `trunk_lean_deg.backswing_end/contact` | 躯干（髋中点→肩中点）偏离竖直的角度，含前倾和侧倾 | 侧面看前倾最准 | 无可靠常模；用于同一个人前后对比（例如击球时是否站直了） |
| `shoulder_turn_deg` | 引拍结束→随挥结束，肩线在水平面的转动；**正值 = 持拍侧肩向前转**（正手的正常方向） | 正面或斜 45°；单机位深度方向噪声大 | **低可信**。研究只证实"骨盆转动角速度与击球拍速正相关"[I2018]、"高水平选手下躯干转动对拍速贡献更大"[IK2009]，没有可直接对照的转动角度标准 |
| `hip_turn_deg` | 同上，髋线的转动 | 同上 | **低可信**；同上 |
| `x_factor_deg` | 引拍结束时肩线与髋线的夹角（肩髋分离） | 同上 | **低可信**；乒乓球的肩髋分离角目前没有同行评审的数值资料 |
| `shoulder_width_ratio.backswing_end/contact` | 画面中两肩距离 ÷ 躯干长度（2D）。正面拍时，转体越多这个数越小 | 正面拍时有用 | 只做同一视频内比较 |

## 三、位置与重心

| 字段 | 含义 | 怎么解读 |
|---|---|---|
| `contact_height_level` | 击球时持拍手腕的高度：0 = 髋部高度，1 = 肩部高度，>1 高于肩，<0 低于髋。髋被球台挡住时，髋线按"肩线 + 躯干长度"估计，quality.notes 会注明 | 结合技术卡里"击球点"一节判断；例如拉下旋来球通常击球点比攻球低 |
| `finish_height_level` / `head_level_finish` | 随挥结束时手腕高度 / 鼻子高度（同一标尺） | 手腕高度和鼻子高度对比，可以判断"收拍到额头/眼前"还是"收拍过低" |
| `free_arm_height_level` | 击球时非执拍手腕的高度 | 非执拍手掉到髋以下往往说明身体平衡和转体配合差（定性） |
| `contact_forward_m` | 击球时手腕在髋中点前方多远（米，3D 估计） | 正值越大击球点越靠前；单机位估计，只做同一视频内比较 |
| `contact_lateral_m` | 击球时手腕在肩中点持拍侧多远（米，沿两肩连线方向） | 正手通常为正（在身体持拍侧），反手接近 0 或为负（在身体前方或另一侧）；脚本用它粗判正反手（>0.15 正手，<0.08 反手，中间为不确定） |
| `elbow_body_gap_m` | 击球时肘离躯干中线多远（米） | 夹得太紧（很小）或甩得太开（很大）都值得看图确认 |
| `weight_shift.backswing_end/finish/delta` | 髋中点在两脚之间的位置：0 = 在持拍侧脚上方，1 = 在另一只脚上方；delta = 随挥结束 − 引拍结束 | 正手攻、拉时重心通常由持拍侧脚移到另一侧脚，delta 应为正（定性要点见 ../fundamentals/power-and-body.md）；脚被挡住时为 null |
| `stance_width_ratio` | 准备时两踝距离 ÷ 两肩宽度（3D） | 准备姿势的站位宽度，同一个人前后对比 |

## 四、挥拍路径与速度

| 字段 | 含义 | 怎么解读 |
|---|---|---|
| `swing_path.up/horiz/angle_deg` | 引拍结束→随挥结束，手腕相对髋中点的位移：向上分量、水平分量（单位：躯干长度），以及仰角（向上为正） | 拉弧圈（尤其拉下旋）仰角明显为正；攻球、快带更平；搓、劈长向前偏下。只看方向和同一视频内的稳定性 |
| `peak_hand_speed` | 引拍结束到随挥结束之间，手腕相对肩中点的最大速度（躯干长度/秒，2D） | 同一机位、同一人前后对比；不要和论文的拍速（米/秒）比较。研究参考：高水平选手正手拉最大拍速约 13–15 m/s [B2018]，比赛式连续拉球时手部最大速度约 5.4–7.5 m/s [B2025] |
| `sequence_ms.hip/shoulder/elbow` | 髋转、肩转、肘屈伸三者角速度峰值相对击球的时间（毫秒，负值 = 击球前） | 只有真实帧率 ≥100 时才计算。顺序应大致为 髋 ≤ 肩 ≤ 肘 ≤ 击球（近端到远端）。研究支持髋/骨盆在动力链中的作用 [I2018][He2022 综述]，但单机位测转动不可靠，`sequence_order_ok=false` 只能当作"值得看图确认"的提示 |
| `sequence_order_ok` | 上面的顺序是否成立；帧率不够时为 null | 同上 |
| `head_motion` | 引拍结束到随挥结束之间鼻子在画面里的最大位移（躯干长度） | 包括身体整体移动；同一视频内对比，异常大时看图确认是否"甩头" |
| `recovery_dist` | 还原时手的位置离准备时的位置多远（躯干长度） | 越小说明还原越到位 |

## 五、质量与警告

| 字段 | 含义 |
|---|---|
| `quality.pose_coverage` | 这板期间识别到人的帧所占比例 |
| `quality.mean_visibility` | 关键关节的平均可见度（0–1） |
| `quality.low_visibility_parts` | 可见度低于 0.5 的关节——**这些关节相关的数字一律不用** |
| `quality.reliable` | 这板期间识别到人的帧 ≥80%，持拍臂和两肩的平均可见度 ≥0.6，且持拍侧肩、肘、腕都可见。腿被球台挡住不影响这一项，只会让腿部相关的数字为 null |
| `quality.notes` | 例如"帧率不足，不判断发力顺序"、"肘角 3D 与 2D 差异过大" |

常见警告码：

| 代码 | 意思 | 该怎么做 |
|---|---|---|
| `LOW_FPS` | 真实帧率 <60 | 时间、发力顺序、拍面角都只能粗看；建议用慢动作重拍 |
| `POSSIBLE_SLOWMO` | 前挥长得不正常 | 问用户是否慢动作导出，用 `--capture-fps` 重跑 |
| `FPS_METADATA_BAD` / `FPS_UNKNOWN` | 文件帧率信息不对 | 时间类数字要谨慎；必要时问用户真实帧率 |
| `HAND_MISMATCH` | 另一只手动作更大 | 可能是左手持拍或画面镜像；看图确认后用 `--hand` 重跑 |
| `MULTI_PERSON` | 画面里常有多人 | 看 timeline 确认跟踪的是不是用户本人；不是就用 `--player` |
| `TRACKING_JUMPS` | 跟踪跳变（剪辑、晃动、跟错人） | 跳变附近的板不用，或换片段 |
| `LOW_COVERAGE` | 大量帧没认出人 | 人太小、太暗或被挡；建议重拍 |
| `NO_STROKES` | 自动模式没找到挥拍 | 看 timeline 自己找，再用 `--strokes` |
| `POSE_UNAVAILABLE` | 没装好骨架识别 | 只能看图；建议运行 setup_env.py |
| `DECODE_STOPPED_EARLY` / `TRUNCATED` | 只分析了一部分 | 告诉用户哪些部分没分析 |

## 参考来源

- [B2020a] Bańkosz Z, Winiarski S. Kinematic Parameters of Topspin Forehand in Table Tennis and Their Inter- and Intra-Individual Variability. J Sports Sci Med 2020;19(1):138-148. https://pmc.ncbi.nlm.nih.gov/articles/PMC7039029/
- [B2020b] Bańkosz Z, Winiarski S. Using Wearable Inertial Sensors to Estimate Kinematic Parameters and Variability in the Table Tennis Topspin Forehand Stroke. Appl Bionics Biomech 2020. https://doi.org/10.1155/2020/8413948
- [B2018] Bańkosz Z, Winiarski S. Correlations between Angular Velocities in Selected Joints and Velocity of Table Tennis Racket during Topspin Forehand and Backhand. J Sports Sci Med 2018;17(2):330-338. https://pmc.ncbi.nlm.nih.gov/articles/PMC5950751/
- [B2025] Bańkosz Z, Winiarski S. Kinematics of Topspin Stroke Combinations in Table Tennis and its Inter-Individual Variability. J Sports Sci Med 2025. https://doi.org/10.52082/jssm.2025.311
- [Xing2022] Xing K 等. Biomechanical Comparison between Down-the-Line and Cross-Court Topspin Backhand in Competitive Table Tennis. IJERPH 2022;19(9):5146. https://doi.org/10.3390/ijerph19095146
- [IK2009] Iino Y, Kojima T. Kinematics of table tennis topspin forehands: effects of performance level and ball spin. J Sports Sci 2009. https://doi.org/10.1080/02640410903264458
- [I2018] Iino Y. Hip joint kinetics in the table tennis topspin forehand: relationship to racket velocity. J Sports Sci 2018. https://doi.org/10.1080/02640414.2017.1344777
- [He2022 综述] He Y, Fekete G, Sun D, Baker JS, Shao S, Gu Y. Lower Limb Biomechanics during the Topspin Forehand in Table Tennis: A Systemic Review. Bioengineering 2022;9(8):336. https://doi.org/10.3390/bioengineering9080336
- [Cao2026] Cao 等. Frontiers in Bioengineering and Biotechnology 2026. https://doi.org/10.3389/fbioe.2026.1854164
- [Shrestha2026] Shrestha 等. Sensors 2026. https://doi.org/10.3390/s26196248
- [Schurr2017] Schurr SA 等. 2D 与 3D 下肢运动学比较. https://europepmc.org/article/PMC/PMC5380858

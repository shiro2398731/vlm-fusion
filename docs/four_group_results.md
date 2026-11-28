# 四组对照：snow_fsc（视觉 / 语义 / 视觉+语义）

评测口径：baseline = 仅物理指数图；+视觉 = 指数图 + ResNet50 特征图；
+语义 = 指数图 + CLIP 相似度文本；+both = 两者都给。
指标：sev 准确率、mean_fsc 读数平均绝对误差（MAE）。

## 第一次：验证集样本（样本级干净，日期与训练重叠）

7 张（no_snow 1、light 3、moderate 3）：

| 组 | sev | MAE |
|---|--:|--:|
| baseline | 7/7 | 0.000 |
| +视觉 | 7/7 | 0.000 |
| +语义 | 7/7 | 0.001 |
| +both | 6/7 | 0.000 |

## 第二次：训练集外新日期（2025-01-15、2025-02-12）

3 张（全部 moderate）：

| 组 | sev | MAE |
|---|--:|--:|
| baseline | 3/3 | 0.000 |
| +视觉 | 3/3 | 0.000 |
| +语义 | 3/3 | 0.000 |
| +both | 3/3 | 0.000 |

## 结论

- 四组管线全部跑通：视觉特征图、CLIP 语义分数都能送进 Qwen，且不破坏 baseline
- 目前**没有可测提升**，原因是样本太容易：基线几乎全对，天花板效应
- +both 在 7 张里错 1 张（light→no_snow），提示多模态输入可能引入噪声，需要更大样本判断

## 下一步

1. 补更多训练集外日期（冬季、不同区域），凑出 no_snow/light/moderate/heavy 各 ≥5 张
2. 在基线本来会错的样本上比较四组，才看得出融合是否有效
3. 扩展到 sev 较弱的元素（coral_dhw、gpm_nt、s dust 等），那里才有提升空间

## 第三次：合并两批训练集外日期（8 张）

2025-01-15 / 2025-02-05 / 2025-02-12 / 2025-02-25 / 2025-03-10，max 覆盖 4 个区域。
等级构成：light 4、moderate 4。

| 组 | sev | MAE |
|---|--:|--:|
| baseline | 8/8 | 0.001 |
| +视觉 | 8/8 | 0.000 |
| +语义 | 8/8 | 0.001 |
| +both | 8/8 | 0.000 |

结论：snow_fsc 的基线在这些外部日期上依然全对，四组无差异。
要检验融合是否有效，必须换到**基线本来会错的元素**，并优先挑错例。

## 第四次：coral_dhw（基线 sev 较弱，8 张→实际 6 张，无 watch 类）

| 组 | sev |
|---|---|
| baseline | 5/6 |
| +视觉 | 5/6 |
| +语义 | 3/6 |
| +both | 2/6 |

结论：**当前文本级语义融合反而有害**。CLIP 分数写成“最相似类别是 X”
会在文字层面替模型下结论，模型跟着文字走，忽略了图像。
视觉特征图（+visual）未造成损害，但也未提升。

下一步：改融合策略——语义只给分数、不给 top-1 结论；再试决策级融合。

## 第五次：coral_dhw，语义改用“CLIP 嵌入 + 训练探针”

CLIP 零样本对渲染图无效：coral top-1 0%、snow_fsc 31%（接近随机）。
改为在 CLIP 嵌入上训练线性探针（532 训练 / 88 验证），验证集准确率：
coral 94.3%、snow_fsc 93.1%。

四组对照（同一批 6 张）：

| 组 | sev |
|---|---|
| baseline | 5/6 |
| +视觉 | 5/6 |
| +语义（探针） | **6/6** |
| +both | 5/6 |

结论：
1. 语义支路要走“嵌入 + 训练头”，不能直接用文本相似度
2. 探针语义把基线错的那张（none→alert1）纠正了
3. 视觉特征图（第二张图）与语义叠加时反而回退一张，说明图像级视觉融合
   还需要调（叠加/权重/换 DINOv2），当前不要和语义同时开

下一步：把探针语义扩到更多元素；视觉支路单独调融合方式，避免拖累语义。

## 第六次：snow_swe（8 张，四组）

| 组 | sev |
|---|--:|
| baseline | 3/8 |
| +视觉（并排特征图） | 1/8 |
| +语义（CLIP 探针） | **4/8** |
| +both | 2/8 |

再测视觉“叠加”融合：2/8，仍低于 baseline。

结论：
1. 语义探针在同分布上稳定带来提升（coral 5/6→6/6，snow_swe 3/8→4/8）
2. 视觉特征图**以图像形式注入会破坏 Qwen 原有判读**（并排 1/8、叠加 2/8）
   原因：挂件是在单张指数图上训练的，第二张图/叠加改变了输入分布
3. 视觉支路应改为“嵌入 + 探针”输出概率文本，与语义同机制；图像级融合不适合 v1

## 第七次：snow_swe，视觉改为“ResNet 嵌入 + 探针”

视觉支路不再注入特征图，改为和语义同一机制：ResNet50 嵌入 + 线性探针
（snow_swe 验证 89.3%、coral 96.6%），把类别概率写成文本。

同一批 8 张：

| 组 | sev |
|---|---|
| baseline | 3/8 |
| +视觉（ResNet 探针） | **5/8** |
| +语义（CLIP 探针） | 4/8 |
| +both | **5/8** |

结论（修正第六次）：
1. 视觉支路**有效**，但必须是“嵌入+探针”的概率文本，不是第二张特征图
2. 两条深度支路各自提升 1–2 张；两者同时给没有叠加增益（5/8），但不再互相拖累
3. 图像级融合（并排/叠加）在 v1 应关闭，论文里作为对照说明即可

下一步：在更多弱 sev 元素上验证（sanddust、volcano_so2、wind_gale），
并补训练集外日期复核。

## 第八次：决策级融合（tau=0.6，语义探针）

规则：Qwen 先出判读；探针最高概率 ≥ tau 时用探针等级覆盖，否则保留 Qwen 等级。
不再把语义分数写进 prompt。

| 元素 | baseline | 文本注入 | 决策融合 |
|---|--:|--:|--:|
| snow_swe | 3/8 | 4/8 | **5/8** |
| coral_dhw | 5/6 | 6/6 | **6/6** |
| sanddust | 4/6 | 3/6 | **5/6** |
| volcano_so2 | 4/8 | 3/8 | **6/8** |

tau=0.8 时：snow 4/8、coral 6/6、sanddust 5/6、so2 5/8。
tau 越低探针覆盖越多；0.6 在这四个元素上最稳。

结论：
1. **决策级融合是当前最优形态**：不再出现“文本注入拖累”
2. 探针单独判级已经接近或超过 Qwen；融合规则保留了 Qwen 在低置信样本上的作用
3. 下一步：在 uvai/gpm_nt/waterlogging/haze/wind_gale 等元素复验；
   并用训练集外日期检验探针泛化（当前探针仍为同日期训练）

## 第九次：决策融合扩到 9 个元素（tau=0.6，语义 CLIP 探针）

每元素取验证集 2 张/类，同一批样本对比 baseline 与决策融合：

| 元素 | baseline | 决策融合 | 变化 |
|---|--:|--:|--:|
| snow_swe | 3/8 | 5/8 | +2 |
| coral_dhw | 5/6 | 6/6 | +1 |
| sanddust | 4/6 | 5/6 | +1 |
| volcano_so2 | 4/8 | 6/8 | +2 |
| uvai_dust | 5/8 | 6/8 | +1 |
| gpm_nt | 3/6 | 6/6 | +3 |
| haze | 2/6 | 6/6 | +4 |
| waterlogging_pct | 4/6 | 4/6 | 0 |
| wind_gale | 3/6 | 4/6 | +1 |

合计：8 升 1 平，无下降。

注意：探针仍在与验证集同日期范围的数据上训练，
这组数字证明的是“决策融合机制有效”，还不是跨日期泛化。
下一步必须用训练集外日期复核（snow_fsc 已备好 8 张外部样本）。

## 第十次：训练集外日期复核（snow_fsc，8 张）

探针只用 2024 年及以前的样本训练，测试用 2025 年新日期（训练集外）：

| 方法 | sev |
|---|--:|
| baseline（之前测过） | 8/8 |
| CLIP 探针单独 | 6/8 |
| ResNet 探针单独 | 7/8 |
| 决策融合（CLIP 探针，tau 0.6） | 6/8 |

结论（重要）：
1. 探针能跨日期工作（CLIP 6/8、ResNet 7/8），不是只会背训练日期
2. 但 baseline 在 snow_fsc 外部样本上本来就 8/8，融合只会把它拉低
3. **融合必须按元素开关**：只在 baseline 弱的元素上启用；
   部署配置应由同分布验证结果决定 go/no-go，而不是全局默认开

## 第十一次：真正的训练集外日期复核（wave / groundwater / earthquake）

这三个元素的验证集在日期层面本来就干净（切分核查 verdict=干净），
探针只用训练日期样本，验证集是训练集外日期。

| 元素 | baseline | 决策融合（tau0.6） | CLIP探针单独 | ResNet探针单独 |
|---|--:|--:|--:|--:|
| wave | 2/8 | **3/8** | 64.3%(全量) | 78.6%(全量) |
| groundwater_overuse | 4/6 | 4/6 | 79.9%(全量) | 79.7%(全量) |
| earthquake_def | 4/6 | **6/6** | 92.6%(全量) | 92.6%(全量) |

结论：跨日期上仍是 2 升 1 平，无下降。
探针在外部日期的全量验证准确率（64–93%）明显高于 baseline sev（54–70%），
说明“深度网络嵌入+探针”确实补到了 Qwen 没利用好的视觉信息。

累计证据：
- 同分布 9 元素：8 升 1 平
- 外部日期 3 元素：2 升 1 平
- snow_fsc 外部：baseline 8/8，融合会拉低 → 关闭
因此融合按元素开关，不是全局默认。

## 第十二次：决策融合再扩 7 个元素（tau=0.6）

| 元素 | baseline | 决策融合 | 变化 |
|---|--:|--:|--:|
| wildfire_smoke | 3/8 | 6/8 | +3 |
| sea_ice_amsr2 | 3/8 | 6/8 | +3 |
| coastal_nt | 5/6 | 6/6 | +1 |
| flood_water | 3/4 | 4/4 | +1 |
| heatwave_lst | 3/4 | 4/4 | +1 |
| volcano_thermal | 3/4 | 4/4 | +1 |
| snowstorm | 7/8 | 7/8 | 0 |

累计：同分布 16 元素 = 14 升 2 平 0 降；
训练集外 3 元素 = 2 升 1 平；另有 snow_fsc 外部（baseline 全对）已关闭。

## 第十三次：全部 32 元素决策融合验证（tau=0.6）

见 [decision_summary.md](decision_summary.md)：升 18、平 13、降 1。

显著提升：flood_precip_global +5、hurricane_rain +5、flare +4、haze +4、wildfire +4、
gpm_nt +3、sea_ice_amsr2 +3、wildfire_smoke +3、earthquake_def +2、sea_ice_s2_arctic +2。
唯一下降：green_tide 5/6→4/6（关闭融合）。
持平的多是 baseline 本来就强的元素（enso/iod/flood_opt 等）。

## 第十四次：DINOv2 支路（timm 本地权重，518 输入）

探针验证准确率（同分布）：

| 元素 | ResNet50 | CLIP | DINOv2 |
|---|--:|--:|--:|
| snow_swe | 89.3 | 89.3 | 94.7 |
| coral_dhw | 96.6 | 94.3 | 95.5 |
| haze | 88.1 | 91.1 | 90.2 |
| sanddust | 81.7 | 80.0 | 90.0 |
| wind_gale | 79.5 | 76.9 | 73.1 |

决策融合（DINOv2 探针，tau 0.6）：

| 元素 | baseline | CLIP 融合 | DINOv2 融合 |
|---|--:|--:|--:|
| snow_swe | 3/8 | 5/8 | 5/8 |
| sanddust | 4/6 | 5/6 | **6/6** |
| haze | 2/6 | 6/6 | 6/6 |
| coral_dhw | 5/6 | 6/6 | 6/6 |

结论：DINOv2 在 sanddust 上优于 CLIP；其他元素持平。
视觉支路现在有三种可选提取器：resnet50 / clip / dinov2，逐元素可配。

## 第十五次：tau 扫描（haze/gpm_nt/flood_precip_global/hurricane_rain/flare/wildfire）

| 元素 | tau0.5 | tau0.6 | tau0.7 |
|---|--:|--:|--:|
| haze | 6/6 | 6/6 | 6/6 |
| gpm_nt | 6/6 | 6/6 | 6/6 |
| flood_precip_global | 7/8 | 7/8 | 7/8 |
| hurricane_rain | 7/8 | 7/8 | 7/8 |
| flare | 6/8 | 6/8 | 4/8 |
| wildfire | 4/4 | 4/4 | 4/4 |

结论：0.6 稳定；flare 对高阈值敏感，保持 0.6 即可。

## 第十六次：训练集外日期的四组对照（验证集即外部日期）

| 元素 | baseline | +视觉 | +语义 | +both |
|---|--:|--:|--:|--:|
| wave | 2/8 | 3/8 | 3/8 | 3/8 |
| earthquake_def | 4/6 | 5/6 | 5/6 | 5/6 |
| groundwater_overuse | 4/6 | 4/6 | 4/6 | 4/6 |

说明：视觉/语义均为嵌入+探针的概率文本，不是特征图。
结论：外部日期上 2 升 1 平，与决策级融合结论一致；
按“文本注入”这条最弱的融合路径也能带来提升，说明探针本身有信息量。

## 第十七次：带 det / 数值 MAE 的外部四组对照

| 元素 | 组 | det | sev | MAE |
|---|---|--:|--:|--:|
| earthquake_def | baseline | 4/6 | 4/6 | – |
| earthquake_def | +视觉 | 4/6 | 5/6 | – |
| earthquake_def | +语义 | 4/6 | 5/6 | – |
| earthquake_def | +both | 4/6 | 5/6 | – |
| wave | baseline | 6/8 | 2/8 | 0.324 |
| wave | +视觉 | 6/8 | 3/8 | 0.000 |
| wave | +语义 | 6/8 | 3/8 | 0.000 |
| wave | +both | 7/8 | 3/8 | 0.370 |

结论：外部日期上 det 基本持平或小升，sev 2 升 1 平；
wave 的 +both 把 det 从 6/8 提到 7/8，但数值 MAE 变差，说明融合对数值读数无帮助，
数值应回到物理统计量直接判读。

## 第十八次：原始特征向量摘要附加（include_vectors）

做法：探针训练时对嵌入做 PCA 降到 16 维，推理时把该样本的 16 维摘要以数字形式写进
输入文本（“Visual/Semantic feature vector (compressed 16 dims): [...]”），与概率一起给 Qwen。

coral_dhw（6 张）：

| 配置 | baseline | +视觉 | +语义 | +both |
|---|--:|--:|--:|--:|
| 不附向量（之前） | 5/6 | 5/6 | 6/6 | 5/6 |
| 附两路向量 | 5/6 | 6/6 | 5/6 | 6/6 |
| 只附视觉向量 | 5/6 | 6/6 | 6/6 | 6/6 |

snow_swe（8 张）：

| 配置 | baseline | +视觉 | +语义 | +both |
|---|--:|--:|--:|--:|
| 不附向量（之前） | 3/8 | 5/8 | 4/8 | 5/8 |
| 附两路向量 | 3/8 | 5/8 | 2/8 | 5/8 |

结论：
1. 原始向量摘要“可以附”，链路已通（`include_vectors` / `include_visual_vectors` /
   `include_semantic_vectors` 三个开关）
2. 视觉向量对 coral 有帮助（+both 从 5/6 到 6/6）
3. 语义向量反而干扰（snow_swe +语义 4/8→2/8），建议语义保持概率、不附向量
4. 默认关闭；按元素验证后再开

## 第十九次：DINOv2 决策融合复验（tau0.6，9 元素）

| 元素 | baseline | CLIP 融合 | DINOv2 融合 |
|---|--:|--:|--:|
| snow_swe | 3/8 | 5/8 | 5/8 |
| coral_dhw | 5/6 | 6/6 | 6/6 |
| sanddust | 4/6 | 5/6 | **6/6** |
| volcano_so2 | 4/8 | 6/8 | 5/8 |
| haze | 2/6 | 6/6 | 6/6 |
| wind_gale | 3/6 | 4/6 | **5/6** |
| earthquake_def | 4/6 | 6/6 | 6/6 |
| wave | 2/8 | 3/8 | **5/8** |
| gpm_nt | 3/6 | 6/6 | 6/6 |

结论：
- DINOv2 在 sanddust、wind_gale、wave 上优于 CLIP（wave 外部日期 3/8→5/8）
- volcano_so2 上不如 CLIP（6/8→5/8）
- 其余持平

因此逐元素开关按“更优探针”选择：sanddust/wind_gale/wave 用 DINOv2，其余默认 CLIP。

## 第二十次：全量三方对照（baseline / CLIP 决策 / DINOv2 决策）

38 个元素全部跑完三方（同分布，2 张/类，tau 0.6）：

- 以最优探针计：**升 26、平 12、降 0**
- CLIP 在 volcano_so2 等元素更强；DINOv2 在 sanddust、wind_gale、wave、
  uvai_dust、flare、gpm_nt 等更强
- green_tide、iod、marine_heatwave 上 DINOv2 明显更差，配置中不选它

明细见 [decision_all_sources.md](decision_all_sources.md)；配置已按“最优探针”重算：
`configs/fusion_selected.yaml` 38 元素中 26 个启用。

## 第二十一次：DINOv2 决策融合的阈值敏感性

| 元素 | tau0.5 | tau0.6 | tau0.7 |
|---|--:|--:|--:|
| sanddust | 6/6 | 6/6 | 6/6 |
| wave | 5/8 | 5/8 | 4/8 |
| wind_gale | 5/6 | 5/6 | 5/6 |
| flare | 6/8 | 6/8 | 6/8 |
| gpm_nt | 6/6 | 6/6 | 6/6 |
| uvai_dust | 7/8 | 7/8 | 6/8 |
| haze | 6/6 | 6/6 | 6/6 |
| coastal_nt | 6/6 | 6/6 | 6/6 |

结论：0.6 与 0.5 持平或更优，0.7 在 wave/uvai 上变差；维持 0.6。

## 第二十二次：三路向量四组对照（haze / gpm_nt）

| 元素 | baseline | +视觉 | +语义 | +both |
|---|--:|--:|--:|--:|
| haze | 2/6 | 2/6 | 2/6 | 2/6 |
| gpm_nt | 3/6 | 5/6 | 4/6 | 5/6 |

结论：
- gpm_nt：视觉文本（含向量）+2，语义 +1，双路 +2
- haze：文本注入无效，但决策级覆盖可到 6/6，说明该元素收益来自“探针高置信覆盖”
  而不是把分数写进 prompt

## 第二十三次：三路向量四组对照（8 个弱元素）

| 元素 | baseline | +视觉 | +语义 | +both |
|---|--:|--:|--:|--:|
| volcano_so2 | 4/8 | 4/8 | 3/8 | 4/8 |
| wind_gale | 3/6 | 4/6 | 2/6 | 4/6 |
| sanddust | 4/6 | 3/6 | 3/6 | 3/6 |
| coastal_sla | 4/8 | 4/8 | 4/8 | 5/8 |
| hurricane_opt | 3/6 | 4/6 | 4/6 | 4/6 |
| wildfire_smoke | 3/8 | 4/8 | 4/8 | 5/8 |
| sea_ice_amsr2 | 3/8 | 6/8 | 5/8 | 6/8 |
| uvai_dust | 5/8 | 5/8 | 4/8 | 5/8 |

结论：
- 视觉文本 +both 在 6/8 元素上升或持平，sea_ice_amsr2 提升最大（3/8→6/8）
- 语义文本单独注入仍不稳定（wind 2/6、sanddust 3/6）
- 决策级覆盖仍优于文本注入（如 haze 2/6→6/6）

## 第二十四次：四组对照全量汇总（三路向量默认开启）

38 元素四组对照（baseline / +视觉 / +语义 / +both），全部脚本记录见
[four_group_all.md](four_group_all.md)。

合计（+both 对比 baseline）：**升 14、平 21、降 4**。
- 视觉文本（+visual）在 sea_ice_amsr2、wildfire_smoke、snowstorm、gpm_nt 等元素有效
- 语义文本单独注入仍不稳定（ground_subsidence、groundwater_grace 下降）
- 文本注入整体不如决策级覆盖（26 升 12 平 0 降）

因此最终采用：三路向量作为输入信息 + 决策级融合；文本注入保留为消融。

## 第二十五次：tau=0.65 中间值复核

| 元素 | tau0.6 | tau0.65 | tau0.7 |
|---|--:|--:|--:|
| sanddust | 6/6 | 6/6 | 6/6 |
| wave | 5/8 | 4/8 | 4/8 |
| wind_gale | 5/6 | 5/6 | 5/6 |
| flare | 6/8 | 6/8 | 6/8 |
| gpm_nt | 6/6 | 6/6 | 6/6 |
| uvai_dust | 7/8 | 7/8 | 6/8 |
| haze | 6/6 | 6/6 | 6/6 |
| coastal_nt | 6/6 | 6/6 | 6/6 |

结论：0.6 仍最优；0.65 已在 wave 上退步。最终定为 tau=0.6。

## 第二十六次：原始链路端到端（snow_swe）

用 raw_workflow 从原始产品渲染 6 个窗口，再走新融合链路（三路特征 + 决策融合）：

| 指标 | 结果 |
|---|---|
| raw baseline | 2/6 |
| raw + 决策融合 | 2/6 |

明细：4 个 heavy 窗口均被判成 moderate/light；2 个 none 窗口判对。
结论：该元素的误差来自降雪量被系统性低估，探针无法纠正；
说明融合不是万能，弱元素还需按元素处理（阈值重估或重训挂件）。

## 第二十七次：原始链路端到端（wind_gale / uvai_dust）

原始产品 → 渲染 → 三路特征 → 决策融合（fusion_best.yaml）：

| 元素 | raw baseline | raw + 决策融合 | 说明 |
|---|--:|--:|---|
| wind_gale | 4/4 | 2/4 | 探针把 storm 误覆盖为 gale（两次） |
| uvai_dust | 1/2 | 2/2 | 修正了 light→none |

结论：融合在真实渲染色上有正有负，说明探针在“训练未用日期”上仍需复核；
wind_gale 的探针把 storm 判成 gale，属于等级边界与类别定义问题，
需要按元素重新校准（探针阈值或标签映射），不能盲信覆盖。

补充：wind_gale 覆盖阈值调到 0.8/0.9 仍为 2/4，说明探针在外部日期上
“高置信地判错”（storm→gale），不是阈值问题，而是类别边界/标签映射问题；
该元素在 fusion_best 中应改为 semantic 或暂时关闭，待重新校准。

## 第二十八次：wind_gale 外部复核后改回 CLIP 探针

原始链路 4 张（2 storm、2 normal）对比：

| 探针 | raw sev |
|---|--:|
| DINOv2 | 2/4（storm 高置信误判为 gale） |
| CLIP | **4/4** |

已把 `fusion_best.yaml` 中 wind_gale 的 source 改回 semantic（CLIP 探针），
重新验证 4/4。说明：逐元素探针选择不能只看同分布 2 张/类的数字，
必须结合训练集外/原始链路复核。

## 第二十九次：原始链路端到端（earthquake_def）

ARIA GUNW 渲染 2 个窗口：

| 指标 | raw baseline | raw + 决策融合 |
|---|--:|--:|
| earthquake_def | 1/2 | **2/2** |

明细：stable 窗口基线误判 slight_deform，探针覆盖为 stable 修正；
obvious_deform 窗口保持正确。
该元素（fusion_best 中 source=visual/DINOv2）在原始链路上验证通过。

## 第三十次：原始链路端到端（InSAR 两元素，配置为关闭）

| 元素 | raw baseline | raw + 配置（关闭） |
|---|--:|--:|
| ground_subsidence | 2/4 | 2/4 |
| groundwater_overuse | 2/4 | 2/4 |

两元素在 fusion_best 中 enabled=false，验证了“关闭即保持 baseline”：
结果与 raw baseline 完全一致。稳定窗口判对，明显沉降窗口被低估为 stable，
属于数值判读问题，需要阈值/挂件层面处理，而非融合能解决。

## 第三十一次：原始链路端到端（sea_ice_s2_arctic / wildfire）

| 元素 | raw baseline | raw + 决策融合 |
|---|--:|--:|
| sea_ice_s2_arctic | 2/2 | 2/2 |
| wildfire | 12/12 | 12/12 |

两元素在 fusion_best 中启用，原始链路结果与 baseline 一致或全对，
说明探针在这些元素上没有引入破坏；sea_ice 探针覆盖后等级保持 ice。

## 第三十二次：原始链路端到端（green_tide）

| 指标 | raw baseline | raw + 配置（关闭） |
|---|--:|--:|
| green_tide | 2/2 | 2/2 |

green_tide 在 fusion_best 中关闭（同分布融合会下降），原始链路保持 baseline 全对。

## 第三十三次：原始链路端到端（coral_dhw）

| 指标 | raw baseline | raw + 决策融合 |
|---|--:|--:|
| coral_dhw | 2/2 | 2/2 |

ERDDAP 截取的 DHW 网格渲染两窗口，均无热胁迫，融合保持 none。

## 第三十四次：wave 原始链路复核与探针修正

| 探针 | raw wave sev |
|---|--:|
| DINOv2 | 2/3（高置信误覆盖） |
| CLIP | **3/3** |

已把 wave 也改回 semantic（CLIP 探针），fusion_best 复核 3/3。
与 wind_gale 相同：同分布小样本选出的 DINOv2 在真实链路外日期上不可靠，
按外部证据回退到 CLIP。

## 第三十五次：flood_precip_global 原始链路复核（关闭融合）

| 探针 | raw fpg sev |
|---|--:|
| DINOv2 | 0/2 |
| CLIP | 0/2 |

两种探针在 GDACS 回退窗口上均高置信误覆盖（flood→none），
raw baseline 本身 2/2。已把 flood_precip_global 设为强制关闭（与 snow_fsc 同）。
说明同分布小样本（2 张/类）选出的探针不能盲信，必须过原始链路复核。

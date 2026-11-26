# 原始影像链路 · 已通元素记录

> 目标：每个元素给出「原始产品文件 → 渲染指数图 → 模型判读 → 阈值对照」的可复现命令与结果。
> 规则：优先本地原始数据；不下载整年/整数据集；验证优先用训练未见时段。
> 口径：**日期只用于挑选样本，不进入判据**；判据一律是物理量阈值（mm、m、%、°C 等）。

## 1. flood_opt（Sentinel-2 L2A）

- 命令：
  ```
  python scripts/raw_s2_to_ndwi.py --safe <xxx.SAFE> --windows 6 --outdir outputs/raw_test
  python scripts/validate_raw_workflow.py --dir outputs/raw_test --api http://127.0.0.1:8078
  ```
- 阈值：水占比 <0.15 正常 / 0.15–0.55 内涝 / ≥0.55 洪水
- 实测：
  - 2025-09-16 信德省（未参与训练）：7/7 一致
  - 2025-08-27 信德省：7/7 一致
  - 2025-10-07 孟加拉：3/3 一致
  - 2024-06-07 鄱阳湖：4/4 一致
- 合计 4 景 21 窗口全部一致；覆盖 normal/waterlogging，**未覆盖 flood 类**
- 限制：真值由同一 NDWI 阈值生成，验证的是链路与判读；判据文字（20%）与实际标签阈值（15%/55%）不一致

## 2. wildfire（FIRMS 火点，2025 年）

- 命令：
  ```
  python scripts/raw_workflow.py --elements wildfire --outdir outputs/raw_pair1 --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/03_thermal_infrared/firms/chn2025`（训练只用 2024 年）
- 阈值：有效火点数 <5 无火 / 5–30 slight / 30–100 moderate / ≥100 severe
- 实测：12/12 一致（2025-12-14 至 12-19，含空白格负样本）
- 限制：区域固定中国范围；窗口 5°

## 3. volcano_thermal（MODVOLC 热异常，2026-10）

- 命令：
  ```
  python scripts/raw_workflow.py --elements volcano_thermal --outdir outputs/raw_pair1_volcano --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/03_thermal_infrared/modvolc/*.txt`，取数据量最大的 4 座火山最近 20 天窗口
- 阈值：窗口内 Excess ≥5 的点数 ≥3 → 有热异常
- 实测：4/4 一致（264230、257040、211040、263300，2026-10-06/07 窗口）
- 限制：训练窗覆盖到 2026-10-03，本测试窗口在其后；窗口边界相邻，需继续拉长时间验证



## 4. snow_swe（AMSR2 积雪当量，2022-03 未训练日期）

- 命令：
  ```
  python scripts/raw_workflow.py --elements snow_swe --outdir outputs/raw_pair2_snow --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/04_passive_microwave/swe_amsr2`（93 个日文件；训练用 47 天，本次取未用的 2022-03-15/25）
- 阈值：平均 SWE <15 无雪 / 15–30 light / 30–60 moderate / ≥60 heavy
- 实测：**检出 6/6 正确**；等级 2/6 一致（厚雪 70–97 mm 被判 moderate/light，偏低一档）
- 结论：链路通；模型对积雪强度系统性偏低，需查渲染色标与训练标签分布

## 5. waterlogging_pct（SMAP 土壤湿度百分位）

- 命令：
  ```
  python scripts/raw_workflow.py --elements waterlogging_pct --outdir outputs/raw_pair2_wlp --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/04_passive_microwave/smap2`（46 个日文件，**全部被训练使用过**，本测试只验证链路）
- 阈值：窗口内最大百分位 <80 无 / 80–90 light / 90–97 moderate / ≥97 heavy
- 实测：4/6 一致。两个“最大百分位 47”的负窗口被模型判成 moderate（假阳性）
- 结论：链路通；负样本含湿区时易误判，需在负样本构造上收紧（当前负样本只按 <50 取，边界太宽）

## 6. drought_grace（GRACE-DA 根区土壤湿度百分位）

- 命令：
  ```
  python scripts/raw_workflow.py --elements drought_grace --outdir outputs/raw_pair3_grace --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/04_passive_microwave/grace_gw`（157 个周资料 .nc4，本地）
- 阈值：窗口平均百分位 ≤10 判 drought，否则 none
- 实测：6/6 一致（2022-12-19/26；正例 2.4–3.1，负例 78.8）

## 7. groundwater_grace（GRACE-DA 地下水储量百分位）

- 命令同上，`--elements groundwater_grace`
- 阈值：≤10 判 depleted
- 实测：6/6 一致（正例 1.7–2.9，负例 83–84）

## 8. wind_gale（CMEMS 风场，2026-10 未训练日期）

- 命令：
  ```
  python scripts/raw_workflow.py --elements wind_gale --outdir outputs/raw_pair4_wind --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/07_scatterometer/wind`（17 个 nc；训练只用 2025-01 前 15 天）
- 阈值：30° 窗内最大风速 <17.2 正常 / 17.2–24.5 gale / ≥24.5 storm
- 实测：**4/4 一致**：storm 38.2、49.7 判 storm；normal 8.6、7.7 判 normal
- 备注：这是首个在未见日期上把「严重级」判对的元素

## 9. hurricane_rain（GPM IMERG 日累积）

- 命令：
  ```
  python scripts/raw_workflow.py --elements hurricane_rain --outdir outputs/raw_pair4_hur --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/06_radar_precipitation/gpm`（1632 个半小时 HDF5，合成为 33 天日累积；**这 33 天全部进过训练**，无未见日期）
- 阈值：窗内最大日雨量 <50 正常 / 50–100 rainstorm / 100–250 heavy / ≥250 extreme
- 实测：2/4。检出全对；两个极端窗（369/497 mm）被判 rainstorm，等级低估
- 结论：链路通；与 snow_swe 一样表现为「强度判轻」

## 10. bloom_chla（ERDDAP 叶绿素日产品，2022-12 未训练日期）

- 命令：
  ```
  python scripts/raw_workflow.py --elements bloom_chla --outdir outputs/raw_pair5_bloom --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/01_optical_vnir/chla_erddap`（36 个月度 nc，chlor_a；训练用 120 天，本次取 2022-12-01/02）
- 阈值：窗内 P90 叶绿素 <1 none / 1–5 light / 5–20 moderate / ≥20 heavy
- 实测：2/4。两个 P90≈8.5–8.9（moderate）的正窗被判 none（**漏检**）；两个 <1 的负窗判对
- 结论：链路通；模型对新日期的中等强度水华明显漏检，需查渲染差异与训练窗选取方式

## 11. sea_ice_amsr2（NSIDC 海冰浓度；产品不匹配，验证未通过）

- 命令：
  ```
  python scripts/raw_workflow.py --elements sea_ice_amsr2 --outdir outputs/raw_pair5_ice --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/04_passive_microwave/ice_conc`（NSIDC CDR 版本，2020-12～2021-03）
- 阈值：窗内冰覆盖比例（≥15% 像元占比）<15 none / 15–50 light / 50–85 moderate / ≥85 heavy
- 实测：**0/4**。模型全部输出 heavy（ice_frac≈95%），真值 light（47.7%）与 none（13.6%）
- 原因判断：训练用的是 NSIDC **AU_SI12 日产品**（HDFEOS 组，12.5 km），本地只有 **CDR 版本**（448×304 极区网格、覆盖率口径不同）。换产品后图像分布变了，模型仍按 AU_SI12 的视觉规律判
- 结论：**链路代码通，但产品不能替换**。要验证该元素，必须用 AU_SI12 的 he5 单日文件（单文件约几十 MB，需单独下载）

## 12. gpm_nt（GPM 日累积，2024-11 未训练日期）

- 命令：
  ```
  python scripts/raw_workflow.py --elements gpm_nt --outdir outputs/raw_pair6_gpmnt --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/06_radar_precipitation/gpm`（训练只用 2026-08-06～12 共 7 天；本次用 2024-11-01/02，未见过）
- 阈值：窗内平均降水 <0.5 dry / 0.5–5 moderate_rain / ≥5 heavy_rain
- 实测：**4/4 一致**：183.8、187.9 mm 判 heavy_rain；0.0 mm 判 dry

## 13. wave（Sentinel-3 SRAL 沿轨有效波高，2026-10）

- 命令：
  ```
  python scripts/raw_workflow.py --elements wave --outdir outputs/raw_pair6_wave --api http://127.0.0.1:8078
  ```
- 数据：`data/raw/08_altimetry/infa`（60 个 SRAL zip，2026-10；训练用的是 2021 年台风 In-fa 轨道）
- 阈值：max Hs <2.5 calm / 2.5–3.5 蓝 / 3.5–4.5 黄 / 4.5–6.0 橙 / ≥6.0 红
- 实测：**3/3 一致**：19.6 m、7.9 m、15.6 m 均判 red
- 限制：三条轨道都是 red，无低波高负样本

## 14. flood_precip_global（GPM 3 日累积）

- 命令：
  ```
  python scripts/raw_workflow.py --elements flood_precip_global --outdir outputs/raw_pair7_flp --api http://127.0.0.1:8078
  ```
- 数据：本地 GPM 33 天（2023-08～2024-11）；GDACS 事件库是 2019–2022，**无重叠**，故回退为"最大 3 日累积窗口"（非事件驱动，已标注）
- 判据：3 日累积降水 ≥100 mm 判 flood，否则 none（该元素在训练数据里就是二档词表）
- 实测：**2/2 一致**：812.8 mm、877.3 mm 均判 flood

## 15. drought_ndvi_clim（MODIS MOD13C2 NDVI 百分位）

- 命令：
  ```
  python scripts/raw_workflow.py --elements drought_ndvi_clim --outdir outputs/raw_pair7_ndvi --api http://127.0.0.1:8078
  ```
- 数据：本地 MODIS NDVI 60 个月文件
- 判据：窗口平均 NDVI 百分位 ≤10 判 drought，否则 none
- 实测：4/6。干旱正例全对（百分位 0.0）；两个百分位 72–78 的负窗被判 drought（**假阳性**）
- 结论：链路通；高植被百分位被误报为干旱，需查模型对色标方向的读取

## 16. geomagnetic（WDC Dst 指数，2026-05/06）

- 命令：
  ```
  python scripts/raw_workflow.py --elements geomagnetic --outdir outputs/raw_pair8_geo --api http://127.0.0.1:8078
  ```
- 数据：WDC Kyoto 月度 Dst 文本（每次约 3.7 KB，共取 2 个月 ≈ 7.5 KB）
- 判据：当日最小 Dst > −50 quiet / −50～−100 weak / −100～−200 moderate / −200～−350 severe / ≤−350 extreme
- 实测：**4/4 一致**：−71、−57 nT 判 weak_storm；+1、−1 nT 判 quiet
- 备注：训练只覆盖 2024 年，本次用 2026-05/06

## 17. haze（MODIS MOD08 AOD，OPeNDAP 取变量）

- 命令：
  ```
  python scripts/raw_workflow.py --elements haze --outdir outputs/raw_pair8_haze --api http://127.0.0.1:8078
  ```
- 数据：**不下载 161 MB 整文件**，用 OPeNDAP 只取 `AOD_550_..._Mean` 的 180×360 网格（约几百 KB 文本）
- 判据：窗内均值 AOD <0.2 clear / 0.2–0.6 haze / >0.6 或局部 >1.5 severe_haze
- 实测：1/2。均值 0.47、局部 >1.5 的窗真值 severe_haze，模型判 haze（边界分歧）；clear 窗判对
- 结论：链路通；重度霾与中度在图像上差异小，模型倾向判中

## 18. sanddust（MOD08 Deep Blue 陆地 AOD，OPeNDAP 取变量）

- 命令：
  ```
  python scripts/raw_workflow.py --elements sanddust --outdir outputs/raw_pair9_dust --api http://127.0.0.1:8078
  ```
- 数据：MOD08 单日 AOD 变量（OPeNDAP，不整下文件）
- 判据：窗内均值 >0.22 或峰值 >1.4 判 dust_storm；均值 >0.22 判 dust；否则 clear
- 实测：1/2。峰值 3.50 的窗真值 dust_storm，模型判 dust；clear 窗判对

## 19. uvai_dust（OMI OMAERUVd，单日 635 KB）

- 命令：
  ```
  python scripts/raw_workflow.py --elements uvai_dust --outdir outputs/raw_pair9_uvai --api http://127.0.0.1:8078
  ```
- 数据：OMAERUVd 单日 L3（635 KB，整份下载）
- 判据：窗内最大 UVAI <1 none / 1–2 light / 2–4 moderate / ≥4 heavy
- 实测：1/2。UVAI 6.62 判 heavy 正确；UVAI 0.19 的窗被判 light（边界误报）

## 20. wildfire_smoke（FIRMS 火点 + OMI UVAI，2025-06-15）

- 命令：
  ```
  python scripts/raw_workflow.py --elements wildfire_smoke --outdir outputs/raw_pair9_smoke --api http://127.0.0.1:8078
  ```
- 数据：本地 FIRMS `chn2025` 当日火点 + OMAERUVd 单日 UVAI（664 KB）
- 判据：窗内最大 UVAI <1 none / 1–2 light / 2–4 moderate / ≥4 heavy
- 实测：1/4。四个窗实为同一区域相邻位置（UVAI 2.25），模型三次判 light、一次判 moderate；真值 moderate
- 结论：链路通；UVAI 2.25 卡在 light/moderate 边界，模型倾向低一档

## 备注：两条被卡住的路线（未下载）

- **heatwave_lst / frost**：MOD11C3 可走 OPeNDAP，但 36 个月百分位基期按中国区取，累计约 40–50 MB 文本。体积未获批准，暂缓。
- **flare**：NCEI/AWS 的 XRS 月度与日文件路径已变更，原 URL 全部 404；需要重找数据源后再接。

## 21. coral_dhw（NOAA CRW DHW，ERDDAP 截取）

- 命令：
  ```
  python scripts/raw_workflow.py --elements coral_dhw --outdir outputs/raw_pair10_coral --api http://127.0.0.1:8078
  ```
- 数据：ERDDAP `NOAA_DHW` 按经纬度截取（大堡礁 10°×10°，约 201×241 网格，几百 KB）
- 判据：窗内最大 DHW <1 none / 1–4 watch / 4–8 alert1 / ≥8 alert2
- 实测：
  - 2026-11-20（无白化）：2/2，DHW=0 判 none
  - 2024-03-15（大堡礁白化）：1/2，DHW 15.72 判 alert2 正确；DHW 7.45（alert1）被判 alert2
- 结论：链路通；等级边界偏高一档

## 22. volcano_so2（OMI OMSO2e 单日 3.4 MB）

- 命令：
  ```
  python scripts/raw_workflow.py --elements volcano_so2 --outdir outputs/raw_pair10_so2 --api http://127.0.0.1:8078
  ```
- 数据：OMSO2e 单日 L3（3.4 MB，整份下载）；**事件库缺失，窗口按最大 SO2 选取**
- 判据：窗内最大 SO2 柱浓度 <1 none / 1–5 light / 5–20 moderate / ≥20 heavy
- 实测：1/2。9.46 DU 判 light（真值 moderate，低一档）；0.67 DU 判 none 正确

## 23. enso（OISST 距平，ERDDAP 截取，2026-11-01）

- 命令：
  ```
  python scripts/raw_workflow.py --elements enso --outdir outputs/raw_pair11_enso --api http://127.0.0.1:8078
  ```
- 数据：ERDDAP `ncdcOisst21Agg.anom`，热带太平洋 60°×100° 小区域
- 判据：Niño3.4 区距平 ≥+0.5℃ El Niño / ≤−0.5℃ La Niña / 否则 neutral
- 实测：**1/1**。Niño3.4 距平 +3.48℃，真值 El Nino，模型 el_nino 正确
- 备注：用当日距平代替 3 个月 ONI

## 24. iod（OISST 距平，印度洋，2026-11-01）

- 命令：
  ```
  python scripts/raw_workflow.py --elements iod --outdir outputs/raw_pair11_iod --api http://127.0.0.1:8078
  ```
- 数据：ERDDAP 印度洋 40–120°E 小区域
- 判据：DMI ≥+0.4 positive_iod / ≤−0.4 negative_iod / 其余 neutral_iod
- 实测：0/1。DMI = +1.00（明确正 IOD），模型判 neutral_iod（**漏判**）

## 25. marine_heatwave（OISST 距平，30°窗，2024-08-01）

- 命令：
  ```
  python scripts/raw_workflow.py --elements marine_heatwave --outdir outputs/raw_pair12_mhw --api http://127.0.0.1:8078
  ```
- 数据：ERDDAP OISST 距平，北太平洋 30°×30° 小区域
- 判据：距平 >+1.0℃ 的面积占比 ≥20% 判 marine_heatwave，否则 normal
- 实测：**1/1**：热面积 36.6%，真值 marine_heatwave，模型一致

## 26. snowstorm（AMSR2 SWE → 雪深分级，2022-03 未训练日期）

- 命令：
  ```
  python scripts/raw_workflow.py --elements snowstorm --outdir outputs/raw_pair12_storm --api http://127.0.0.1:8078
  ```
- 数据：本地 AMSR2 积雪当量（swe_amsr2），训练只用到 2024 年，本次用 2022-03-25/30
- 判据：按 GB/T 20482-2017 雪深 <10 无 / 10–20 轻 / 20–30 中 / ≥30 重（10:1 由 SWE 换算）
- 实测：**4/4**：214 cm、240 cm 判 severe；无雪窗判 no_disaster

## 27. flare（SWPC XRS + SDO AIA193，2026-11 最近 7 天）

- 命令：
  ```
  python scripts/raw_workflow.py --elements flare --outdir outputs/raw_pair13_flare --api http://127.0.0.1:8078
  ```
- 数据：NOAA SWPC 7 天 XRS JSON（4.6 MB）+ SDO AIA193 图像两张（各约 1 MB）
- 判据：XRS-B 峰值 X≥1e-4 / M≥1e-5 / C≥1e-6，其余 quiet；模型只输出 detected（flare/none）
- 实测：1/2。M 级（2.4e-5）判 detected=true 正确；静默时刻（1.2e-7）误报 detected=true

## 28. heatwave_lst（MOD11C3 月度 LST 百分位，42 个月）

- 命令：
  ```
  python scripts/raw_workflow.py --elements heatwave_lst --outdir outputs/raw_pair14_heat --api http://127.0.0.1:8078
  ```
- 数据：MOD11C3 月度 LST 中国区 0.1° 子集（2023-01～2026-06，OPeNDAP，已缓存 data/interim/mod11c3_china_0p1.npz）
- 判据：窗口 LST 百分位 ≥90 判 heatwave
- 实测：**1/3**。百分位 90.1 判 heatwave 正确；90.0、90.1 两个窗判 none（漏检）
- 备注：百分位基期含 2026 年测试月，属同分布验证，非严格留出

## 附：frost 不是已选元素

`frost` 在服务 `/api/elements` 的 38 个里不存在（只有 heatwave_lst、marine_heatwave）。
它能渲染、能判读，但没有对应的 LoRA，接口返回 `unknown element`。因此只作为额外渲染器记录，不计入 38 元素进度。

## 29. sea_ice_s2_arctic（Sentinel-2 L2A 单景 NDSI）

- 命令：
  ```
  python scripts/raw_workflow.py --elements sea_ice_s2_arctic --outdir outputs/raw_pair15_s2ice --api http://127.0.0.1:8078
  ```
- 数据：CDSE 下载的 S2 L2A 单景（瓦片 T33XWJ，巴伦支海）
  - 2025-05-30 场景 218 MB（化冰期，无冰）
  - 2025-03-25 场景 986 MB（有冰）
- 判据：窗口内 NDSI>0.4 的像元比例 ≥15% 判 ice，否则 open_water
- 实测：**4/4**：3 月两窗冰比例 100%、62.3% 均判 ice；5 月两窗冰比例 0% 均判 open_water

## 30. flood_water（Sentinel-1 GRD，巴格达/孟加拉湾一带，2024-09-04）

- 命令：
  ```
  python scripts/raw_workflow.py --elements flood_water --outdir outputs/raw_pair16_grd --api http://127.0.0.1:8078
  ```
- 数据：CDSE 单景 S1 GRD（1253 MB）
- 判据：窗口内归一化后向散射 <60 的暗区占比 >0.35 判 water，<0.10 判 land
- 实测：1/2。暗区 82.7% 判 water 正确；暗区 6.9% 的陆窗被判 water（假阳性）

## 31. green_tide（Sentinel-2 L2A，FAI，黄海 2024-08-30）

- 命令：
  ```
  python scripts/raw_workflow.py --elements green_tide --outdir outputs/raw_pair17_gtide --api http://127.0.0.1:8078
  ```
- 数据：CDSE 单景 S2 L2A（T51SUU，816 MB）
- 判据：逐景自适应阈值（清澈海水 FAI 中位数 + 2σ）；窗内超阈占比 ≥0.15 green_tide / 0.03–0.15 patchy / <0.03 clean_water
- 实测：**2/2**：超阈占比 3.4% 判 patchy、0.2% 判 clean_water

## 32. hurricane_opt（VIIRS L1B I05 亮温，2024-09-04）

- 命令：
  ```
  python scripts/raw_workflow.py --elements hurricane_opt --outdir outputs/raw_pair18_typh --api http://127.0.0.1:8078
  ```
- 数据：LAADS VJ102IMG 单景（约 260 MB；扫了 6 景，选冷云最强的一景，min BT 183.5 K）
- 判据：窗内亮温 ≤200 K 的冷云占比 ≥30% mature / 2%–30% developing / <2% weak
- 实测：**2/2**：冷云占比 0.4% 判 developing、0.0% 判 weak

## 33. snow_fsc（VIIRS NDSI→FSC，内蒙古 2024-02-20）

- 命令：
  ```
  python scripts/raw_workflow.py --elements snow_fsc --outdir outputs/raw_pair19_fsc --api http://127.0.0.1:8078
  ```
- 数据：VJ102MOD L1B + GEO 单轨（LAADS，本次约 2 个 granule）
- 判据：FSC = 0.5·tanh(2.65·NDSI−1.42)+0.5，窗内平均 FSC <0.02 无雪 / 0.02–0.20 轻 / 0.20–0.60 中 / ≥0.60 重
- 实测：**1/1**：平均 FSC 0.468 判 moderate 正确

## 34. coastal_sla（NASA SSH 格网 SSHA，替代 CMEMS）

- 命令：
  ```
  python scripts/raw_workflow.py --elements coastal_sla --outdir outputs/raw_pair21_coastal_sla --api http://127.0.0.1:8078
  ```
- 数据：PODAAC NASA_SSH_REF_SIMPLE_GRID_V11（单景 1.1 MB，取 15 个日期），变量 `ssha`
- 判据：窗内平均 SLA 百分位 <90 none / 90–95 light / 95–98 moderate / ≥98 heavy
- 实测：**2/2**，两个窗百分位均为 0（平静期，只验证了负例）
- 说明：原训练数据是 CMEMS L4；本机无 CMEMS 账号，改用同物理量的 PODAAC 格网 SSHA

## 35. coastal_nt（NASA SSH 格网 SSHA，30° 分区）

- 命令：
  ```
  python scripts/raw_workflow.py --elements coastal_nt --outdir outputs/raw_pair21_coastal --api http://127.0.0.1:8078
  ```
- 数据：同上，最后一日的 SSHA 场
- 判据：30° 分区内 |max SLA| ≥0.30 m high_water / 0.15–0.30 elevated / <0.15 normal
- 实测：**2/2**：最大 SLA 0.645 m 判 high_water、0.157 m 判 elevated
- 说明：同上，产品替代

## 36. earthquake_def（ARIA GUNW 解缠相位）

- 命令：
  ```
  python scripts/raw_workflow.py --elements earthquake_def --outdir outputs/raw_pair22_eqd --api http://127.0.0.1:8078
  ```
- 数据：ASF/ARIA GUNW 单景（土耳其 2023-02-22 vs 2023-02-10，54 MB）；相位按 λ/4π 换算 LOS 位移
- 判据：|中位位移| <5 mm stable / 5–30 slight_deform / ≥30 obvious_deform
- 实测：1/2。中位 76.8 mm 判 obvious_deform 正确；中位 0.0 mm 的窗被判 slight_deform（假阳性）

## 37. ground_subsidence（本地衡水 InSAR ENVI 位移产品）

- 命令：
  ```
  python scripts/raw_workflow.py --elements ground_subsidence --outdir outputs/raw_pair20_insar --api http://127.0.0.1:8078
  ```
- 数据：`outputs/henshui_pair01`、`outputs/henshui_6month` 的 displacement_VV.img
- 判据：|P90 位移| <5 mm stable / 5–10 slight / ≥10 obvious
- 实测：2/4。28.0、24.1 mm 两窗真值 obvious，模型判 stable（漏检）；0 mm 两窗判 stable 正确

## 38. groundwater_overuse（本地衡水 InSAR ENVI 位移产品）

- 命令：
  ```
  python scripts/raw_workflow.py --elements groundwater_overuse --outdir outputs/raw_pair20_insar --api http://127.0.0.1:8078
  ```
- 数据：同上
- 判据：|中位位移| <5 mm stable / 5–15 slight / ≥15 obvious
- 实测：2/4。两个 0 mm 窗中一个判 stable 正确、另一个被判 slight/obvious（假阳性）
- 说明：训练用的喀斯特区产品本地已删，本次用衡水产品；该产品窗口内中位位移多为 0，正例不足

## 进度

| 汇总 | 数 |
|---|---|
| 链路已通 | 38 / 38 |
| 验证通过 | 19 / 38 |
| 待接 | 0 |
| 说明 | coastal 两个用 PODAAC SSH 替代 CMEMS；earthquake 用 ARIA GUNW；SAR 沉降用本地 ENVI 产品 |


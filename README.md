# 遥感灾害识别元素库（vlmdisaster）

把不同遥感数据源的**物理量**渲染成带色标的指数图，用一个 4B 多模态基座 + 每个灾种一个 LoRA读出
**有没有灾、多重、判据是什么**，再交给大模型生成中文研判。

- **38 个元素**（灾种 × 数据源），全部通过输出质量审计，27 个可直接用
- 未见日期验证：按年 9/9、按日期 12/14 过线；全元素端到端自测 38/38
- 演示服务：按 10 类采集方式分栏，上传图 → 结构化判定 → 逆地理编码 → 中文研判

> 适用边界：模型输入是**渲染后的指数图**（带色标），不是任意照片；自然影像需先经过渲染管线。

---

## 一、原理

### 1. 每个源先看物理量

| 源 | 观测量 | 渲染成 | 本仓库全部元素（数量） |
|---|---|---|---|
| 光学 OPT（Sentinel-2 / MODIS） | 表观反射率 | NDVI / NDSI / NDWI / FAI / 叶绿素 / 云顶高 | 洪水·光学、干旱、绿潮、积雪覆盖率、水华、海冰·光学、台风（7 个） |
| 热红外 TIR（MODIS LST / OISST） | 亮温 → 地表/海表温度 | LST/SST 距平、百分位、亮温异常 | 野火、印度洋偶极、海洋热浪、珊瑚白化、ENSO、火山热异常、热浪（7 个） |
| 被动微波 MW（SMAP / AMSR2） | 亮温 | 土壤水分、积雪当量、海冰密集度 | 内涝、积雪当量、海冰·微波、暴雪（4 个） |
| 雷达降水 PRECIP（GPM / IMERG） | 后向散射 → 降水率 | 日/多日累计降水 | 暴雨、洪水·降水、台风降水（3 个） |
| 测高 ALT（卫星高度计） | 海面高度 | 海面高度异常 SLA | 海岸高水位、海岸夜间热、巨浪（3 个） |
| 重力 GRACE | 等效水高 | 陆地水/地下水距平、百分位 | 干旱、地下水（2 个） |
| SAR（Sentinel-1） | 后向散射 / 干涉相位 | 形变速率、相干性 | 地震形变、地下水超采、地面沉降、洪水范围（4 个） |
| 大气 ATM（OMI / MOD08） | 吸收性气溶胶 / SO₂ 柱量 | UVAI、SO₂ 浓度 | 野火烟羽、沙尘气溶胶、火山SO2、沙尘暴、雾/霾（5 个） |
| 散射计 SCAT | 后向散射 → 风场 | 10 m 风速 | 海上大风（1 个） |
| 空间天气 SPACE（NOAA / GFZ） | X 射线通量 / 地磁指数 | 耀斑等级、Kp | 太阳耀斑、地磁暴（2 个） |

### 2. 渲染图即接口

每个元素把物理量渲染成固定色标的图：**色标就是图例**，阈值写进模型输出的 `criterion` 字段。
原则：图上没有色标的量不进输出；均匀场用百分位/距平；长期趋势场先去除趋势。

### 3. 一基座 + 多挂件

同一 **Qwen3-VL-4B**（4bit 量化）承载 38 个 rank-4 LoRA，每个元素一个。
换元素只换挂件，不换基座；输出统一为 JSON：`disaster_type / detected / source / physical_quantity / severity / criterion`（部分元素另有数值字段）。

### 4. 下游

结构化 JSON → 离线逆地理编码（中国到区县；国外到国家/一级行政区+最近聚落；任何位置返回坐标）→ 大模型研判（单元素报告 / 多元素综合评估）→ 前端展示。

---

## 二、结果速览

| 指标 | 数值 |
|---|---|
| 采用元素 | 38（10 类采集方式） |
| 质量审计通过 | 38 / 38 |
| 检出 det ≥ 70% | 37 / 38 |
| 按年留出 | 8 / 8 |
| 按天留出 | 11 / 12 |
| 时间留出 | 9 / 10 |
| 未验证（无日期字段 / 日期与类别耦合） | 8 |
| 端到端实跑（每元素一样例） | 38 / 38，421 秒 |

逐元素结果见 Releases 的 `eval_evidence.zip`；元素清单见根目录 [`elements_capability.json`](elements_capability.json)。

---

## 三、数据出处

全为公开数据源，渲染后删除原始文件；凭证不入库（`scripts/credentials_local.py` 与 `config/llm.json` 已忽略）。

| 源 | 产品/平台 | 获取 |
|---|---|---|
| Sentinel-2 L1C | Copernicus CDSE | `scripts/download_sentinel2.py`、`download_s2_requests.py`（需 CDSE 账号） |
| MODIS NDVI | MOD13 | `scripts/download_modis_ndvi.py` |
| MODIS LST | MOD11C3 | `scripts/download_mod11c3.py` |
| MODIS 大气 | MOD08 | `scripts/download_mod08.py` |
| GPM / IMERG | NASA GES DISC | `scripts/download_imerg_flood_days.py`、`download_gpm.py`（需 Earthdata token） |
| SMAP | NSIDC | `scripts/download_smap_token.py` |
| AMSR2 海冰/积雪 | NSIDC | `scripts/download_swe_amsr2.py` |
| GRACE / GRACE-DA | JPL | `scripts/download_grace.py` |
| 卫星测高 | CMEMS / PO.DAAC | `scripts/download_cmems.py` |
| OMI SO₂ / UVAI | NASA GES DISC | `scripts/download_omi_so2.py`、`download_tropomi.py` |
| 太阳 X 射线 / 地磁 | NOAA SWPC / GFZ | 公开数据（无需下载脚本） |
| 行政区划（全球） | Natural Earth admin-0 / admin-1 | 已入库 `data/geo/global/`（177 国家 + 4089 一级行政区，简化版） |
| 行政区划（中国） | 阿里云 DataV GeoAtlas | 已入库 `data/geo/`（省 34 + 地级 475 + **区县 2728**） |

**授权**：CDSE、Earthdata、NSIDC、PO.DAAC 均为公开研究数据，遵守各自使用条款。

下载脚本从 `scripts/credentials_local.py` 读取账号/token（该文件已 gitignore）：
复制 `scripts/credentials_local.example.py` 为 `credentials_local.py`，填入自己的 Earthdata 账号与 token；也可改用环境变量 `EARTHDATA_USER` / `EARTHDATA_PASS`。

**已知缺口（未做的源）**：雷暴×闪电需 GOES GLM 分钟级数据（量级过大）；沙尘亮温差需 MODIS L1B（体积过大，改用 OMI 吸收性气溶胶）；
滑坡范围真值需 EM-DAT/COOLR 注册；太阳耀斑缺少同周期静默样本（检出无法验证）。

---

## 四、使用方式

### 1. 克隆

```bash
git clone https://github.com/shiro2398731/vlm.git
```

脚本默认以 `C:\vlmdisaster` 为根目录（约 370 处写死）。**不要改代码**，执行一次：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

它会在 `C:\vlmdisaster` 建立指向本仓库的目录联接（junction），脚本即可直接运行。

### 2. 下载权重与标签（Releases）

| 文件 | 内容 | 体积 |
|---|---|---|
| `adapters_slim.zip` | 38 个 LoRA（`adapter_config.json` + `adapter_model.safetensors`） | 约 200 MB |
| `datasets_json.zip` | 38 元素的训练/验证标签 JSON + `dataset_info.json` | 约 1.5 MB |
| `eval_evidence.zip` | 评估结果 + 端到端自测报告（可选） | 约 6 KB |

> adapters 是**训练产物**；datasets 是**训练输入**（渲染器产出的标签 JSON，不含图片）。

### 3. 解压后的目录

把 `adapters_slim.zip` 与 `datasets_json.zip` 都解压到 **`C:\vlmdisaster\`**（仓库根/联接目录）：

```text
C:\vlmdisaster\
├─ scripts\  src\  web\  configs\  demo_samples\
├─ elements_capability.json
├─ outputs\                          ← adapters_slim.zip 解压到这里
│   ├─ qwen3vl-lora-flood_opt-json\
│   │   ├─ adapter_config.json
│   │   └─ adapter_model.safetensors
│   ├─ qwen3vl-lora-heatwave_lst-json\
│   └─ ...（共 38 个）
└─ LLaMA-Factory\
    └─ data\                         ← datasets_json.zip 解压到这里
        ├─ train_flood_opt_json.json
        ├─ train_flood_opt_json_val.json
        ├─ ...
        └─ dataset_info.json
```

校验：

```powershell
(Get-ChildItem C:\vlmdisaster\outputs -Directory -Filter 'qwen3vl-lora-*-json').Count   # 期望 38
(Get-ChildItem C:\vlmdisaster\LLaMA-Factory\data -Filter '*.json').Count                # 期望 77
```

### 4. 环境与基座

```powershell
py -3.11 -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

基座单独下载 **Qwen3-VL-4B-Instruct**（约 8 GB）。推荐 ModelScope（国内直连，不需要 VPN）：

```powershell
pip install modelscope
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir C:\vlmdisaster\models\Qwen3-VL-4B
```

或用 HuggingFace（需能访问 hf.co，也可设镜像 `$env:HF_ENDPOINT="https://hf-mirror.com"`）：

```powershell
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir C:\vlmdisaster\models\Qwen3-VL-4B
```

- 默认读取 `models\Qwen3-VL-4B`；放在别处就改 `scripts/serve.py` 顶部 `MODEL` 常量
- 4bit 量化推理，显存 ≥ 6 GB 可跑；首次加载约 1–2 分钟

### 5. 启动演示

```powershell
.\venv\Scripts\python -m uvicorn scripts.serve:app --host 127.0.0.1 --port 8078
# 浏览器打开 http://127.0.0.1:8078
```

首次推理加载 4B 基座约 1–2 分钟，之后切换元素秒级。

### 6. 页面怎么用

- 顶部按**采集方式**分栏（光学 / 热红外 / 被动微波 / 雷达降水 / 测高 / 重力 / 散射计 / 空间天气 / SAR / ATM）
- 每个元素带样例缩略图（优先「训练集之外的日期」，标橙字）；点缩略图即推理
- 也支持上传自己的渲染图；文件名含词根会自动推荐元素（如 `*_ndwi` → 洪水）
- 跑 2–3 个元素后可点「汇总研判」，由大模型给综合风险与处置建议
- 键盘：`1`–`9` 切栏、`S` 跑样例、`A` 汇总

接口：`/api/catalog`、`/api/predict`、`/api/report_aggregate`、`/api/route`、`/api/status`、`/api/llm`、`/api/artifacts`。

前端结构、接口对应与定制方法见 [`docs/前端说明.md`](docs/前端说明.md)。

另有一套 **Gradio 界面**（`web/gradio_app.py`），与主界面共用同一后端、不重复加载模型：

```powershell
# 主服务保持运行，然后另开一个终端
.\venv\Scripts\python web\gradio_app.py --port 7860 --api http://127.0.0.1:8078
```

打开 http://127.0.0.1:7860 即可。

两套前端都已接大模型：结果区的后端可切 `template` / `llm`，选 `llm` 时由配置的大模型生成中文研判（失败自动降级模板）。

### 7. 逆地理编码（全球，离线）

- **中国（到区县）**：`data/geo/china_provinces.json`（34 省）+ `china_cities.json`（475 地级）+ `china_counties_compact.json`（**2728 个区县**，约 8 MB）
- **全球（到一级行政区）**：`data/geo/global/ne_110m_admin_0_countries.geojson`（177 国）+ `admin1_compact.geojson`（4089 个省/州，约 4.7 MB）
- **最近聚落**：`data/geo/global/places_compact.json`（7342 个城镇，含坐标与人口），任何位置都返回「最近聚落 + 距离 km」
- 实现：纯 Python 射线法，无 shapely/网络依赖；1° 网格索引 + bbox 粗筛；行政区未命中时用最近中心兜底（`approx: true`）
- 任何坐标都原样返回 `lat / lon`；公海或荒野返回坐标 + 最近聚落与距离（远距离标 `far: true`）

```python
from scripts.reverse_geocode import geocode
geocode(30.5928, 114.3055)   # 中国：湖北省 / 武汉市 / 江岸区
geocode(29.6520, 91.1721)    # 中国：西藏自治区 / 拉萨市 / 城关区
geocode(35.6762, 139.6503)   # 日本：Japan / Tokyo · 最近聚落 Tokyo 9 km
geocode(-23.5505, -46.6333)  # 巴西：Brazil / São Paulo · 1 km
geocode(23.4, 12.5)          # 撒哈拉：Libya + 最近聚落 Tajarhi 228 km（far）
geocode(0.0, -140.0)         # 公海：坐标 + 最近聚落 Papeete 2223 km（far）
```

- 返回字段：`country`、`province`（中国省/直辖市，国外州/省/大区）、`city`、`county`（仅中国）、`lat`/`lon`、`nearest_place{name,country,distance_km,far}`、可选 `approx`；另附 `display` 可直接展示的字符串
- 说明：中国精度到区县；**国外到国家 + 一级行政区（省/州）**，并给最近聚落与距离，任何位置都带精确坐标；台湾省边界接口抓取失败，全球库按国家边界覆盖。

```text
中国 / 湖北省 / 武汉市 / 江岸区 @ 30.5928, 114.3055
Japan / Tokyo（最近聚落 Tokyo 9.0 km） @ 35.6762, 139.6503
Libya（最近聚落 Tajarhi 228.5 km，偏远） @ 23.4000, 12.5000
```

### 8. 大模型研判（可选）

结构化判定交给大模型写成中文研判。支持任何 OpenAI 兼容接口（默认 DeepSeek）。

```powershell
# 命令行配置
.\venv\Scripts\python scripts\llm_config.py set --api-key sk-xxx --model deepseek-flash
.\venv\Scripts\python scripts\llm_config.py test
```

- 网页里也能配（页面「大模型接口配置」面板，key 只显示掩码）
- 配置存 `config/llm.json`（已 gitignore）
- 调用失败自动降级为模板研判，结果标 `backend: template(fallback: ...)`，不中断演示
- 多元素综合研判：`POST /api/report_aggregate`，把多个元素结论交给模型给综合风险与建议

### 9. 训练与评估（可选）

```powershell
.\venv\Scripts\python scripts\register_element.py --element <元素> --train <train.json> --val <val.json>
.\venv\Scripts\python scripts\preflight_check.py --element <元素>
.\venv\Scripts\python scripts\retrain_safe.py --elements <元素>
.\venv\Scripts\python scripts\eval_batch.py <元素> --limit 200 --batch 6
.\venv\Scripts\python scripts\quality_audit.py --limit 80 --batch 6 --elements <元素>
```

样本图片需按第三节的数据源自行下载，用 `src/preprocess/` 的渲染器生成。

---

### 10. 脚本一览（scripts/）

| 脚本 | 作用 / 用法 |
|---|---|
| **— 服务与演示 —** | |
| `serve.py` | 演示服务：接口 + 前端静态；`python -m uvicorn scripts.serve:app --port 8078` |
| `demo_check.py` | 上台前自检（服务/元素/样例/8 接口/大模型）：`python scripts/demo_check.py [--warm]` |
| `demo_run.py` | 彩排：光学洪水→热红外热浪→重力地下水→汇总研判 |
| `e2e_all.py` | 全元素端到端自测：`python scripts/e2e_all.py [--elements flood_opt,...]` |
| `smoke_test.py` | 交付一致性：38 元素/10 组/接口/模型 |
| `start_demo.ps1` | 一键开演（Windows）：端口检查→uvicorn→等就绪→自检 |
| **— 逆地理与注册表 —** | |
| `reverse_geocode.py` | 逆地理：中国省/市/区县 + 全球国家/省州 + 最近聚落；单独运行可自检示例 |
| `geo_global.py` | 全球边界查询（被 reverse_geocode 调用，也可单独运行看示例） |
| `elements_registry.py` | 读 `elements_capability.json`（服务与检查脚本共用） |
| **— 大模型研判 —** | |
| `report_llm.py` | 单元素/多元素中文研判（模板或大模型，失败自动降级） |
| `llm_config.py` | 配置大模型：`set / show / test` |
| **— 原始影像链路 —** | |
| `raw_workflow.py` | 统一入口：原始产品→渲染→判读→阈值；`--elements a,b --outdir ... --api ...` |
| `raw_scene_runner.py` | CDSE 单景下载→解压→渲染（S2 水体指数示例） |
| `raw_s2_to_ndwi.py` | S2 SAFE→NDWI 渲染图 |
| `validate_raw_workflow.py` | 渲染图批量调接口，与阈值标签对照 |
| `check_split_integrity.py` | 检查训练/测试是否同日期、同图（切分合法性） |
| `raw_rescore_boundary.py` | 按“距阈值余量”重算一致率（`REL=0.1/0.2/0.3`） |
| **— 训练与评估 —** | |
| `register_element.py` | 注册数据集到 LLaMA-Factory：`--element --train --val` |
| `preflight_check.py` | 训练前检查（样本数/字段/图片可读）：`--element` |
| `retrain_safe.py` | 备份后重训：`--elements a,b` |
| `eval_batch.py` | 批量评估 det/sev：`<元素> --limit 200 --batch 6` |
| `eval_single.py` | 单元素快速评估（简化版） |
| `quality_audit.py` | 质量审计（解析/字段/塌缩/一致性）：`--limit 80 --batch 6 --elements a,b` |
| `batch_to_json.py` | 把渲染产出的 CSV 转成训练 JSON 标签 |
| **— 数据下载（需自备账号/token） —** | |
| `dl_common.py` | 下载公共函数（请求/重试/鉴权），被 `download_*` 共用 |
| `config_local.py` | **凭据模板（全空）**：填自己的账号/token，或改用环境变量 |
| `download_sentinel2.py` | Sentinel-2 L1C（CDSE 账号） |
| `download_sentinel1.py` | Sentinel-1 SAR（ASF/CDSE） |
| `download_s2_requests.py` | Sentinel-2 批量请求（CDSE） |
| `download_modis_ndvi.py` | MODIS NDVI（Earthdata） |
| `download_mod11c3.py` | MODIS LST（MOD11C3） |
| `download_mod08.py` | MODIS 大气（MOD08） |
| `download_lst.py` | LST 其他产品/补采 |
| `download_imerg_flood_days.py` | GPM IMERG 日降水（Earthdata token） |
| `download_gpm.py` | GPM 降水（通用入口） |
| `download_smap_token.py` | SMAP 土壤水分（Earthdata token） |
| `download_swe_amsr2.py` | AMSR2 积雪当量/海冰 |
| `download_grace.py` | GRACE/GRACE-DA 等效水高 |
| `download_omi_so2.py` | OMI SO₂ 柱量 |
| `download_tropomi.py` | TROPOMI 气溶胶/SO₂（被 OMI 脚本引用） |
| `download_ascat.py` | ASCAT 海面风场 |
| `download_cmems.py` | CMEMS 海面高度 SLA |
| `download_firms_cn.py` | FIRMS 火点（中国区） |

### 10. 训练超参数与质量审计

**训练超参数（各元素一致）**

| 项目 | 设置 |
|---|---|
| 微调方式 | SFT + LoRA |
| LoRA rank / dropout | 4 / 0.05 |
| 作用层 | q_proj, v_proj |
| 量化 | 4 bit（nf4，双量化） |
| 学习率 / 调度 | 4e-5 / cosine，warmup 0.1 |
| 批大小 | 1 × 梯度累积 16 |
| 训练轮数 | 2 |
| 截断长度 | 512 |
| 图像像素上限 | 131,072 |

**质量审计规则**（每元素训练后自动执行，任一不通过则不采用）

| 检查项 | 判定方式 |
|---|---|
| 输出可解析 | 模型输出能解析为合法 JSON |
| 字段齐全 | 按该元素提示词逐项核对必需字段 |
| 无塌缩 | 不出现“全部判为同一类”的退化 |
| 自洽 | detected 与 severity 不矛盾 |
| 数值合理 | 水占比、火点数、温度等落在物理范围内 |
| 判据复述 | criterion 字段与训练数据中的判据一致 |

### 11. 原始影像链路（新增）

38 个元素已全部建立“原始产品 → 渲染 → 判读 → 阈值对照”的链路，统一入口：

```bash
python scripts/raw_workflow.py --elements snow_swe,wind_gale --outdir outputs/check --api http://127.0.0.1:8078
```

逐元素命令、渲染结果与模型输出见 [docs/raw_workflow/raw_chain_status.md](docs/raw_workflow/raw_chain_status.md)。

| 指标 | 数值 |
|---|---|
| 链路已建立 | 38 / 38 |
| 全部测试窗口一致 | 19 / 38 |
| 样本级切分干净（测试图不在训练集） | 38 / 38 |
| 日期级切分干净（测试日期不在训练集） | 7 / 38 |

说明：

- 每元素仅 2–6 个窗口（洪水光学 21 个、野火 12 个），只验证链路可用性，不构成统计准确率
- 31 个元素测试日期与训练日期重叠，只能算“同样场景、不同窗口”，不是跨日期泛化
- 服务层修复：多 adapter 切换曾因共用 adapter 名导致权重污染，现改为每元素独立 adapter 名（训练评估脚本 `eval_all_adapters_v2.py` 原本已处理同类问题）

补充文档：

- [render_consistency.md](docs/raw_workflow/render_consistency.md)：训练样本与新渲染图的像素统计对照
- [severity_experiment.md](docs/raw_workflow/severity_experiment.md)：模型读数套阈值 vs 直接 severity
- [split_integrity.md](docs/raw_workflow/split_integrity.md)：切分合法性检查
- [quality_plan.md](docs/raw_workflow/quality_plan.md)：逐元素质量提升清单

## 五、目录

| 路径 | 内容 |
|---|---|
| `scripts/` | 下载、渲染、注册、训练、评估、审计、服务、自检 |
| `src/preprocess/` | 各元素渲染器（每个源一个物理量） |
| `web/` | 两套前端：`index.html`（主，分栏/样例/汇总）+ `gradio_app.py`（Gradio 版）；说明见 `docs/前端说明.md` |
| `configs/` | 训练与数据集配置示例 |
| `demo_samples/` | 页面展示的 114 张渲染样例（按源/元素分目录，本地数据集缺失时前端回退用它） |
| `data/geo/` | 离线边界：`global/`（国家 + 一级行政区 + 聚落点）、中国省/地级/区县 |
| `docs/` | 原理说明：十类源采集/渲染/判读 + 38 元素判据 |
| `elements_capability.json` | 38 元素能力表（脚本与前端的数据源） |

## 六、原理参考

- 逐元素原理：`docs/采集与渲染原理.md`（十类源「采集→渲染→判读」+ 38 元素判据原文）
- 元素能力表：`elements_capability.json`（下节字段说明）

## 七、elements_capability.json

字段：

| 字段 | 含义 |
|---|---|
| `zh` / `source` | 中文名 / 数据源代号（OPT/TIR/MW/PRECIP/ALT/GRACE/SCAT/SPACE/SAR/ATM） |
| `n` | 验证集样本数 |
| `det` / `sev` | 同分布下的检出/分级准确率（%） |
| `validation` | 验证方式：按年留出 / 按天留出 / 时间留出 / 未验证原因 |
| `holdout_det` / `holdout_sev` | 未见日期成绩（无可验证数据时为空） |

## 七、限制

- 输入必须是渲染后的指数图；自然照片需前置渲染
- 未见日期测试用的是我们渲染的指数图（不是原始卫星影像、也不是网络图片）
- 页面展示的样例图随仓库提供（`demo_samples/`，114 张）；本地 `datasets/` 缺失时页面自动回退到仓库副本
- 8 个元素缺有效日期验证（无日期字段 / 日期与类别耦合）
- 分级（severity）普遍弱于检出（det）
- 海上大风、洪水范围两个元素跨日期未达标，仅作演示
- 原始影像链路只做了小样本（每元素 2–6 窗），19/38 全窗一致，不构成统计准确率
- 38 个元素中仅 7 个的测试日期与训练日期完全分离，其余 31 个只能算“同样场景、不同窗口”

## 许可

MIT（见 `LICENSE`）。

---

## 多模态三路特征融合（vlm-fusion）

在「物理指数图 → Qwen」之上，按老师要求增加两路深度网络特征，
三路特征**并发**作为输入交给 Qwen 做多模态协同分类：

```
物理：指数渲染图 + 每样本物理统计量（数值特征）
视觉：ResNet50 / DINOv2 → 每样本视觉特征（分数 + 16 维向量摘要）
语义：CLIP → 每样本语义特征（分数 + 16 维向量摘要）
                ↓ 三路并发
        Qwen3-VL 基座（权重不变）
                ↓
        结构化判读 JSON + LLM 研判 + 逆地理编码
```

### 三路输入的形式

| 路径 | 特征（分数） | 特征向量 | 默认 |
|---|---|---|---|
| 数值 | 每样本物理量（归一化） | 物理量向量 | 开 |
| 视觉 | 每元素探针的类别概率 | PCA-16 摘要 | 开 |
| 语义 | 每元素 CLIP 探针概率 | PCA-16 摘要 | 开 |

三路的分数与向量都写进输入文本（`extra_text` 通道），与指数图一起送入 Qwen。
开关：`include_vectors` / `include_visual_vectors` / `include_semantic_vectors`。

### 融合模式

| 模式 | 做法 | 结论 |
|---|---|---|
| A 文本级 | 三路分数写进 prompt | 部分元素会拖累，默认关闭 |
| B 图像级 | 视觉特征图并排/叠加成图 | 会破坏挂件判读，默认关闭 |
| C 决策级 | Qwen 判读 + 探针高置信覆盖（tau 0.6） | **采用**，32 元素 18 升 13 平 1 降 |
| D 嵌入级 | projector → token | 未实现，列为后续 |

### 验证结果速览

- 同分布（32 个元素，2 张/类）：**18 升、13 平、1 降**
- 训练集外日期：wave 2/8→3/8、earthquake_def 4/6→6/6、groundwater 4/6→4/6
- snow_fsc 外部 baseline 8/8，融合会拉低 → 该元素关闭融合
- CLIP 零样本对渲染图无效（coral 0%、snow_fsc 31%），改「嵌入 + 探针」后 93–94%

### 配置与文档

- 逐元素开关：[configs/fusion_selected.yaml](configs/fusion_selected.yaml)（35 元素中 21 个启用）
- 配置样例：[configs/fusion.example.yaml](configs/fusion.example.yaml)
- 架构说明：[docs/architecture_fusion.md](docs/architecture_fusion.md)
- 实验记录：[docs/four_group_results.md](docs/four_group_results.md)
- 逐元素汇总：[docs/decision_summary.md](docs/decision_summary.md)、[docs/probe_summary.md](docs/probe_summary.md)
- 独立附件：[docs/附件_三路特征融合架构.md](docs/附件_三路特征融合架构.md) 与同名 docx

### 复现命令

```bash
# 训练探针（逐元素）
python scripts/train_clip_probe.py coral_dhw snow_swe
python scripts/train_resnet_probe.py coral_dhw snow_swe
python scripts/train_dinov2_probe.py coral_dhw snow_swe

# 决策级融合评测
$env:TAU="0.6"; python scripts/eval_decision_fusion.py coral_dhw

# 四组对照（baseline / +视觉 / +语义 / +both）
$env:VEC="1"; python scripts/eval_four_groups_element.py coral_dhw
```

基线（vlm）完全未改动，所有结果与旧版本可比。

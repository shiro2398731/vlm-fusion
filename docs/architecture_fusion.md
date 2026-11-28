# 多模态融合架构（vlm-fusion）

目标：在现有「物理指数渲染图 → Qwen 判读」之上，按老师要求增加两路深度网络特征，
最终输入大模型的是 **物理特征 + 视觉特征 + 语义特征**，由 Qwen 综合分析。

基线（vlm）不变，本仓库只做加法。

## 1. 总体架构

```
                      原始产品 / 渲染图
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ 物理支路       │   │ 视觉支路         │   │ 语义支路          │
│ 指数渲染图     │   │ 冻结 CNN         │   │ 冻结 CLIP         │
│ + 窗口统计量   │   │ (ResNet/DINOv2)  │   │ 图像嵌入 ⇄ 类别文本│
│ (已有)         │   │ → 特征图/全局向量 │   │ → 相似度分布/嵌入  │
└───────┬───────┘   └────────┬────────┘   └─────────┬────────┘
        │                    │                      │
        └──────────┬─────────┴──────────┬───────────┘
                   ▼                    ▼
            ┌────────────────────────────────┐
            │ 融合层 Fusion                   │
            │ A 文本级 / B 图像级 / C 决策级   │
            │ D 嵌入级（projector → token）   │
            └───────────────┬────────────────┘
                            ▼
                 ┌────────────────────────┐
                 │ Qwen3-VL + 每元素挂件    │
                 │ 输出结构化 JSON 判读     │
                 └────────────────────────┘
```

### 三路特征的定位

| 支路 | 来源 | 输出 | 可训练 |
|---|---|---|---|
| 物理 | 现有渲染器 | 指数图、物理统计量（均值/分位/峰值） | 否（已冻结为基线） |
| 视觉 | 冻结 CNN（ResNet50 / DINOv2） | 特征图（PCA-RGB）或全局向量 | 否（提取器冻结） |
| 语义 | 冻结 CLIP（或其他图文模型） | 图像嵌入、与类别文本的相似度、top-k | 否（提取器冻结） |

语义特征说明：CLIP 也是深度网络，但它的输出是「图像与文字的相似度」，
与 CNN 的纹理/结构特征不同，因此单独成路，不并入视觉支路。

## 2. 融合模式

| 模式 | 做法 | 训练 | 用途 |
|---|---|---|---|
| A 文本级 | CLIP 相似度、物理统计量写成文本，拼进 prompt | 零训练 | v1 默认 |
| B 图像级 | 视觉特征图与指数图并排/叠加成一张图 | 零训练 | v1 默认 |
| C 决策级 | 三路各自出分数，加权/规则融合 | 零训练 | 备选 |
| D 嵌入级 | CNN/CLIP 向量经 projector 变成 Qwen 的额外 token | 需训练 projector | v2 |

v1 = A + B；v2 = D。C 作为对照。

## 3. 接口契约

### 3.1 FeatureBundle（三路特征的统一容器）

```python
@dataclass
class FeatureBundle:
    element: str
    image_path: str                 # 物理指数图
    physical_stats: dict            # 窗口统计量，如 {"mean_fsc": 0.28}
    visual_map: np.ndarray | None   # HxWx3 uint8，视觉特征图
    visual_vector: np.ndarray | None
    semantic_scores: dict | None    # {"no_snow": 0.1, "light": 0.2, ...}
    semantic_vector: np.ndarray | None
    meta: dict                      # 提取器名称、耗时、版本
```

### 3.2 FeatureExtractor

```python
class FeatureExtractor(ABC):
    name: str
    def load(self) -> None: ...
    def extract(self, image_path: str, labels: list[str] | None = None) -> FeatureBundle: ...
```

- `visual.*` 提取器只填 `visual_map/visual_vector`
- `semantic.*` 提取器只填 `semantic_*`
- 统一注册表：`features.register("clip_vit_b32", ClipExtractor)`

### 3.3 Fusion

```python
@dataclass
class FusionInput:
    images: list[str]     # 交给 Qwen 的图像（1 张或 2 张）
    text: str             # 追加到 prompt 的文本（语义分数/统计量）
    meta: dict

class Fusion(ABC):
    mode: str
    def compose(self, bundle: FeatureBundle) -> FusionInput: ...
```

### 3.4 元素管线

```python
class ElementPipeline:
    def __init__(self, cfg: dict): ...
    def build(self): ...        # 加载提取器与融合器
    def run(self, image_path: str) -> dict:  # 返回融合输入与模型输出
        ...
```

## 4. 配置

```yaml
snow_fsc:
  physical: {source: raw_workflow, renderer: snow_fsc}
  visual:
    extractor: resnet50          # none | resnet50 | dinov2_small | dinov2_base
    output: map                  # map | vector
  semantic:
    extractor: clip_vit_b32      # none | clip_vit_b32 | clip_vit_l14
    labels: [no_snow, light, moderate, heavy]
    prompt_template: "a satellite snow-cover map showing {label} snow"
  fusion:
    mode: [text, image]          # text / image / decision / embedding
  eval:
    backend: local               # local | api
    model: qwen3-vl-4b
```

## 5. 评估口径

每个元素固定四组对照：

| 组 | 输入 |
|---|---|
| baseline | 仅物理指数图 |
| +视觉 | 指数图 + 视觉特征图 |
| +语义 | 指数图 + 语义分数文本 |
| +视觉+语义 | 全部 |

指标：det、sev、数值偏差；测试样本必须来自训练集之外的日期；
每元素记录提升与上限，不把同日期样本当泛化证据。

## 6. 实现顺序

1. 接口与骨架（本次）：`features/`、`fusion/`、`pipeline.py`、配置样例；`none` 提取器可复现 baseline
2. 视觉：先接 ResNet50（torchvision 自带），再接 DINOv2
3. 语义：接 CLIP（transformers 版），输出相似度
4. v1 对照实验：snow_fsc 一个元素跑四组
5. v2：projector + 额外 token，训练量最小化
6. 扩展到 det 高的其他元素

## 7. 已知约束

- 8GB 显存，Qwen 4B 4bit 常驻约 3.5GB；CNN/CLIP 需冻结且分批
- 不重训 Qwen 基线挂件，保证旧结果可比
- 语义模型的类别文本必须与元素判据一致，否则相似度无意义

## 8. 实现状态（2026-11-27）

| 部分 | 状态 |
|---|---|
| 数值支路 | 已实现（物理统计量→归一化向量+文本） |
| 视觉支路 | 已实现：ResNet50 嵌入 + 线性探针（验证 79.5%–96.6%） |
| 语义支路 | 已实现：CLIP 嵌入 + 线性探针（验证 81.2%–99.6%） |
| 文本级融合 | 已实现，但实测会拖累（模型跟文字走），默认关闭 |
| 图像级融合 | 已实现（并排/叠加），实测破坏 Qwen 判读，默认关闭 |
| 决策级融合 | 已实现：Qwen 判读 + 探针高置信覆盖（tau 0.6），9 元素 8 升 1 平 |
| 逐元素开关 | `configs/fusion_selected.yaml`：仅对验证有增益的元素启用 |
| 训练集外复核 | snow_fsc 外部 8 张：探针 6–7/8；baseline 8/8，故该元素关闭融合 |

结论：老师要求的“深度网络补视觉特征 + 语义特征 + 大模型协同分类”，
在当前算力下落地为“冻结深度网络嵌入 + 每元素轻量探针 + 决策级协同”，
不需要重训 Qwen，baseline 完全不动。

后续：需要“特征接入模型内部”的端到端版本时，再训练 projector（8GB 可行但周期长）。

## 9. 可配置后端（本地 / API）

| 组件 | 本地 | 远程 |
|---|---|---|
| 语义支路 | `clip_probe`：CLIP 嵌入 + 每元素线性探针 | `clip_api`：POST 图片 base64，返回 `{embedding}` 或 `{scores}` |
| 判读 LLM | `llm.backend: local`：本机 serve 的 Qwen3-VL | `llm.backend: api`：OpenAI 兼容视觉接口，图片走 data URL |

配置字段：`semantic.extractor`、`semantic.base_url/api_key/model`、
`llm.backend/base_url/api_key/model`。环境变量：`SEMANTIC_API_URL`、
`SEMANTIC_API_KEY`、`LLM_API_KEY`。

验证脚本：`scripts/test_clip_api.py`、`scripts/test_llm_api.py`（mock 服务）。

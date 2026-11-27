# -*- coding: utf-8 -*-
"""融合层：三路特征 → Qwen 输入（图像 + 追加文本）。

可配置项：
  include_numeric / include_visual / include_semantic : 开关
  image_layout : side（并排）| overlay（叠加）
  visual_weight: 叠加时特征图权重
  max_text_lines: 文本预算
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from .base import Fusion, FusionInput


def _tmp_path(index_path, tag):
    d = os.path.join(os.path.dirname(os.path.dirname(index_path)), "fusion_tmp")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "%s_%s.png" % (os.path.basename(index_path)[:-4], tag))


def _blend(index_path, vmap, weight):
    base = Image.open(index_path).convert("RGB").resize((384, 384))
    vm = Image.fromarray(vmap).convert("RGB").resize((384, 384))
    return Image.blend(base, vm, max(0.0, min(1.0, weight)))


class TextImageFusion(Fusion):
    mode = "text+image"

    def __init__(self, include_numeric=True, include_visual=True,
                 include_semantic=True, image_layout="side",
                 visual_weight=0.35, max_text_lines=6,
                 semantic_style="scores", **kw):
        super().__init__(**kw)
        self.include_numeric = include_numeric
        self.include_visual = include_visual
        self.include_semantic = include_semantic
        self.layout = image_layout
        self.visual_weight = visual_weight
        self.max_text_lines = max_text_lines
        # scores: 只给分数，不给结论；hint: 追加“最相似类别”；both: 两者
        self.semantic_style = semantic_style

    def _numeric_line(self, bundle):
        if not self.include_numeric:
            return None
        meta = (bundle.meta or {}).get("numeric") or {}
        fields, vals = meta.get("fields"), meta.get("values")
        if fields and vals:
            pairs = ["%s=%.4g" % (k, v) for k, v in zip(fields, vals)]
            return "Physical statistics (normalized): " + ", ".join(pairs) + "."
        if bundle.physical_stats:
            return "Physical statistics: " + ", ".join(
                "%s=%.4g" % kv for kv in bundle.physical_stats.items()) + "."
        return None

    def _visual_line(self, bundle):
        if not (self.include_visual and bundle.visual_scores):
            return None
        top = sorted(bundle.visual_scores.items(), key=lambda x: -x[1])
        s = ", ".join("%s=%.3f" % (k, v) for k, v in top[:4])
        return "Visual model scores (deep network, not zero-shot): " + s + \
               ". Treat them as supporting evidence, not a label."

    def _semantic_line(self, bundle):
        if not (self.include_semantic and bundle.semantic_scores):
            return None
        top = sorted(bundle.semantic_scores.items(), key=lambda x: -x[1])
        s = ", ".join("%s=%.3f" % (k, v) for k, v in top[:4])
        head = "Semantic similarity scores: " + s + "."
        if self.semantic_style == "scores":
            return head + " Treat them as a weak prior; trust the image and physical values."
        if self.semantic_style == "hint":
            return head + " Most similar class: %s (hint only)." % top[0][0]
        return head + " Most similar class: %s (hint only)." % top[0][0]

    def compose(self, bundle) -> FusionInput:
        images = [bundle.image_path]
        lines = []

        if self.include_visual and bundle.visual_map is not None:
            if self.layout == "overlay":
                p = _tmp_path(bundle.image_path, "ovl")
                _blend(bundle.image_path, bundle.visual_map, self.visual_weight).save(p)
                images = [p]
                lines.append("Image is a blend of the physical index map and a deep-network "
                             "visual feature map.")
            else:
                p = _tmp_path(bundle.image_path, "vmap")
                Image.fromarray(bundle.visual_map).save(p)
                images = [bundle.image_path, p]
                lines.append("Second image: deep-network visual feature map of the same window; "
                             "use it together with the physical index map.")

        for fn in (self._numeric_line, self._visual_line, self._semantic_line):
            s = fn(bundle)
            if s:
                lines.append(s)

        lines = lines[:self.max_text_lines]
        return FusionInput(images=images, text="\n".join(lines),
                           meta={"mode": self.mode,
                                 "streams": {"numeric": bool(self._numeric_line(bundle)),
                                             "visual": self.include_visual and (
                                                 bundle.visual_map is not None or
                                                 bool(bundle.visual_scores)),
                                             "semantic": bool(bundle.semantic_scores)}})


class DecisionFusion(Fusion):
    """C 决策级：三路各出先验，后续可替换为分类头。"""

    mode = "decision"

    def compose(self, bundle) -> FusionInput:
        lines = []
        if bundle.physical_stats:
            lines.append("Physical prior: " + ", ".join(
                "%s=%.4g" % kv for kv in bundle.physical_stats.items()) + ".")
        if bundle.semantic_scores:
            top = max(bundle.semantic_scores.items(), key=lambda x: x[1])
            lines.append("Semantic prior: %s (%.3f)." % (top[0], top[1]))
        if bundle.visual_map is not None:
            lines.append("Visual feature map available but not verbalized in decision mode.")
        return FusionInput(images=[bundle.image_path], text="\n".join(lines),
                           meta={"mode": self.mode})


MODES = {"text+image": TextImageFusion, "decision": DecisionFusion}


def build(modes, **cfg):
    if any(m in ("text", "image", "text+image") for m in modes):
        return TextImageFusion(**cfg)
    if "decision" in modes:
        return DecisionFusion(**cfg)
    raise KeyError("unknown fusion modes: %s" % modes)

# -*- coding: utf-8 -*-
"""A 文本级 + B 图像级融合。"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from .base import Fusion, FusionInput


def _save_map(arr, index_path, tag):
    d = os.path.join(os.path.dirname(os.path.dirname(index_path)), "fusion_tmp")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "%s_%s.png" % (os.path.basename(index_path)[:-4], tag))
    Image.fromarray(arr).save(p)
    return p


class TextImageFusion(Fusion):
    """把语义分数与物理统计写进文本，把视觉特征图作为第二张图。"""

    mode = "text+image"

    def compose(self, bundle) -> FusionInput:
        images = [bundle.image_path]
        lines = []

        if bundle.physical_stats:
            kv = ", ".join("%s=%.4g" % (k, v) for k, v in bundle.physical_stats.items())
            lines.append("Physical statistics: %s." % kv)

        if bundle.semantic_scores:
            top = sorted(bundle.semantic_scores.items(), key=lambda x: -x[1])
            sem = ", ".join("%s=%.3f" % (k, v) for k, v in top)
            lines.append("Semantic similarity to class descriptions: %s." % sem)
            lines.append("The most similar class is %s; treat this as a hint, not a label."
                         % top[0][0])

        if bundle.visual_map is not None:
            images.append(_save_map(bundle.visual_map, bundle.image_path, "vmap"))
            lines.append("The second image is a deep-network visual feature map of the same window; "
                         "use it together with the physical index map.")

        return FusionInput(images=images, text="\n".join(lines), meta={"mode": self.mode})


class DecisionFusion(Fusion):
    """C 决策级：先应用规则，把三路分数合成一个先验文本（后续接分类头）。"""

    mode = "decision"

    def compose(self, bundle) -> FusionInput:
        lines = []
        if bundle.semantic_scores:
            top = max(bundle.semantic_scores.items(), key=lambda x: x[1])
            lines.append("Semantic prior: %s (score %.3f)." % (top[0], top[1]))
        if bundle.physical_stats:
            lines.append("Physical prior: %s." %
                         ", ".join("%s=%.4g" % kv for kv in bundle.physical_stats.items()))
        return FusionInput(images=[bundle.image_path], text="\n".join(lines),
                           meta={"mode": self.mode})


MODES = {
    "text+image": TextImageFusion,
    "decision": DecisionFusion,
}


def build(modes, **cfg):
    for m in modes:
        if m in ("text+image", "image", "text"):
            return TextImageFusion(**cfg)
    if "decision" in modes:
        return DecisionFusion(**cfg)
    raise KeyError("unknown fusion modes: %s" % modes)

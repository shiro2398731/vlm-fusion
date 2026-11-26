# -*- coding: utf-8 -*-
"""元素级多模态管线：物理 + 视觉 + 语义 → 融合 → Qwen 判读。"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List

from PIL import Image

from features import extractors as FE
from features.base import FeatureBundle
from fusion import layers as FL


def _composite(images: List[str]) -> str:
    """把多张图横向拼成一张，接口只收单图时使用。"""
    if len(images) == 1:
        return images[0]
    ims = [Image.open(p).convert("RGB") for p in images]
    w = sum(i.width for i in ims)
    h = max(i.height for i in ims)
    c = Image.new("RGB", (w, h), "white")
    x = 0
    for im in ims:
        c.paste(im, (x, 0)); x += im.width
    p = os.path.join(tempfile.gettempdir(), "fusion_%d.png" % os.getpid())
    c.save(p)
    return p


class ElementPipeline:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        vis = cfg.get("visual") or {}
        sem = cfg.get("semantic") or {}
        self.visual = FE.build(vis.get("extractor", "none"),
                               output=vis.get("output", "map"))
        self.semantic = FE.build(sem.get("extractor", "none"))
        self.labels = sem.get("labels") or []
        self.fusion = FL.build(cfg.get("fusion", {}).get("mode", ["text+image"]))

    def run(self, image_path: str, api: str, element: str, stats=None) -> Dict[str, Any]:
        bundle = FeatureBundle(element=element, image_path=image_path,
                               physical_stats=stats or {})
        bundle = self.visual.extract(bundle)
        bundle = self.semantic.extract(bundle, labels=self.labels)
        fused = self.fusion.compose(bundle)
        img = _composite(fused.images)
        # 服务端支持 extra_text 后即可直接拼接；当前为骨架，先按单图调用
        pred = self._predict(api, element, img, fused.text)
        return {"fusion": fused.text, "meta": bundle.meta, "prediction": pred}

    def _predict(self, api, element, image, extra_text):
        import requests
        with open(image, "rb") as f:
            r = requests.post(api.rstrip("/") + "/api/predict",
                              data={"element": element, "extra_text": extra_text},
                              files={"file": ("x.png", f, "image/png")},
                              timeout=180)
        try:
            return r.json()
        except Exception:
            return {"error": r.status_code, "raw": r.text[:200]}

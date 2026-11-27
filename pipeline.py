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
        self.element = cfg.get("element")
        num = cfg.get("numeric") or {}
        vis = dict(cfg.get("visual") or {})
        vis_name = vis.pop("extractor", "none")
        sem = dict(cfg.get("semantic") or {})
        self.numeric = FE.build(num.get("extractor", "numeric"),
                                scale=num.get("scale"), fields=num.get("fields"))
        self.visual = FE.build(vis_name, **vis)
        sem_name = sem.pop("extractor", "none")
        self.labels = sem.pop("labels", []) or []
        if sem_name == "clip_probe":
            sem.setdefault("element", self.element)
        self.semantic = FE.build(sem_name, labels=self.labels, **sem)
        self.fusion = FL.build(cfg.get("fusion", {}).get("mode", ["text+image"]),
                               **dict(cfg.get("fusion") or {}))

    def run(self, image_path: str, api: str, element: str, stats=None) -> Dict[str, Any]:
        bundle = FeatureBundle(element=element, image_path=image_path,
                               physical_stats=stats or {})
        bundle = self.numeric.extract(bundle)
        bundle = self.visual.extract(bundle)
        bundle = self.semantic.extract(bundle, labels=self.labels)

        if getattr(self.fusion, "mode", "") == "decision":
            pred = self._predict(api, element, image_path, "")
            base = dict(pred.get("prediction") or pred)
            fc = dict(self.cfg.get("fusion") or {})
            tau = float(fc.get("tau", 0.8))
            src = fc.get("source", "semantic")
            scores = bundle.semantic_scores if src == "semantic" else bundle.visual_scores
            if scores:
                top = max(scores, key=scores.get)
                if scores[top] >= tau:
                    base["base_severity"] = base.get("severity")
                    base["severity"] = top
                    base["severity_source"] = "probe"
                    base["probe_conf"] = round(float(scores[top]), 3)
                else:
                    base["severity_source"] = "qwen"
                    base["probe_conf"] = round(float(scores[top]), 3)
            return {"fusion": "", "meta": bundle.meta, "prediction": base}

        fused = self.fusion.compose(bundle)
        img = _composite(fused.images)
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

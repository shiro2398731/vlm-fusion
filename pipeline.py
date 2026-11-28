# -*- coding: utf-8 -*-
"""元素级多模态管线：物理 + 视觉 + 语义 → 融合 → Qwen 判读。"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

from PIL import Image

from features import extractors as FE
from features.base import FeatureBundle
from fusion import layers as FL

PROMPT_DIR = os.environ.get("VLM_LF_DATA", r"C:\vlmdisaster\LLaMA-Factory\data")


def load_prompt(element: str) -> str:
    """从元素训练集文件里取判读提示词（去掉 <image> 标记）。"""
    path = os.path.join(PROMPT_DIR, "train_%s_json.json" % element)
    if os.path.exists(path):
        try:
            import json
            rows = json.load(open(path, encoding="utf-8"))
            txt = rows[0]["conversations"][0]["value"]
            return txt.replace("<image>", "").strip()
        except Exception:
            pass
    return ("Classify this remote-sensing disaster map for element %s. "
            "Output a JSON object with disaster_type, detected, source, "
            "physical_quantity, severity, numeric fields and criterion." % element)


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
    @classmethod
    def from_yaml(cls, path: str, element: str) -> "ElementPipeline":
        """从 YAML 配置构造：文件里按元素分节。"""
        import yaml
        cfg_all = yaml.safe_load(open(path, encoding="utf-8")) or {}
        if element not in cfg_all:
            raise KeyError("element not in config: %s" % element)
        cfg = dict(cfg_all[element])
        cfg["element"] = element
        return cls(cfg)

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
        # 判读 LLM：local（本机 serve）| api（OpenAI 兼容视觉接口）
        self.llm_cfg = dict(cfg.get("llm") or {})
        self.llm = None

    def _predict(self, api: str, element: str, image: str, extra_text: str):
        backend = self.llm_cfg.get("backend", "local")
        if backend == "api":
            if self.llm is None:
                from llmeval import LLMClient
                self.llm = LLMClient(backend="api",
                                     base_url=self.llm_cfg.get("base_url", ""),
                                     api_key=self.llm_cfg.get("api_key", ""),
                                     model=self.llm_cfg.get("model", ""),
                                     timeout=self.llm_cfg.get("timeout", 120))
            prompt = load_prompt(element)
            if extra_text:
                prompt = prompt + "\n\nAdditional features:\n" + extra_text
            txt = self.llm.chat_image(prompt, image, max_tokens=self.llm_cfg.get("max_tokens", 512))
            try:
                s = txt[txt.find("{"):txt.rfind("}") + 1]
                return {"prediction": json.loads(s), "raw": txt}
            except Exception:
                return {"prediction": {}, "raw": txt}
        return self._predict_local(api, element, image, extra_text)

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

    def _predict_local(self, api, element, image, extra_text):
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

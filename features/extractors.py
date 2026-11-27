# -*- coding: utf-8 -*-
"""提取器实现集合。

视觉：none / resnet50 / dinov2_*
语义：none / clip_vit_b32 / clip_vit_l14

未安装依赖时降级为空特征，并在 meta 里标注原因，管线仍可跑 baseline。
"""
from __future__ import annotations

import os
import numpy as np
from typing import List, Optional
from PIL import Image

from .base import FeatureBundle, FeatureExtractor
from .numeric import NumericExtractor


def _to_rgb(path):
    return Image.open(path).convert("RGB")


class NoneExtractor(FeatureExtractor):
    name = "none"

    def extract(self, bundle, labels=None):
        bundle.meta["%s_status" % self.name] = "skipped"
        return bundle


class ResNetExtractor(FeatureExtractor):
    """冻结 ResNet50：把 layer4 特征图 PCA 到 3 通道。"""

    name = "resnet50"

    def __init__(self, output="map", device=None, **kw):
        super().__init__(**kw)
        self.output = output
        self.device = device
        self.model = None

    def load(self):
        import torch
        import torchvision.models as tvm
        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval().to(self.device)
        self._feat = {}
        self.model.layer4.register_forward_hook(
            lambda m, i, o: self._feat.__setitem__("map", o.detach()))

    def extract(self, bundle, labels=None):
        if self.model is None:
            self.load()
        torch = self.torch
        tf = __import__("torchvision.transforms", fromlist=["Compose"]).Compose([
            __import__("torchvision.transforms", fromlist=["Resize"]).Resize((224, 224)),
            __import__("torchvision.transforms", fromlist=["ToTensor"]).ToTensor(),
            __import__("torchvision.transforms", fromlist=["Normalize"]).Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = tf(_to_rgb(bundle.image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model(x)
        f = self._feat["map"][0].cpu().numpy()
        if self.output == "vector":
            bundle.visual_vector = f.mean(axis=(1, 2))
            bundle.meta["visual"] = "resnet50:vector"
            return bundle
        f = f.reshape(f.shape[0], -1).T
        f = f - f.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(f, full_matrices=False)
        pc = u[:, :3] * s[:3]
        pc = (pc - pc.min(axis=0)) / (pc.max(axis=0) - pc.min(axis=0) + 1e-6)
        h = int(np.sqrt(f.shape[0]))
        m = (pc.reshape(h, h, 3) * 255).astype("uint8")
        bundle.visual_map = np.array(Image.fromarray(m).resize((384, 384), Image.BILINEAR))
        bundle.meta["visual"] = "resnet50:map"
        return bundle


class DinoV2Extractor(FeatureExtractor):
    """DINOv2（占位）：依赖 timm / torch.hub，后续填入。"""

    name = "dinov2_small"

    def extract(self, bundle, labels=None):
        bundle.meta["visual_status"] = "dinov2_not_implemented"
        return bundle


class ClipExtractor(FeatureExtractor):
    """冻结 CLIP：图像嵌入与类别文本相似度。"""

    name = "clip_vit_b32"

    def __init__(self, model_name=os.environ.get("VLM_CLIP_DIR", r"C:\vlmdisaster\models\clip-vit-b32"),
                 prompt_template="a satellite remote-sensing disaster map showing {label}",
                 cache_dir=None, device=None, **kw):
        super().__init__(**kw)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.cache_dir = cache_dir or os.environ.get("VLM_HF_HOME", r"C:\vlmdisaster\models\hf")
        self.device = device
        self.model = None
        self.proc = None

    def load(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.proc = CLIPProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model.eval().to(self.device)

    def extract(self, bundle, labels=None):
        labels = list(labels or [])
        if not labels:
            bundle.meta["semantic_status"] = "no_labels"
            return bundle
        if self.model is None:
            self.load()
        torch = self.torch
        img = _to_rgb(bundle.image_path)
        prompts = [self.prompt_template.format(label=l) for l in labels]
        inp = self.proc(text=prompts, images=img, return_tensors="pt", padding=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad():
            out = self.model(**inp)
        probs = out.logits_per_image[0].softmax(dim=0)
        bundle.semantic_scores = {str(l): float(p) for l, p in zip(labels, probs)}
        bundle.semantic_vector = out.image_embeds[0].detach().cpu().numpy()
        bundle.meta["semantic"] = {"model": self.model_name, "labels": labels}
        return bundle


class ClipProbeExtractor(ClipExtractor):
    """CLIP 嵌入 + 线性探针：把 CLIP 当特征提取器，分类头在本地训练。

    这是语义支路的正式形态：零样本文本相似度对渲染图基本无效
    （实测珊瑚 0%、积雪 31%），但嵌入+探针可达 90%+。
    """

    name = "clip_probe"

    def __init__(self, probe_path=None, element=None, **kw):
        super().__init__(**kw)
        self.probe_path = probe_path
        self.element = element
        self.clf = None
        self.labels_probe = None

    def load(self):
        super().load()
        import joblib
        if not self.probe_path or not os.path.exists(self.probe_path):
            raise FileNotFoundError("probe not found: %s" % self.probe_path)
        z = joblib.load(self.probe_path)
        self.clf = z["model"]
        self.labels_probe = list(z["labels"])

    def extract(self, bundle, labels=None):
        if self.clf is None:
            self.load()
        img = _to_rgb(bundle.image_path)
        inp = self.proc(images=img, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with self.torch.no_grad():
            out = self.model.get_image_features(pixel_values=inp["pixel_values"])
        vec = out.pooler_output[0].detach().cpu().numpy()
        probs = self.clf.predict_proba(vec.reshape(1, -1))[0]
        bundle.semantic_vector = vec
        bundle.semantic_scores = {str(l): float(p) for l, p in zip(self.labels_probe, probs)}
        bundle.meta["semantic"] = {"model": "clip_probe", "probe": self.probe_path,
                                   "labels": self.labels_probe}
        return bundle



REGISTRY = {
    "none": NoneExtractor,
    "numeric": NumericExtractor,
    "resnet50": ResNetExtractor,
    "dinov2_small": DinoV2Extractor,
    "clip_vit_b32": ClipExtractor,
    "clip_probe": ClipProbeExtractor,
}


def build(name: str, **cfg) -> FeatureExtractor:
    if name not in REGISTRY:
        raise KeyError("unknown extractor: %s" % name)
    return REGISTRY[name](**cfg)

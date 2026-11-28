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


class ResNetProbeExtractor(ResNetExtractor):
    """ResNet50 嵌入 + 线性探针：视觉支路的概率输出（不注入特征图）。"""

    name = "resnet_probe"

    def __init__(self, probe_path=None, output=None, **kw):
        super().__init__(output="vector", **kw)
        self.probe_path = probe_path
        self.clf = None
        self.labels_probe = None

    def load(self):
        super().load()
        import joblib
        if not self.probe_path or not os.path.exists(self.probe_path):
            raise FileNotFoundError("resnet probe not found: %s" % self.probe_path)
        z = joblib.load(self.probe_path)
        self.clf = z["model"]
        self.labels_probe = list(z["labels"])
        self.pca = z.get("pca")

    def extract(self, bundle, labels=None):
        if self.clf is None:
            self.load()
        bundle = super().extract(bundle)
        probs = self.clf.predict_proba(bundle.visual_vector.reshape(1, -1))[0]
        bundle.visual_scores = {str(l): float(p) for l, p in zip(self.labels_probe, probs)}
        if getattr(self, "pca", None) is not None:
            bundle.visual_digest = [round(float(x), 4) for x in
                                    self.pca.transform(bundle.visual_vector.reshape(1, -1))[0]]
        bundle.meta["visual"] = {"model": "resnet_probe", "probe": self.probe_path}
        return bundle


class ClipAPIExtractor(FeatureExtractor):
    """语义支路的 API 形态：远程视觉语言模型返回嵌入或类别分数。

    契约（POST base_url）：
      {"model": ..., "image_b64": ...}
    响应二选一：
      {"embedding": [..]}  -> semantic_vector
      {"scores": {label: prob}} -> semantic_scores
    """

    name = "clip_api"

    def __init__(self, base_url=None, api_key=None, model="clip", timeout=60, **kw):
        super().__init__(**kw)
        self.base_url = base_url or os.environ.get("SEMANTIC_API_URL", "")
        self.api_key = api_key or os.environ.get("SEMANTIC_API_KEY", "")
        self.model = model
        self.timeout = timeout

    def extract(self, bundle, labels=None):
        import base64
        import requests
        if not self.base_url:
            bundle.meta["semantic_status"] = "clip_api: no base_url"
            return bundle
        b64 = base64.b64encode(open(bundle.image_path, "rb").read()).decode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        r = requests.post(self.base_url,
                          headers=headers,
                          json={"model": self.model, "image_b64": b64,
                                "labels": list(labels or [])},
                          timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if "scores" in data:
            bundle.semantic_scores = {str(k): float(v) for k, v in data["scores"].items()}
        elif "embedding" in data:
            bundle.semantic_vector = np.asarray(data["embedding"], dtype="float32")
        else:
            bundle.meta["semantic_status"] = "clip_api: unknown response keys"
        bundle.meta["semantic"] = {"model": self.model, "backend": "api",
                                   "endpoint": self.base_url}
        return bundle


class DinoV2Extractor(FeatureExtractor):
    """冻结 DINOv2（timm 本地权重）：CLS 嵌入。"""

    name = "dinov2_small"
    REPO = "vit_small_patch14_dinov2.lvd142m"

    def __init__(self, model_dir=None, device=None, **kw):
        super().__init__(**kw)
        self.model_dir = model_dir or os.environ.get(
            "VLM_DINO_DIR", r"C:\vlmdisaster\models\dinov2-small")
        self.device = device
        self.model = None

    def load(self):
        import timm
        import torch
        from safetensors.torch import load_file
        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = timm.create_model(self.REPO, pretrained=False, num_classes=0)
        sd = load_file(os.path.join(self.model_dir, "model.safetensors"))
        m.load_state_dict(sd, strict=False)
        self.model = m.eval().to(self.device)
        from torchvision import transforms as T
        self.tf = T.Compose([T.Resize((518, 518)), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def embed(self, image_path):
        if self.model is None:
            self.load()
        x = self.tf(_to_rgb(image_path)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            f = self.model.forward_features(x)  # (1,1+N,384)
        return f[:, 0].squeeze(0).cpu().numpy()

    def extract(self, bundle, labels=None):
        bundle.visual_vector = self.embed(bundle.image_path)
        bundle.meta["visual"] = {"model": "dinov2_small"}
        return bundle


class DinoProbeExtractor(DinoV2Extractor):
    """DINOv2 嵌入 + 线性探针，输出类别概率。"""

    name = "dinov2_probe"

    def __init__(self, probe_path=None, **kw):
        super().__init__(**kw)
        self.probe_path = probe_path
        self.clf = None
        self.labels_probe = None

    def load(self):
        super().load()
        import joblib
        if not self.probe_path or not os.path.exists(self.probe_path):
            raise FileNotFoundError("dinov2 probe not found: %s" % self.probe_path)
        z = joblib.load(self.probe_path)
        self.clf = z["model"]
        self.labels_probe = list(z["labels"])
        self.pca = z.get("pca")

    def extract(self, bundle, labels=None):
        if self.clf is None:
            self.load()
        v = self.embed(bundle.image_path)
        bundle.visual_vector = v
        probs = self.clf.predict_proba(v.reshape(1, -1))[0]
        bundle.visual_scores = {str(l): float(p) for l, p in zip(self.labels_probe, probs)}
        if getattr(self, "pca", None) is not None:
            bundle.visual_digest = [round(float(x), 4) for x in
                                    self.pca.transform(v.reshape(1, -1))[0]]
        bundle.meta["visual"] = {"model": "dinov2_probe", "probe": self.probe_path}
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
        self.pca = z.get("pca")

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
        if getattr(self, "pca", None) is not None:
            bundle.semantic_digest = [round(float(x), 4) for x in
                                      self.pca.transform(vec.reshape(1, -1))[0]]
        bundle.meta["semantic"] = {"model": "clip_probe", "probe": self.probe_path,
                                   "labels": self.labels_probe}
        return bundle



REGISTRY = {
    "none": NoneExtractor,
    "numeric": NumericExtractor,
    "resnet50": ResNetExtractor,
    "resnet_probe": ResNetProbeExtractor,
    "dinov2_small": DinoV2Extractor,
    "dinov2_probe": DinoProbeExtractor,
    "clip_vit_b32": ClipExtractor,
    "clip_probe": ClipProbeExtractor,
    "clip_api": ClipAPIExtractor,
}


def build(name: str, **cfg) -> FeatureExtractor:
    if name not in REGISTRY:
        raise KeyError("unknown extractor: %s" % name)
    return REGISTRY[name](**cfg)

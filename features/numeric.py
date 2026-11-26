# -*- coding: utf-8 -*-
"""数值支路：把渲染器给出的窗口物理量标准化成向量与文本。

数值来源是渲染前计算的物理统计量（mean/p90/max 等），不是从图里猜的，
因此这条支路本身不带估计误差，是三条支路里唯一“客观”的一条。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .base import FeatureBundle, FeatureExtractor


class NumericExtractor(FeatureExtractor):
    name = "numeric"

    def __init__(self, scale: Optional[Dict[str, List[float]]] = None,
                 fields: Optional[List[str]] = None, **kw):
        super().__init__(**kw)
        # scale: {"mean_fsc": [0.0, 1.0]} 用于归一化到 0-1
        self.scale = scale or {}
        self.fields = fields

    def extract(self, bundle: FeatureBundle, labels=None) -> FeatureBundle:
        stats = bundle.physical_stats or {}
        keys = self.fields or sorted(stats)
        vec = []
        for k in keys:
            v = stats.get(k)
            if v is None:
                vec.append(0.0); continue
            lo, hi = self.scale.get(k, [None, None])
            if lo is not None and hi is not None and hi > lo:
                v = (float(v) - lo) / (hi - lo)
            vec.append(float(v))
        bundle.numeric_vector = np.asarray(vec, dtype="float32")
        bundle.meta["numeric"] = {"fields": keys, "values": [float(x) for x in vec]}
        return bundle

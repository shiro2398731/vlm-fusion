# -*- coding: utf-8 -*-
"""三路特征的数据契约与提取器接口。

物理特征来自现有渲染器；视觉/语义特征由本包新增的深度网络提取。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FeatureBundle:
    """一个元素、一张图、一次判读所需的全部特征。"""

    element: str
    image_path: str
    physical_stats: Dict[str, float] = field(default_factory=dict)
    visual_map: Optional[np.ndarray] = None      # HxWx3 uint8
    visual_vector: Optional[np.ndarray] = None    # 1D
    semantic_scores: Optional[Dict[str, float]] = None
    semantic_vector: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class FeatureExtractor(ABC):
    """视觉或语义提取器的统一接口。"""

    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = kwargs

    def load(self) -> None:
        """加载权重；默认无需加载。"""

    @abstractmethod
    def extract(self, bundle: FeatureBundle, labels: Optional[List[str]] = None) -> FeatureBundle:
        """原地补充特征并返回 bundle。"""

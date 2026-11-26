# -*- coding: utf-8 -*-
"""融合层：把三路特征变成 Qwen 的输入（图像列表 + 追加文本）。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FusionInput:
    images: List[str] = field(default_factory=list)
    text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class Fusion(ABC):
    mode: str = "base"

    def __init__(self, **cfg: Any) -> None:
        self.cfg = cfg

    @abstractmethod
    def compose(self, bundle) -> FusionInput: ...

# -*- coding: utf-8 -*-
"""采用元素注册表：统一读根目录的 elements_capability.json。"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = os.path.join(ROOT, "elements_capability.json")
CAP_OLD = os.path.join(ROOT, "处理方案", "元素能力_20261125.json")


def _cap_path():
    return CAP if os.path.exists(CAP) else (CAP_OLD if os.path.exists(CAP_OLD) else CAP)


def adopted():
    """返回 38 个采用元素 id（排序）。"""
    if not os.path.exists(_cap_path()):
        return []
    try:
        return sorted(json.load(open(_cap_path(), encoding="utf-8")).keys())
    except Exception:
        return []


def meta():
    """返回 {元素: (中文名, 源)}。"""
    out = {}
    if not os.path.exists(_cap_path()):
        return out
    try:
        for k, v in json.load(open(_cap_path(), encoding="utf-8")).items():
            zh = str(v.get("zh", k)).split("（")[0]
            out[k] = (zh, v.get("source", ""))
    except Exception:
        pass
    return out

# -*- coding: utf-8 -*-
r"""
定位：影像坐标 → 经纬度 → 地名

## 为什么定位不靠模型

26 个元素里只有 3 个的训练数据带地理字段（wave 有经纬度范围、sea_ice 有 tile），
其余元素渲染时没把地理变换存下来，模型无法从像素反推坐标。

所以定位的坐标只能来自**输入端**：卫星 API 返回的影像自带 bbox，
或者使用者手动给出经纬度。模型负责识别与定级，坐标由数据源提供。

## 逆地理编码

复用 src/inference/geo_coder.py（Nominatim 免费 / 高德需 key）。
Nominatim 在国内可能连不上，失败时返回经纬度字符串而不是抛异常。
"""

import os
import sys
from typing import Dict, Optional

ROOT = r"C:\vlmdisaster"
sys.path.insert(0, os.path.join(ROOT, "src"))

from inference.geo_coder import GeoCoder  # noqa: E402


def bbox_center(bbox) -> Optional[tuple]:
    """bbox = (min_lon, min_lat, max_lon, max_lat) 或 'minx,miny,maxx,maxy'"""
    if not bbox:
        return None
    if isinstance(bbox, str):
        try:
            bbox = [float(x) for x in bbox.replace(" ", "").split(",")]
        except ValueError:
            return None
    if len(bbox) != 4:
        return None
    min_lon, min_lat, max_lon, max_lat = [float(x) for x in bbox]
    return (min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0


def resolve(bbox="", lat=None, lon=None) -> Dict:
    """
    返回 {"lat":..., "lon":..., "how":..., "ok":bool}

    优先手动经纬度，其次 bbox 中心点。
    """
    if lat is not None and lon is not None:
        return {"lat": float(lat), "lon": float(lon), "how": "手动给定", "ok": True}
    c = bbox_center(bbox)
    if c:
        return {"lat": c[0], "lon": c[1], "how": "bbox 中心", "ok": True}
    return {"lat": None, "lon": None, "how": "未提供坐标", "ok": False}


def describe(lat, lon, provider="nominatim", amap_key="") -> str:
    """经纬度 → 地名。失败时返回经纬度原文，不抛异常。"""
    if lat is None or lon is None:
        return "位置未知"
    try:
        return GeoCoder(provider=provider, amap_key=amap_key).reverse(lat, lon)
    except Exception as e:  # 网络不通 / 服务不可用
        return "纬度%.4f, 经度%.4f（逆地理编码失败：%s）" % (lat, lon, e)

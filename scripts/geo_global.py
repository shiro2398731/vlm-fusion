# -*- coding: utf-8 -*-
r"""
全球逆地理（国家 + 一级行政区），离线。

数据：
  data/geo/global/admin1_compact.geojson   各国省/州（Natural Earth 10m 简化版，约 4.7 MB）
  data/geo/global/ne_110m_admin_0_countries.geojson  国家边界（约 0.8 MB）

用法：
    from geo_global import lookup
    lookup(35.6762, 139.6503)   # 东京
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO = os.path.join(ROOT, "data", "geo", "global")
ADMIN1 = os.path.join(GEO, "admin1_compact.geojson")
ADMIN0 = os.path.join(GEO, "ne_110m_admin_0_countries.geojson")

_cache = {}


def _in_ring(x, y, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def _in_polys(x, y, polys):
    for poly in polys:
        if not poly:
            continue
        if not _in_ring(x, y, poly[0]):
            continue
        hit = True
        for hole in poly[1:]:
            if _in_ring(x, y, hole):
                hit = False
                break
        if hit:
            return True
    return False


def _load(path, key_fields):
    """返回 ([(bbox, polys, props)], 网格索引)。"""
    if not os.path.exists(path):
        return [], {}
    d = json.load(open(path, encoding="utf-8"))
    items = []
    for f in d.get("features", []):
        geom = f.get("geometry") or {}
        ty = geom.get("type")
        if ty == "Polygon":
            polys = [geom["coordinates"]]
        elif ty == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue
        xs, ys = [], []
        for poly in polys:
            for ring in poly:
                for p in ring:
                    xs.append(p[0]); ys.append(p[1])
        if not xs:
            continue
        props = {k: (f.get("properties") or {}).get(k, "") for k in key_fields}
        items.append(((min(xs), min(ys), max(xs), max(ys)), polys, props))
    grid = {}
    for idx, (bbox, _, _) in enumerate(items):
        x0, y0, x1, y1 = bbox
        for gx in range(int(x0 // 1), int(x1 // 1) + 1):
            for gy in range(int(y0 // 1), int(y1 // 1) + 1):
                grid.setdefault((gx, gy), []).append(idx)
    return items, grid


def _pick(lat, lon, items, grid):
    for idx in grid.get((int(lon // 1), int(lat // 1)), []):
        bbox, polys, props = items[idx]
        if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
            continue
        if _in_polys(lon, lat, polys):
            return props
    return None


def _nearest(lat, lon, items, max_deg=1.5):
    """最近行政区中心兜底（解决沿海/小行政区简化后点落空）。"""
    best, bestd = None, max_deg * max_deg
    for bbox, _polys, props in items:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        d = (cx - lon) ** 2 + (cy - lat) ** 2
        if d < bestd:
            best, bestd = props, d
    return best



PLACES = os.path.join(GEO, "places_compact.json")


def _load_places():
    if "places" not in _cache:
        p = PLACES
        _cache["places"] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
    return _cache["places"]


def nearest_place(lat, lon, max_km=150.0):
    """最近聚落（首都/城市），返回名称与距离（km）。"""
    import math
    best, bestd = None, None
    for r in _load_places():
        dlat = r["lat"] - lat
        dlon = (r["lon"] - lon) * math.cos(math.radians(lat))
        d2 = dlat * dlat + dlon * dlon
        if bestd is None or d2 < bestd:
            best, bestd = r, d2
    if not best:
        return None
    km = math.sqrt(bestd) * 111.32
    if km > max_km:
        return {"name": best["name"], "country": best["country"],
                "distance_km": round(km, 1), "far": True}
    return {"name": best["name"], "country": best["country"], "distance_km": round(km, 1)}


def lookup(lat, lon):
    if "a1" not in _cache:
        _cache["a1"], _cache["a1g"] = _load(ADMIN1, ("name", "admin", "iso2"))
        _cache["a0"], _cache["a0g"] = _load(ADMIN0, ("NAME", "ADM0_A3", "ISO_A2"))
    base = {"lat": lat, "lon": lon, "nearest_place": nearest_place(lat, lon)}
    a1 = _pick(lat, lon, _cache["a1"], _cache["a1g"])
    if a1:
        return dict(base, country=a1.get("admin") or "", province=a1.get("name") or "",
                    city="", iso2=a1.get("iso2") or "")
    a0 = _pick(lat, lon, _cache["a0"], _cache["a0g"])
    if a0:
        # 国家命中但行政区没中：用最近行政区中心兜底（限定 1.5°）
        near = _nearest(lat, lon, _cache["a1"])
        if near and (near.get("admin") or "") == (a0.get("NAME") or ""):
            return dict(base, country=near.get("admin") or "", province=near.get("name") or "",
                        city="", iso2=near.get("iso2") or "", approx=True)
        return dict(base, country=a0.get("NAME") or "", province="", city="",
                    iso2=a0.get("ISO_A2") or "")
    return dict(base, country="", province="", city="", iso2="")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for name, la, lo in [("东京", 35.6762, 139.6503), ("纽约", 40.7128, -74.0060),
                         ("巴黎", 48.8566, 2.3522), ("日内瓦", 46.2044, 6.1432),
                         ("圣保罗", -23.5505, -46.6333), ("悉尼", -33.8688, 151.2093),
                         ("开普敦", -33.9249, 18.4241), ("太平洋", 0.0, -140.0)]:
        print(name, lookup(la, lo))

# -*- coding: utf-8 -*-
r"""
逆地理编码（离线，全球）

中国：省 → 市 → 区县（DataV 边界）
国外：国家 → 一级行政区（Natural Earth），并给最近聚落与距离
任何位置都返回原始经纬度；公海/荒野返回坐标 + 最近聚落。

用法：
    from reverse_geocode import geocode
    geocode(39.9042, 116.4074)   # 中国：北京市 / 东城区
    geocode(35.6762, 139.6503)   # 日本：Japan / Tokyo / 最近聚落 9 km
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROV = os.path.join(ROOT, "data", "geo", "china_provinces.json")
CITY = os.path.join(ROOT, "data", "geo", "china_cities.json")
COUNTY = os.path.join(ROOT, "data", "geo", "china_counties_compact.json")

_cache = {}


def _in_ring(x, y, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def _in_poly(x, y, poly):
    if not _in_ring(x, y, poly[0]):
        return False
    for hole in poly[1:]:
        if _in_ring(x, y, hole):
            return False
    return True


def _load(path, key, extra=None):
    if key in _cache:
        return _cache[key]
    if not os.path.exists(path):
        _cache[key] = None
        return None
    j = json.load(open(path, encoding="utf-8"))
    feats = []
    for f in j.get("features", []):
        geom = f.get("geometry") or {}
        props = f.get("properties") or {}
        name = props.get("name", "")
        parent = props.get("parent", "") if extra else ""
        ty = geom.get("type")
        if ty == "Polygon":
            polys = [geom["coordinates"]]
        elif ty == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue
        for poly in polys:
            ring = poly[0]
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            feats.append({"name": name, "parent": parent,
                          "bbox": (min(xs), min(ys), max(xs), max(ys)), "poly": poly})
    _cache[key] = feats
    return feats


def _find(lat, lon, feats, parent=None):
    if not feats:
        return None
    for f in feats:
        if parent is not None and f.get("parent") != parent:
            continue
        x0, y0, x1, y1 = f["bbox"]
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            continue
        if _in_poly(lon, lat, f["poly"]):
            return f["name"]
    return None


def _display(out):
    parts = []
    for k in ("country", "province", "city", "county"):
        v = out.get(k)
        if v and v not in parts:
            parts.append(v)
    s = " / ".join(parts) if parts else "未知区域"
    npl = out.get("nearest_place") or {}
    if npl.get("name"):
        s += "（最近聚落 %s %.1f km%s）" % (npl["name"], npl.get("distance_km", 0),
                                         "，偏远" if npl.get("far") else "")
    s += " @ %.4f, %.4f" % (out.get("lat", 0), out.get("lon", 0))
    return s


def geocode(lat, lon):
    lat, lon = float(lat), float(lon)
    out = {"lat": lat, "lon": lon, "country": None, "province": None,
           "city": None, "county": None}

    # 中国：省市县
    if 3.0 <= lat <= 54.0 and 73.0 <= lon <= 136.0:
        out["country"] = "中国"
        out["province"] = _find(lat, lon, _load(PROV, "prov"))
        out["city"] = _find(lat, lon, _load(CITY, "city"))
        if out["city"]:
            out["county"] = _find(lat, lon, _load(COUNTY, "county", extra=True),
                                  parent=out["city"])
        # 直辖市：省=市，区县在最细一级
        if out["province"] in ("北京市", "上海市", "天津市", "重庆市") and out["city"]:
            out["county"], out["city"] = out["city"], out["province"]
        if out["province"]:
            out["nearest_place"] = None
            out["display"] = _display(out)
            return out

    # 全球：国家 + 一级行政区 + 最近聚落
    try:
        from geo_global import lookup
        g = lookup(lat, lon)
        out["country"] = g.get("country") or out["country"]
        out["province"] = g.get("province") or out["province"]
        if g.get("iso2"):
            out["iso2"] = g["iso2"]
        if g.get("approx"):
            out["approx"] = True
        if g.get("nearest_place"):
            out["nearest_place"] = g["nearest_place"]
    except Exception:
        pass
    out["display"] = _display(out)
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for n, la, lo in [("北京", 39.9042, 116.4074), ("武汉", 30.5928, 114.3055),
                      ("东京", 35.6762, 139.6503), ("巴黎", 48.8566, 2.3522),
                      ("圣保罗", -23.5505, -46.6333), ("撒哈拉", 23.4, 12.5),
                      ("太平洋", 0.0, -140.0)]:
        print(n, geocode(la, lo))

# -*- coding: utf-8 -*-
"""按边界窗口径重算原始链路各元素的一致率。"""
import glob, json, os, sys

ROOT = r"C:\vlmdisaster"
REL = float(os.environ.get("REL", "0.10"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# element: (数值字段列表, 阈值列表, 比较字段)
SPEC = {
    "snow_swe": (["truth_mean_swe"], [15, 30, 60], "severity"),
    "waterlogging_pct": (["truth_pct"], [80, 90, 97], "severity"),
    "hurricane_rain": (["truth_max_mm"], [50, 100, 250], "severity"),
    "bloom_chla": (["truth_p90"], [1, 5, 20], "severity"),
    "sea_ice_amsr2": (["truth_ice_frac"], [15, 50, 85], "severity"),
    "gpm_nt": (["truth_mean_mm"], [0.5, 5], "severity"),
    "wave": (["truth_max_hs"], [2.5, 3.5, 4.5, 6], "severity"),
    "flood_precip_global": (["truth_cum3_max"], [100], "severity"),
    "drought_ndvi_clim": (["truth_pct"], [10], "detected"),
    "geomagnetic": (["truth_min_dst"], [-50, -100, -200, -350], "severity"),
    "haze": (["truth_mean_aod", "truth_max_aod"], [0.2, 0.6, 1.5], "severity"),
    "sanddust": (["truth_mean_aod", "truth_max_aod"], [0.22, 1.4], "severity"),
    "uvai_dust": (["truth_max_uvai"], [1, 2, 4], "severity"),
    "wildfire_smoke": (["truth_max_uvai"], [1, 2, 4], "severity"),
    "coral_dhw": (["truth_max_dhw"], [1, 4, 8], "severity"),
    "volcano_so2": (["truth_max_so2"], [1, 5, 20], "severity"),
    "enso": (["truth_nino34_anom"], [-0.5, 0.5], "severity"),
    "iod": (["truth_dmi"], [-0.4, 0.4], "severity"),
    "marine_heatwave": (["truth_area_frac"], [20], "severity"),
    "snowstorm": (["truth_snow_cm"], [10, 20, 30], "severity"),
    "flare": (["truth_flux"], [1e-6], "detected_truth"),
    "wind_gale": (["truth_max_ms"], [17.2, 24.5], "severity"),
    "drought_grace": (["truth_pct"], [10, 50], "none"),
    "groundwater_grace": (["truth_pct"], [10, 50], "none"),
}

def near_boundary(vals, ths, rel=None):
    rel = REL if rel is None else rel
    for v in vals:
        if v is None or not isinstance(v, (int, float)):
            continue
        for t in ths:
            if t == 0:
                continue
            if abs(v - t) <= rel * abs(t):
                return True
    return False

def consistent(row, mode):
    if mode == "detected_truth":
        return bool(row.get("model_detected")) == bool(row.get("truth_detected"))
    if mode == "detected":
        return bool(row.get("model_detected")) == (row.get("truth_label") not in ("none",))
    return row.get("model_severity") == row.get("truth_label")

rows_by_el = {}
for p in glob.glob(os.path.join(ROOT, "outputs", "raw_pair*", "pair_result.json")):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    for r in d:
        rows_by_el.setdefault(r.get("element"), []).append(r)

print("%-22s %6s %6s %6s %s" % ("element", "all", "pass", "bnd", "verdict"))
out = []
for el, rows in sorted(rows_by_el.items()):
    if el not in SPEC:
        total = len(rows)
        ok = sum(1 for r in rows if r.get("match", r.get("model_severity") == r.get("truth_label")))
        out.append({"element": el, "rows": total, "pass": ok, "boundary": 0, "verdict": "no-spec"})
        print("%-22s %6d %6d %6d %s" % (el, total, ok, 0, "no-spec"))
        continue
    fields, ths, mode = SPEC[el]
    nonb = [r for r in rows if not near_boundary([r.get(f) for f in fields], ths)]
    bnd = len(rows) - len(nonb)
    ok = sum(1 for r in nonb if consistent(r, mode))
    if len(nonb) < 2:
        verdict = "证据不足"
    elif ok == len(nonb):
        verdict = "通过"
    else:
        verdict = "未通过"
    out.append({"element": el, "rows": len(rows), "pass_nonboundary": ok,
                "nonboundary": len(nonb), "boundary": bnd, "verdict": verdict})
    print("%-22s %6d %6d %6d %s" % (el, len(rows), ok, bnd, verdict))

json.dump(out, open(os.path.join(ROOT, "处理方案", "边界窗重算.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved 边界窗重算.json")

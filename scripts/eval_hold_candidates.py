# -*- coding: utf-8 -*-
"""对未启用元素在 hold 集上测试“候选融合”，看是否漏开。"""
import glob, json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
PROBES = r"C:\vlm-fusion\outputs\probes"
D = r"C:\vlm-fusion\outputs\four_groups"
ELS = ["bloom_chla", "enso", "flood_opt", "green_tide", "ground_subsidence",
       "iod", "marine_heatwave", "snow_fsc", "snowstorm"]
OUT = r"C:\vlm-fusion\docs\hold_candidates.json"

def best_source(el):
    best = ("semantic", -1)
    for src in ("semantic", "dinov2"):
        f = os.path.join(D, "%s_decision_%s.json" % (el, src))
        if os.path.exists(f):
            j = json.load(open(f, encoding="utf-8"))
            if j.get("sev_ok", -1) > best[1]:
                best = (src, j["sev_ok"])
    return best[0]

res = {}
for el in ELS:
    p = os.path.join(VAL, "train_%s_hold_json_val.json" % el)
    if not os.path.exists(p):
        continue
    recs = [r for r in json.load(open(p, encoding="utf-8"))
            if os.path.exists(r.get("image") or (r.get("images") or [""])[0])]
    by = {}
    for r in recs:
        by.setdefault(r.get("label"), []).append(r)
    picked = [r for lab in sorted(by) for r in by[lab][:2]]
    src = best_source(el)
    probe = os.path.join(PROBES, "%s_clip_probe.joblib" % el if src == "semantic"
                         else "%s_dinov2_probe.joblib" % el)
    cfg = {"element": el, "numeric": {"extractor": "none"},
           "visual": ({"extractor": "dinov2_probe", "probe_path": probe}
                      if src == "dinov2" else {"extractor": "none"}),
           "semantic": ({"extractor": "clip_probe", "probe_path": probe}
                        if src == "semantic" else {"extractor": "none"}),
           "fusion": {"mode": ["decision"], "tau": 0.6, "source":
                      ("visual" if src == "dinov2" else "semantic")}}
    base = ElementPipeline({"element": el, "numeric": {"extractor": "none"},
                            "visual": {"extractor": "none"},
                            "semantic": {"extractor": "none"},
                            "fusion": {"mode": ["text", "image"]}})
    fused = ElementPipeline(cfg)
    okb = okf = 0
    for r in picked:
        img = r.get("image") or r["images"][0]
        ob = base.run(img, API, el, {})
        of = fused.run(img, API, el, {})
        vb = (ob["prediction"].get("prediction") or ob["prediction"]).get("severity")
        vf = (of["prediction"].get("prediction") or of["prediction"]).get("severity")
        okb += vb == r.get("label"); okf += vf == r.get("label")
    res[el] = {"n": len(picked), "baseline": okb, "candidate": okf, "source": src}
    print("%-22s %-8s baseline %d/%d  candidate %d/%d" % (
        el, src, okb, len(picked), okf, len(picked)), flush=True)

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved")

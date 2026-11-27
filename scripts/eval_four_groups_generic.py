# -*- coding: utf-8 -*-
"""四组对照：snow_swe（语义用 CLIP 探针）。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
EL = "snow_swe"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data\train_snow_swe_json_val.json"
PROBE = r"C:\vlm-fusion\outputs\probes\snow_swe_clip_probe.joblib"
OUT = r"C:\vlm-fusion\outputs\four_groups"
os.makedirs(OUT, exist_ok=True)

recs = [r for r in json.load(open(VAL, encoding="utf-8")) if os.path.exists(r.get("image", ""))]
by = {}
for r in recs:
    by.setdefault(r.get("label"), []).append(r)
picked = []
for lab in ("none", "light", "moderate", "heavy"):
    picked += by.get(lab, [])[:2]
print("picked", len(picked), {k: len(v) for k, v in by.items()}, flush=True)

def cfg(visual=False, semantic=False):
    return {
        "element": EL,
        "numeric": {"extractor": "none"},
        "visual": ({"extractor": "resnet_probe",
                    "probe_path": r"C:\vlm-fusion\outputs\probes\snow_swe_resnet_probe.joblib"}
                   if visual else {"extractor": "none"}),
        "semantic": ({"extractor": "clip_probe", "probe_path": PROBE}
                     if semantic else {"extractor": "none"}),
        "fusion": {"mode": ["text", "image"], "include_numeric": False,
                   "include_visual": visual, "include_semantic": semantic},
    }

GROUPS = [("baseline", False, False), ("+visual", True, False),
          ("+semantic", False, True), ("+both", True, True)]
results = {}
for name, vis, sem in GROUPS:
    p = ElementPipeline(cfg(vis, sem))
    ok = 0; n = 0
    for r in picked:
        try:
            out = p.run(r["image"], API, EL, {})
        except Exception as e:
            print(name, "ERR", str(e)[:60], flush=True); continue
        pred = out["prediction"].get("prediction") or out["prediction"]
        n += 1
        if pred.get("severity") == r.get("label"):
            ok += 1
        print("%-9s %-30s truth %-8s pred %-8s" % (name, os.path.basename(r["image"]),
                                                   r.get("label"), pred.get("severity")), flush=True)
    results[name] = {"n": n, "sev_ok": ok, "sev_acc": round(100.0 * ok / max(n, 1), 1)}
    print("%-9s sev %d/%d" % (name, ok, n), flush=True)

json.dump(results, open(os.path.join(OUT, "%s_four_groups.json" % EL), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved")

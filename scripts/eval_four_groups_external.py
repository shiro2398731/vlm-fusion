# -*- coding: utf-8 -*-
"""四组对照：使用训练集外新日期（snow_fsc）。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
REC = r"C:\vlmdisaster\outputs\quality\snow_fsc_new\records.json"
OUT = r"C:\vlm-fusion\outputs\four_groups"
os.makedirs(OUT, exist_ok=True)

IMG_DIR = r"C:\vlmdisaster\outputs\quality\snow_fsc_new\images"
recs = [r for r in json.load(open(REC, encoding="utf-8"))
        if r.get("product") == "intrinsic_fractional_snow_cover"]
for r in recs:
    if not os.path.isabs(r["image"]):
        r["image"] = os.path.join(IMG_DIR, os.path.basename(r["image"]))
print("external samples", len(recs), flush=True)

def cfg(visual=False, semantic=False):
    return {
        "numeric": {"extractor": "none"},
        "visual": {"extractor": "resnet50" if visual else "none", "output": "map"},
        "semantic": {"extractor": "clip_vit_b32" if semantic else "none",
                     "labels": ["no snow", "light snow cover", "moderate snow cover", "heavy snow cover"]},
        "fusion": {"mode": ["text", "image"], "include_numeric": False,
                   "include_visual": visual, "include_semantic": semantic},
    }

GROUPS = [("baseline", False, False), ("+visual", True, False),
          ("+semantic", False, True), ("+both", True, True)]
results = {}
for name, vis, sem in GROUPS:
    p = ElementPipeline(cfg(vis, sem))
    ok = 0; n = 0; errs = []
    for r in recs:
        stats = {"mean_fsc": r.get("mean_fsc")}
        try:
            out = p.run(r["image"], API, "snow_fsc", stats)
        except Exception as e:
            print(name, "ERR", str(e)[:60], flush=True); continue
        pred = out["prediction"].get("prediction") or out["prediction"]
        n += 1
        if pred.get("severity") == r.get("label"):
            ok += 1
        try:
            errs.append(abs(float(pred.get("mean_fsc")) - float(r.get("mean_fsc"))))
        except Exception:
            pass
        print("%-9s %-24s truth %-8s pred %-8s" % (name, os.path.basename(r["image"]),
                                                   r.get("label"), pred.get("severity")), flush=True)
    mae = sum(errs) / len(errs) if errs else float("nan")
    results[name] = {"n": n, "sev_ok": ok, "sev_acc": round(100.0 * ok / max(n, 1), 1),
                     "mae": round(mae, 3)}
    print("%-9s sev %d/%d MAE %.3f" % (name, ok, n, mae), flush=True)

json.dump(results, open(os.path.join(OUT, "snow_fsc_external_four_groups.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved")

# -*- coding: utf-8 -*-
"""四组对照：baseline / +视觉 / +语义 / +视觉+语义（snow_fsc）。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data\train_snow_fsc_json_val.json"
OUT = r"C:\vlm-fusion\outputs\four_groups"
os.makedirs(OUT, exist_ok=True)

recs = [r for r in json.load(open(VAL, encoding="utf-8"))
        if r.get("product") == "intrinsic_fractional_snow_cover" and os.path.exists(r["image"])]
# 按等级均衡取 12 张
by = {}
for r in recs:
    by.setdefault(r.get("label"), []).append(r)
picked = []
for lab in ("no_snow", "light", "moderate", "heavy"):
    picked += by.get(lab, [])[:3]
print("picked", len(picked), {k: len(v) for k, v in by.items()}, flush=True)

def cfg(visual=False, semantic=False):
    c = {
        "numeric": {"extractor": "none"},
        "visual": {"extractor": "resnet50" if visual else "none", "output": "map"},
        "semantic": {"extractor": "clip_vit_b32" if semantic else "none",
                     "labels": ["no snow", "light snow cover", "moderate snow cover", "heavy snow cover"]},
        "fusion": {"mode": ["text", "image"], "include_numeric": False,
                   "include_visual": visual, "include_semantic": semantic},
    }
    return c

GROUPS = [("baseline", False, False), ("+visual", True, False),
          ("+semantic", False, True), ("+both", True, True)]

results = {}
for name, vis, sem in GROUPS:
    p = ElementPipeline(cfg(vis, sem))
    ok = 0; n = 0; errs = []
    for r in picked:
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
        print("%-9s %-28s truth %-8s pred %-8s" % (name, os.path.basename(r["image"]),
                                                   r.get("label"), pred.get("severity")), flush=True)
    mae = sum(errs) / len(errs) if errs else float("nan")
    results[name] = {"n": n, "sev_ok": ok, "sev_acc": round(100.0 * ok / max(n, 1), 1),
                     "mae": round(mae, 3)}
    print("%-9s sev %d/%d  MAE %.3f" % (name, ok, n, mae), flush=True)

json.dump(results, open(os.path.join(OUT, "snow_fsc_four_groups.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved", os.path.join(OUT, "snow_fsc_four_groups.json"))

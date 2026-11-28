# -*- coding: utf-8 -*-
"""采集 snow_swe 验证集：模型读数、真值读数、真值等级。"""
import json, os, re, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline

VAL = r"C:\vlmdisaster\LLaMA-Factory\data\train_snow_swe_json_val.json"
OUT = r"C:\vlm-fusion\outputs\calib_snow_swe.json"
API = "http://127.0.0.1:8078"

rows = json.load(open(VAL, encoding="utf-8"))
cfg = {"element": "snow_swe",
       "numeric": {"extractor": "none"}, "visual": {"extractor": "none"},
       "semantic": {"extractor": "none"},
       "fusion": {"mode": ["text", "image"]}}
p = ElementPipeline(cfg)
recs = []
for r in rows:
    img = r.get("image")
    if not img or not os.path.exists(img):
        continue
    try:
        truth = json.loads(r["conversations"][1]["value"])
    except Exception:
        truth = {}
    out = p.run(img, API, "snow_swe", {})
    pred = out["prediction"].get("prediction") or out["prediction"]
    rec = {"image": os.path.basename(img), "truth_label": r.get("label"),
           "truth_mean_swe": truth.get("mean_swe_mm"),
           "pred_label": pred.get("severity"),
           "pred_mean_swe": pred.get("mean_swe_mm")}
    recs.append(rec)
    print("%-28s truth %-8s %.1f | pred %-8s %s" % (
        rec["image"], rec["truth_label"], rec["truth_mean_swe"] or -1,
        rec["pred_label"], rec["pred_mean_swe"]), flush=True)
json.dump(recs, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT, len(recs))

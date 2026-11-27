# -*- coding: utf-8 -*-
"""同一批样本的 baseline 判读（不接任何特征）。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
OUT = r"C:\vlm-fusion\outputs\four_groups"

for el in sys.argv[1:]:
    recs = [r for r in json.load(open(os.path.join(VAL, "train_%s_json_val.json" % el), encoding="utf-8"))
            if os.path.exists(r.get("image") or (r.get("images") or [""])[0])]
    by = {}
    for r in recs:
        by.setdefault(r.get("label"), []).append(r)
    picked = [r for lab in sorted(by) for r in by[lab][:2]]
    p = ElementPipeline({"element": el, "numeric": {"extractor": "none"},
                         "visual": {"extractor": "none"}, "semantic": {"extractor": "none"},
                         "fusion": {"mode": ["text", "image"]}})
    ok = 0; n = 0
    for r in picked:
        try:
            out = p.run(r.get("image") or r["images"][0], API, el, {})
        except Exception as e:
            continue
        pred = out["prediction"].get("prediction") or out["prediction"]
        n += 1
        ok += (pred.get("severity") == r.get("label"))
    print("%-18s baseline sev %d/%d" % (el, ok, n), flush=True)
    json.dump({"element": el, "n": n, "sev_ok": ok},
              open(os.path.join(OUT, "%s_baseline.json" % el), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
print("done")

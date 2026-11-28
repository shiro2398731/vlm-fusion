# -*- coding: utf-8 -*-
"""用 fusion_best.yaml 对启用元素做部署口径评测（2 张/类）。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import yaml
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
CFG = r"C:\vlm-fusion\configs\fusion_best.yaml"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
OUT = r"C:\vlm-fusion\docs\config_eval.json"

cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
results = {}
ok_all = n_all = 0
for el, c in cfg.items():
    if not isinstance(c, dict) or not c.get("enabled"):
        continue
    p = os.path.join(VAL, "train_%s_json_val.json" % el)
    if not os.path.exists(p):
        continue
    recs = [r for r in json.load(open(p, encoding="utf-8"))
            if os.path.exists(r.get("image") or (r.get("images") or [""])[0])]
    by = {}
    for r in recs:
        by.setdefault(r.get("label"), []).append(r)
    picked = [r for lab in sorted(by) for r in by[lab][:2]]
    pipe = ElementPipeline.from_yaml(CFG, el)
    ok = n = 0
    for r in picked:
        img = r.get("image") or r["images"][0]
        try:
            out = pipe.run(img, API, el, {})
        except Exception:
            continue
        pred = out["prediction"].get("prediction") or out["prediction"]
        n += 1
        ok += pred.get("severity") == r.get("label")
    results[el] = {"n": n, "ok": ok}
    ok_all += ok; n_all += n
    print("%-22s %d/%d" % (el, ok, n), flush=True)

json.dump({"elements": results, "total_ok": ok_all, "total_n": n_all},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("TOTAL %d/%d" % (ok_all, n_all))

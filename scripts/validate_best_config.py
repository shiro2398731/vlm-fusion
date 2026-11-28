# -*- coding: utf-8 -*-
"""验证 fusion_best.yaml 可执行：逐个启用元素跑一张样本。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import yaml
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
CFG = r"C:\vlm-fusion\configs\fusion_best.yaml"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
OUT = r"C:\vlm-fusion\docs\best_config_smoke.json"

cfgs = yaml.safe_load(open(CFG, encoding="utf-8"))
rows = []
ok = 0
for el, cfg in cfgs.items():
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        continue
    p = os.path.join(VAL, "train_%s_json_val.json" % el)
    if not os.path.exists(p):
        rows.append({"element": el, "status": "no_val"}); continue
    recs = [r for r in json.load(open(p, encoding="utf-8"))
            if os.path.exists(r.get("image") or (r.get("images") or [""])[0])]
    if not recs:
        rows.append({"element": el, "status": "no_image"}); continue
    r = recs[0]
    img = r.get("image") or r["images"][0]
    try:
        pipe = ElementPipeline.from_yaml(CFG, el)
        out = pipe.run(img, API, el, {})
        pred = out["prediction"]
        rows.append({"element": el, "status": "ok",
                     "severity": pred.get("severity"),
                     "source": pred.get("severity_source")})
        ok += 1
        print("%-22s ok sev=%-10s src=%s" % (el, pred.get("severity"),
                                             pred.get("severity_source")), flush=True)
    except Exception as e:
        rows.append({"element": el, "status": "error", "error": str(e)[:120]})
        print("%-22s ERR %s" % (el, str(e)[:80]), flush=True)
json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok %d / %d" % (ok, len(rows)))

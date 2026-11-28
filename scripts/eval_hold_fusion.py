# -*- coding: utf-8 -*-
"""hold 集上对比 baseline 与最终配置融合。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import yaml
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
CFG = r"C:\vlm-fusion\configs\fusion_best.yaml"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
OUT = r"C:\vlm-fusion\docs\hold_fusion.json"

cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
els = sys.argv[1:]
base_cfg = lambda el: {"element": el, "numeric": {"extractor": "none"},
                       "visual": {"extractor": "none"}, "semantic": {"extractor": "none"},
                       "fusion": {"mode": ["text", "image"]}}
res = {}
for el in els:
    p = os.path.join(VAL, "train_%s_hold_json_val.json" % el)
    if not os.path.exists(p):
        print(el, "no hold"); continue
    recs = [r for r in json.load(open(p, encoding="utf-8"))
            if os.path.exists(r.get("image") or (r.get("images") or [""])[0])]
    by = {}
    for r in recs:
        by.setdefault(r.get("label"), []).append(r)
    picked = [r for lab in sorted(by) for r in by[lab][:2]]
    pb = ElementPipeline(base_cfg(el))
    pf = ElementPipeline.from_yaml(CFG, el)
    okb = okf = 0
    for r in picked:
        img = r.get("image") or r["images"][0]
        ob = pb.run(img, API, el, {})
        of = pf.run(img, API, el, {})
        vb = (ob["prediction"].get("prediction") or ob["prediction"]).get("severity")
        vf = (of["prediction"].get("prediction") or of["prediction"]).get("severity")
        okb += vb == r.get("label"); okf += vf == r.get("label")
    res[el] = {"n": len(picked), "baseline": okb, "fused": okf}
    print("%-22s baseline %d/%d  fused %d/%d" % (el, okb, len(picked), okf, len(picked)), flush=True)

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
up = sum(1 for v in res.values() if v["fused"] > v["baseline"])
tie = sum(1 for v in res.values() if v["fused"] == v["baseline"])
down = sum(1 for v in res.values() if v["fused"] < v["baseline"])
print("hold: up %d tie %d down %d" % (up, tie, down))

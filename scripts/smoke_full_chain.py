# -*- coding: utf-8 -*-
"""完整新链路冒烟测试：38 元素逐个走 物理→视觉→语义→融合→Qwen。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
CAP = r"C:\vlm-fusion\elements_capability.json"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
PROBES = r"C:\vlm-fusion\outputs\probes"
OUT = r"C:\vlm-fusion\docs\full_chain_smoke.json"

cap = json.load(open(CAP, encoding="utf-8"))
results = []
ok = 0
for el in sorted(cap):
    p = os.path.join(VAL, "train_%s_json_val.json" % el)
    if not os.path.exists(p):
        results.append({"element": el, "status": "no_val_json"})
        continue
    recs = [r for r in json.load(open(p, encoding="utf-8"))
            if os.path.exists(r.get("image") or (r.get("images") or [""])[0])]
    if not recs:
        results.append({"element": el, "status": "no_image"})
        continue
    r = recs[0]
    img = r.get("image") or r["images"][0]
    cfg = {
        "element": el,
        "numeric": {"extractor": "numeric"},
        "visual": {"extractor": "resnet_probe",
                   "probe_path": os.path.join(PROBES, "%s_resnet_probe.joblib" % el)},
        "semantic": {"extractor": "clip_probe",
                     "probe_path": os.path.join(PROBES, "%s_clip_probe.joblib" % el)},
        "fusion": {"mode": ["text", "image"]},
    }
    try:
        pipe = ElementPipeline(cfg)
        out = pipe.run(img, API, el, {})
        pred = out["prediction"].get("prediction") or out["prediction"]
        lines = out["fusion"].splitlines()
        has_vec = any("feature vector" in s for s in lines)
        results.append({"element": el, "status": "ok",
                        "severity": pred.get("severity"),
                        "detected": pred.get("detected"),
                        "input_lines": len(lines), "vector_attached": has_vec})
        ok += 1
        print("%-22s ok sev=%-10s lines=%d vec=%s" % (el, pred.get("severity"), len(lines), has_vec), flush=True)
    except Exception as e:
        results.append({"element": el, "status": "error", "error": str(e)[:120]})
        print("%-22s ERR %s" % (el, str(e)[:80]), flush=True)

json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok %d / %d" % (ok, len(cap)))

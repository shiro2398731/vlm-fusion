# -*- coding: utf-8 -*-
"""决策级融合评测：Qwen 出基础判读，探针高置信时覆盖等级。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
PROBES = r"C:\vlm-fusion\outputs\probes"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
OUT = r"C:\vlm-fusion\outputs\four_groups"
os.makedirs(OUT, exist_ok=True)
TAU = float(os.environ.get("TAU", "0.8"))
SRC = os.environ.get("SRC", "semantic")
ELS = sys.argv[1:] or ["snow_swe", "coral_dhw", "sanddust", "volcano_so2"]

for el in ELS:
    vap = os.path.join(VAL, "train_%s_json_val.json" % el)
    recs = [r for r in json.load(open(vap, encoding="utf-8"))
            if os.path.exists(r.get("image") or (r.get("images") or [""])[0])]
    by = {}
    for r in recs:
        by.setdefault(r.get("label"), []).append(r)
    picked = [r for lab in sorted(by) for r in by[lab][:2]]
    probe = os.path.join(PROBES, "%s_clip_probe.joblib" % el if SRC == "semantic"
                         else "%s_resnet_probe.joblib" % el)
    cfg = {
        "element": el,
        "numeric": {"extractor": "none"},
        "visual": ({"extractor": "resnet_probe", "probe_path": probe}
                   if SRC == "visual" else {"extractor": "none"}),
        "semantic": ({"extractor": "clip_probe", "probe_path": probe}
                     if SRC == "semantic" else {"extractor": "none"}),
        "fusion": {"mode": ["decision"], "tau": TAU, "source": SRC},
    }
    p = ElementPipeline(cfg)
    ok = 0; n = 0; used = 0
    for r in picked:
        try:
            out = p.run(r.get("image") or r["images"][0], API, el, {})
        except Exception as e:
            print(el, "ERR", str(e)[:60], flush=True); continue
        pred = out["prediction"]
        n += 1
        if pred.get("severity") == r.get("label"):
            ok += 1
        if pred.get("severity_source") == "probe":
            used += 1
        print("%-12s %-30s truth %-10s final %-10s src %s" % (
            el, os.path.basename(r.get("image") or r["images"][0]), r.get("label"),
            pred.get("severity"), pred.get("severity_source")), flush=True)
    print("%-12s tau %.2f %s: sev %d/%d (probe覆盖 %d)" % (el, TAU, SRC, ok, n, used), flush=True)
    json.dump({"element": el, "tau": TAU, "source": SRC, "n": n, "sev_ok": ok},
              open(os.path.join(OUT, "%s_decision_%s.json" % (el, SRC)), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
print("done")

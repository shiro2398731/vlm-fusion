# -*- coding: utf-8 -*-
"""通用四组对照：argv[1]=element，自动取探针与验证集，2 张/类。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
EL = sys.argv[1]
PROBES = r"C:\vlm-fusion\outputs\probes"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data\train_%s_json_val.json" % EL
OUT = r"C:\vlm-fusion\outputs\four_groups"
os.makedirs(OUT, exist_ok=True)

recs = [r for r in json.load(open(VAL, encoding="utf-8")) if os.path.exists(r.get("image", ""))]
by = {}
for r in recs:
    by.setdefault(r.get("label"), []).append(r)
picked = []
for lab in sorted(by):
    picked += by[lab][:2]
print(EL, "picked", len(picked), {k: len(v) for k, v in by.items()}, flush=True)

VIS_PROBE = os.path.join(PROBES, "%s_resnet_probe.joblib" % EL)
SEM_PROBE = os.path.join(PROBES, "%s_clip_probe.joblib" % EL)
if not os.path.exists(VIS_PROBE):
    VIS_PROBE = None
if not os.path.exists(SEM_PROBE):
    SEM_PROBE = None

def cfg(visual=False, semantic=False):
    return {
        "element": EL,
        "numeric": {"extractor": "none"},
        "visual": ({"extractor": "resnet_probe", "probe_path": VIS_PROBE}
                   if visual and VIS_PROBE else {"extractor": "none"}),
        "semantic": ({"extractor": "clip_probe", "probe_path": SEM_PROBE}
                     if semantic and SEM_PROBE else {"extractor": "none"}),
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
        print("%-9s %-30s truth %-10s pred %-10s" % (name, os.path.basename(r["image"]),
                                                     r.get("label"), pred.get("severity")), flush=True)
    results[name] = {"n": n, "sev_ok": ok, "sev_acc": round(100.0 * ok / max(n, 1), 1)}
    print("%-9s sev %d/%d" % (name, ok, n), flush=True)

json.dump(results, open(os.path.join(OUT, "%s_four_groups.json" % EL), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved")

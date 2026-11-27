# -*- coding: utf-8 -*-
"""四组对照：记录 det、sev、数值偏差（数值真值从训练 JSON 的对话里解析）。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline

API = "http://127.0.0.1:8078"
PROBES = r"C:\vlm-fusion\outputs\probes"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
OUT = r"C:\vlm-fusion\outputs\four_groups"

def truth_json(r):
    try:
        return json.loads(r["conversations"][1]["value"])
    except Exception:
        return {}

def numerics(d):
    return {k: v for k, v in (d or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}

for el in sys.argv[1:]:
    recs = [r for r in json.load(open(os.path.join(VAL, "train_%s_json_val.json" % el), encoding="utf-8"))
            if os.path.exists(r.get("image") or (r.get("images") or [""])[0])]
    by = {}
    for r in recs:
        by.setdefault(r.get("label"), []).append(r)
    picked = [r for lab in sorted(by) for r in by[lab][:2]]

    def cfg(visual=False, semantic=False):
        return {
            "element": el,
            "numeric": {"extractor": "none"},
            "visual": ({"extractor": "resnet_probe",
                        "probe_path": os.path.join(PROBES, "%s_resnet_probe.joblib" % el)}
                       if visual else {"extractor": "none"}),
            "semantic": ({"extractor": "clip_probe",
                          "probe_path": os.path.join(PROBES, "%s_clip_probe.joblib" % el)}
                         if semantic else {"extractor": "none"}),
            "fusion": {"mode": ["text", "image"], "include_numeric": False,
                       "include_visual": visual, "include_semantic": semantic},
        }

    groups = [("baseline", False, False), ("+visual", True, False),
              ("+semantic", False, True), ("+both", True, True)]
    res = {}
    for name, vis, sem in groups:
        p = ElementPipeline(cfg(vis, sem))
        det_ok = sev_ok = n = 0
        errs = []
        for r in picked:
            img = r.get("image") or r["images"][0]
            out = p.run(img, API, el, {})
            pred = out["prediction"].get("prediction") or out["prediction"]
            n += 1
            det_ok += (bool(pred.get("detected")) == bool(truth_json(r).get("detected")))
            sev_ok += (pred.get("severity") == r.get("label"))
            t, q = numerics(truth_json(r)), numerics(pred)
            for k in t:
                if k in q:
                    try:
                        errs.append(abs(float(q[k]) - float(t[k])))
                    except Exception:
                        pass
        mae = sum(errs) / len(errs) if errs else float("nan")
        res[name] = {"n": n, "det": "%d/%d" % (det_ok, n), "sev": "%d/%d" % (sev_ok, n),
                     "num_mae": round(mae, 4) if errs else None}
        print("%-12s %-9s det %d/%d sev %d/%d MAE %s" % (el, name, det_ok, n, sev_ok, n,
                                                          res[name]["num_mae"]), flush=True)
    json.dump(res, open(os.path.join(OUT, "%s_metrics.json" % el), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
print("done")

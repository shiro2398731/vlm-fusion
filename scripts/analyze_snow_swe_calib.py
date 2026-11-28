# -*- coding: utf-8 -*-
"""snow_swe 数值校准与阈值规则分析。"""
import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

SRC = r"C:\vlm-fusion\outputs\calib_snow_swe.json"
OUT = r"C:\vlm-fusion\outputs\calib_snow_swe_report.json"

rows = [r for r in json.load(open(SRC, encoding="utf-8"))
        if r.get("truth_mean_swe") is not None and r["truth_mean_swe"] >= 0
        and r.get("pred_mean_swe") is not None]
truth = np.array([r["truth_mean_swe"] for r in rows], dtype=float)
pred = np.array([r["pred_mean_swe"] for r in rows], dtype=float)
labels = [r["truth_label"] for r in rows]

base_ok = sum(1 for r in rows if r["pred_label"] == r["truth_label"])
print("n", len(rows), "baseline sev", base_ok)

# 线性校准 truth = a*pred + b
A = np.vstack([pred, np.ones_like(pred)]).T
a, b = np.linalg.lstsq(A, truth, rcond=None)[0]
print("calibration: truth = %.3f * pred + %.3f" % (a, b))
calib = a * pred + b
mae_raw = float(np.mean(np.abs(pred - truth)))
mae_cal = float(np.mean(np.abs(calib - truth)))
print("MAE raw %.2f  calib %.2f" % (mae_raw, mae_cal))

EDGES = [(8.7, "none"), (29.3, "light"), (58.2, "moderate")]
def rule(v):
    for edge, lab in EDGES:
        if v < edge:
            return lab
    return "heavy"

raw_ok = sum(1 for v, r in zip(pred, rows) if rule(v) == r["truth_label"])
cal_ok = sum(1 for v, r in zip(calib, rows) if rule(v) == r["truth_label"])
print("rule raw", raw_ok, "/", len(rows))
print("rule calibrated", cal_ok, "/", len(rows))

json.dump({"n": len(rows), "baseline_sev": base_ok,
           "calibration": {"a": round(float(a), 3), "b": round(float(b), 3)},
           "mae_raw": round(mae_raw, 3), "mae_cal": round(mae_cal, 3),
           "rule_raw": raw_ok, "rule_cal": cal_ok,
           "edges": [e for e, _ in EDGES]},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)

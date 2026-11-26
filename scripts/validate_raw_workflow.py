# -*- coding: utf-8 -*-
r"""
原始影像工作流验收：渲染窗口 → 调服务判读 → 与水体占比阈值对照。

输入：raw_s2_to_ndwi.py 产出的 windows.json
输出：逐窗口的模型结论 vs 阈值标签，以及汇总准确率；写 raw_workflow_result.json

用法：
  python scripts/validate_raw_workflow.py --dir outputs/raw_test --api http://127.0.0.1:8078
"""
import argparse
import json
import os
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"C:\vlmdisaster\outputs\raw_test")
    ap.add_argument("--api", default="http://127.0.0.1:8078")
    ap.add_argument("--element", default="flood_opt")
    args = ap.parse_args()

    rows = json.load(open(os.path.join(args.dir, "windows.json"), encoding="utf-8"))
    print("窗口数", len(rows))

    out, right = [], 0
    for r in rows:
        img = r["image"]
        with open(img, "rb") as f:
            data = f.read()
        resp = requests.post(args.api + "/api/predict",
                             data={"element": args.element},
                             files={"file": (os.path.basename(img), data, "image/png")},
                             timeout=300)
        d = resp.json()
        pred = d.get("prediction") or {}
        model_sev = pred.get("severity")
        model_det = pred.get("detected")
        model_wf = pred.get("water_frac")
        truth = r["label"]
        ok = (model_sev == truth)
        right += 1 if ok else 0
        out.append({
            "image": os.path.basename(img),
            "truth_label": truth,
            "truth_water_frac": r["water_frac"],
            "model_severity": model_sev,
            "model_detected": model_det,
            "model_water_frac": model_wf,
            "match": ok,
        })
        print("%-34s 真值 %-12s 模型 %-12s %s" % (
            os.path.basename(img), truth, model_sev, "OK" if ok else "MISS"))

    summary = {"n": len(out), "match": right,
               "acc": round(right / max(1, len(out)), 3)}
    json.dump({"summary": summary, "rows": out},
              open(os.path.join(args.dir, "raw_workflow_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("汇总：%d/%d，准确率 %.3f" % (right, len(out), summary["acc"]))


if __name__ == "__main__":
    main()

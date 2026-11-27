# -*- coding: utf-8 -*-
"""汇总所有已跑过的 baseline vs 决策融合结果。"""
import glob, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = r"C:\vlm-fusion\outputs\four_groups"
rows = {}
for f in glob.glob(os.path.join(D, "*_baseline.json")):
    el = os.path.basename(f)[:-len("_baseline.json")]
    rows.setdefault(el, {})["base"] = json.load(open(f, encoding="utf-8"))
for f in glob.glob(os.path.join(D, "*_decision_semantic.json")):
    el = os.path.basename(f)[:-len("_decision_semantic.json")]
    rows.setdefault(el, {})["dec"] = json.load(open(f, encoding="utf-8"))

out = ["# 决策融合逐元素汇总", "",
       "| 元素 | baseline sev | 决策融合 | 变化 |", "|---|--:|--:|--:|"]
up = tie = down = 0
for el in sorted(rows):
    r = rows[el]
    if "base" not in r or "dec" not in r:
        continue
    b, d = r["base"], r["dec"]
    if b.get("n") != d.get("n"):
        continue
    delta = d["sev_ok"] - b["sev_ok"]
    if delta > 0:
        up += 1
    elif delta == 0:
        tie += 1
    else:
        down += 1
    out.append("| %s | %d/%d | %d/%d | %+d |" % (
        el, b["sev_ok"], b["n"], d["sev_ok"], d["n"], delta))
out.append("")
out.append("合计：升 %d，平 %d，降 %d（共 %d 个元素）" % (up, tie, down, up + tie + down))
p = r"C:\vlm-fusion\docs\decision_summary.md"
open(p, "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))

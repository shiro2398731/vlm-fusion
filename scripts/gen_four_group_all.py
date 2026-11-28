# -*- coding: utf-8 -*-
"""汇总所有元素的四组对照（baseline/+视觉/+语义/+both）。"""
import glob, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = r"C:\vlm-fusion\outputs\four_groups"
OUT = r"C:\vlm-fusion\docs\four_group_all.md"

import json as _json
CAP = _json.load(open(r"C:\vlm-fusion\elements_capability.json", encoding="utf-8"))
rows = []
for f in glob.glob(os.path.join(D, "*_four_groups.json")):
    el = os.path.basename(f)[:-len("_four_groups.json")]
    if el not in CAP:
        continue
    j = json.load(open(f, encoding="utf-8"))
    if "baseline" not in j:
        continue
    b = j["baseline"]
    if not b.get("n"):
        continue
    def acc(k):
        v = j.get(k) or {}
        return "%d/%d" % (v.get("sev_ok", 0), v.get("n", 0))
    rows.append((el, b["n"], acc("baseline"), acc("+visual"), acc("+semantic"), acc("+both"),
                 (j.get("+both") or {}).get("sev_ok", 0) - b.get("sev_ok", 0)))

rows.sort(key=lambda r: -r[6])
lines = ["# 四组对照全量汇总（三路向量默认开启）", "",
         "| 元素 | n | baseline | +视觉 | +语义 | +both | both-baseline |",
         "|---|--:|--:|--:|--:|--:|--:|"]
up = tie = down = 0
for r in rows:
    d = r[6]
    if d > 0: up += 1
    elif d == 0: tie += 1
    else: down += 1
    lines.append("| %s | %d | %s | %s | %s | %s | %+d |" % r[:7])
lines += ["", "合计（+both 对比 baseline）：升 %d、平 %d、降 %d，共 %d 元素" % (
    up, tie, down, len(rows))]
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("saved", OUT, "elements", len(rows))
print("\n".join(lines[:8]))

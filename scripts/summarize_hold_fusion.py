# -*- coding: utf-8 -*-
"""生成 hold 集 baseline vs 融合汇总表。"""
import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

d = json.load(open(r"C:\vlm-fusion\docs\hold_fusion.json", encoding="utf-8"))
lines = ["# hold 集（留出/外部）baseline vs 融合", "",
         "| 元素 | n | baseline | 融合 | 变化 |", "|---|--:|--:|--:|--:|"]
up = tie = down = 0
for el, v in sorted(d.items()):
    delta = v["fused"] - v["baseline"]
    if delta > 0: up += 1
    elif delta == 0: tie += 1
    else: down += 1
    lines.append("| %s | %d | %d | %d | %+d |" % (el, v["n"], v["baseline"], v["fused"], delta))
lines += ["", "合计：升 %d、平 %d、降 %d（共 %d 元素）" % (up, tie, down, len(d))]
open(r"C:\vlm-fusion\docs\hold_fusion.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

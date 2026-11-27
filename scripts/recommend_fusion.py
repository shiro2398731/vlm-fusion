# -*- coding: utf-8 -*-
"""按探针增益生成候选融合清单（只列候选，不代表已验证）。"""
import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

g = json.load(open(r"C:\vlm-fusion\outputs\probe_gains.json", encoding="utf-8"))
rows = sorted(g.items(), key=lambda kv: -kv[1]["gap"])
lines = ["# 候选融合元素（按探针相对 baseline sev 的增益排序）",
         "# 仅候选：需决策融合实测确认后才可开启", ""]
n = 0
for el, v in rows:
    if v["gap"] < 5:
        continue
    best = "clip" if v["probes"].get("clip", 0) >= v["probes"].get("resnet", 0) else "resnet"
    lines.append("%-22s gap=%+5.1f  baseline_sev=%5.1f  best_probe=%5.1f (%s)" % (
        el, v["gap"], v["baseline_sev"], v["best_probe"], best))
    n += 1
lines.append("")
lines.append("合计 %d 个候选" % n)
out = r"C:\vlm-fusion\configs\fusion_candidates.txt"
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines[:12]))
print("saved", out)

# -*- coding: utf-8 -*-
"""汇总 baseline / CLIP 决策 / DINOv2 决策 三方对照。"""
import glob, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = r"C:\vlm-fusion\outputs\four_groups"
OUT = r"C:\vlm-fusion\docs\decision_all_sources.md"

def load(kind):
    out = {}
    for f in glob.glob(os.path.join(D, "*_%s.json" % kind)):
        el = os.path.basename(f)[:-len("_%s.json" % kind)]
        out[el] = json.load(open(f, encoding="utf-8"))
    return out

base = load("baseline")
clip = load("decision_semantic")
dino = load("decision_dinov2")

lines = ["# 决策融合三方对照", "",
         "| 元素 | baseline | CLIP 决策 | DINOv2 决策 | 最优 |",
         "|---|--:|--:|--:|--:|"]
up = tie = down = 0
for el in sorted(base):
    b = base.get(el)
    c = clip.get(el)
    d = dino.get(el)
    if not b:
        continue
    row = [el, "%d/%d" % (b["sev_ok"], b["n"])]
    best = ("baseline", b["sev_ok"])
    for name, j in (("CLIP", c), ("DINOv2", d)):
        if j and j.get("n") == b.get("n"):
            row.append("%d/%d" % (j["sev_ok"], j["n"]))
            if j["sev_ok"] > best[1]:
                best = (name, j["sev_ok"])
        else:
            row.append("-")
    delta = best[1] - b["sev_ok"]
    if delta > 0:
        up += 1
    elif delta == 0:
        tie += 1
    else:
        down += 1
    row.append(("%s (%+d)" % (best[0], delta)) if delta else "baseline (0)")
    lines.append("| " + " | ".join(row) + " |")

lines += ["", "合计（以最优探针计）：升 %d、平 %d、降 %d，共 %d 元素" % (up, tie, down, up + tie + down)]
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("saved", OUT)
print("\n".join(lines[:12]))

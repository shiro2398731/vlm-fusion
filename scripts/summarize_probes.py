# -*- coding: utf-8 -*-
"""汇总 38 元素探针准确率与 baseline sev 的对照表。"""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_CLIP = r"C:\vlm-fusion\outputs\probes_clip.log"
LOG_RES = r"C:\vlm-fusion\outputs\probes_resnet.log"
CAP = r"C:\vlm-fusion\elements_capability.json"
OUT = r"C:\vlm-fusion\docs\probe_summary.md"

pat = re.compile(r"^(\S+)\s+train \d+ val \d+\s+acc ([\d.]+)%")
acc = {}
for path, kind in ((LOG_CLIP, "clip"), (LOG_RES, "resnet")):
    if not os.path.exists(path):
        continue
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.match(line.strip())
        if m:
            acc.setdefault(m.group(1), {})[kind] = float(m.group(2))

cap = json.load(open(CAP, encoding="utf-8"))
lines = ["# 38 元素探针与 baseline 对照", "",
         "| 元素 | baseline sev | CLIP 探针 | ResNet 探针 |",
         "|---|--:|--:|--:|"]
rows = []
for el in sorted(cap):
    a = acc.get(el, {})
    base = cap[el].get("sev")
    rows.append((el, base, a.get("clip"), a.get("resnet")))
    lines.append("| %s | %s | %s | %s |" % (
        el, base,
        "%.1f%%" % a["clip"] if "clip" in a else "-",
        "%.1f%%" % a["resnet"] if "resnet" in a else "-"))
lines += ["", "说明：baseline sev 为 vlm 同分布验证准确率；探针为该元素验证集上的准确率。",
          "注意：多数元素的验证集与训练集同日期，探针数字不代表跨日期泛化。"]
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print(open(OUT, encoding="utf-8").read()[:1500])
print("saved", OUT)

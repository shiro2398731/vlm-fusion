# -*- coding: utf-8 -*-
"""生成全元素总矩阵：baseline / 三种探针 / 最优决策 / 是否启用。"""
import glob, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CAP = r"C:\vlm-fusion\elements_capability.json"
D = r"C:\vlm-fusion\outputs\four_groups"
CFG = r"C:\vlm-fusion\configs\fusion_best.yaml"
OUT = r"C:\vlm-fusion\docs\final_matrix.md"

cap = json.load(open(CAP, encoding="utf-8"))

pat = re.compile(r"^(\S+)\s+train \d+ val \d+\s+acc ([\d.]+)%")
acc = {}
for path, kind in ((r"C:\vlm-fusion\outputs\probes_clip.log", "clip"),
                   (r"C:\vlm-fusion\outputs\probes_resnet.log", "resnet"),
                   (r"C:\vlm-fusion\outputs\probes_dinov2.log", "dinov2")):
    if not os.path.exists(path):
        continue
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.match(line.strip())
        if m:
            acc.setdefault(m.group(1), {})[kind] = float(m.group(2))

import yaml
cfg = yaml.safe_load(open(CFG, encoding="utf-8"))

base = {}
for f in glob.glob(os.path.join(D, "*_baseline.json")):
    el = os.path.basename(f)[:-len("_baseline.json")]
    base[el] = json.load(open(f, encoding="utf-8"))

dec = {}
for f in glob.glob(os.path.join(D, "*_decision_*.json")):
    name = os.path.basename(f)[:-5]
    el, src = name.split("_decision_", 1)
    j = json.load(open(f, encoding="utf-8"))
    if base.get(el, {}).get("n") != j.get("n"):
        continue
    cur = dec.get(el)
    if cur is None or j["sev_ok"] > cur["sev_ok"]:
        dec[el] = j

lines = ["# 全元素矩阵（baseline / 探针 / 决策融合）", "",
         "| 元素 | baseline sev | CLIP | ResNet | DINOv2 | 最优决策 | 启用 |",
         "|---|--:|--:|--:|--:|--:|:--:|"]
for el in sorted(cap):
    a = acc.get(el, {})
    b = base.get(el, {})
    d = dec.get(el, {})
    en = cfg.get(el, {})
    en = "✓" if isinstance(en, dict) and en.get("enabled") else ""
    lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
        el,
        "%d/%d" % (b["sev_ok"], b["n"]) if b else "-",
        "%.1f%%" % a["clip"] if "clip" in a else "-",
        "%.1f%%" % a["resnet"] if "resnet" in a else "-",
        "%.1f%%" % a["dinov2"] if "dinov2" in a else "-",
        "%d/%d" % (d["sev_ok"], d["n"]) if d else "-",
        en))
lines += ["", "说明：baseline 为同分布验证 sev；探针为各元素验证集准确率；",
          "最优决策为 2 张/类样本上的 sev；启用状态来自 fusion_best.yaml。"]
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("saved", OUT, "rows", len(cap))

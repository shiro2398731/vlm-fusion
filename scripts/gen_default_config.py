# -*- coding: utf-8 -*-
"""为所有有探针的元素生成默认融合配置（CLIP 为默认语义模型）。"""
import glob, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CAP = r"C:\vlm-fusion\elements_capability.json"
PROBES = r"C:\vlm-fusion\outputs\probes"
OUT = r"C:\vlm-fusion\configs\fusion_all.yaml"

cap = json.load(open(CAP, encoding="utf-8"))
lines = ["# 全部元素默认配置（由脚本生成）",
         "# 语义默认 CLIP 探针，视觉默认 ResNet50 探针，数值现算。",
         "# 逐元素开关见 fusion_selected.yaml；本文件用于完整链路默认跑通。",
         ""]
n = 0
for el in sorted(cap):
    cp = os.path.join(PROBES, "%s_clip_probe.joblib" % el)
    rp = os.path.join(PROBES, "%s_resnet_probe.joblib" % el)
    if not (os.path.exists(cp) and os.path.exists(rp)):
        continue
    n += 1
    lines.append("%s:" % el)
    lines.append("  numeric: {extractor: numeric}")
    lines.append("  visual: {extractor: resnet_probe, probe_path: %s}" % rp.replace("\\", "/"))
    lines.append("  semantic: {extractor: clip_probe, probe_path: %s}" % cp.replace("\\", "/"))
    lines.append("  fusion:")
    lines.append("    mode: [text, image]")
    lines.append("    include_numeric: true")
    lines.append("    include_visual: true")
    lines.append("    include_semantic: true")
    lines.append("    include_vectors: true")
    lines.append("  eval: {backend: local}")
    lines.append("  llm: {backend: local}")
    lines.append("")
lines.append("# 元素数：%d" % n)
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", OUT, "elements", n)

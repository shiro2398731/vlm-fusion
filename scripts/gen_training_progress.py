# -*- coding: utf-8 -*-
"""生成训练数据与模型进度表。"""
import glob, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import yaml

CAP = r"C:\vlm-fusion\elements_capability.json"
DATA = r"C:\vlmdisaster\LLaMA-Factory\data"
OUT_ROOT = r"C:\vlmdisaster\outputs"
CFG = r"C:\vlm-fusion\configs\fusion_best.yaml"
OUT = r"C:\vlm-fusion\docs\训练数据与模型进度.md"

cap = json.load(open(CAP, encoding="utf-8"))
cfg = yaml.safe_load(open(CFG, encoding="utf-8"))

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

def count(path):
    try:
        return len(json.load(open(path, encoding="utf-8")))
    except Exception:
        return 0

lines = ["# 训练数据与模型进度（38 元素）", "",
         "| 元素 | 训练样本 | 验证样本 | hold样本 | CLIP探针 | ResNet探针 | DINOv2探针 | LoRA挂件 | 融合 |",
         "|---|--:|--:|--:|--:|--:|--:|:--:|:--:|"]
for el in sorted(cap):
    tr = count(os.path.join(DATA, "train_%s_json.json" % el))
    va = count(os.path.join(DATA, "train_%s_json_val.json" % el))
    ho = count(os.path.join(DATA, "train_%s_hold_json_val.json" % el))
    a = acc.get(el, {})
    lora = "✓" if glob.glob(os.path.join(OUT_ROOT, "qwen3vl-lora-%s-*" % el)) else ""
    c = cfg.get(el) if isinstance(cfg, dict) else None
    fused = "开" if isinstance(c, dict) and c.get("enabled") else "关"
    lines.append("| %s | %d | %d | %d | %s | %s | %s | %s | %s |" % (
        el, tr, va, ho,
        "%.1f%%" % a["clip"] if "clip" in a else "-",
        "%.1f%%" % a["resnet"] if "resnet" in a else "-",
        "%.1f%%" % a["dinov2"] if "dinov2" in a else "-",
        lora, fused))
lines += ["", "说明：",
          "- 训练/验证样本数为 LLaMA-Factory data 中记录条数",
          "- 探针准确率为各元素验证集（多数与训练同日期）",
          "- LoRA 挂件为 vlm 基线训练产物（本仓库未重训）",
          "- 融合开/关来自 fusion_best.yaml"]
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("saved", OUT, "rows", len(cap))

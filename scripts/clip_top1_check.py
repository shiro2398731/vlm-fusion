# -*- coding: utf-8 -*-
"""CLIP 在渲染图上的 top-1 准确率（不经过 Qwen）。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from PIL import Image
from features.extractors import ClipExtractor

CASES = [
    (r"C:\vlmdisaster\LLaMA-Factory\data\train_coral_dhw_json_val.json", "image",
     {"none": "no heat stress", "watch": "bleaching watch",
      "alert1": "bleaching alert level 1", "alert2": "bleaching alert level 2"}),
    (r"C:\vlmdisaster\LLaMA-Factory\data\train_snow_fsc_json_val.json", "image",
     {"no_snow": "no snow cover", "light": "light snow cover",
      "moderate": "moderate snow cover", "heavy": "heavy snow cover"}),
]

ext = ClipExtractor()
ext.load()
for path, key, label_map in CASES:
    recs = json.load(open(path, encoding="utf-8"))
    recs = [r for r in recs if r.get("label") in label_map and os.path.exists(r.get(key, ""))]
    recs = recs[:30]
    labels = list(label_map.values())
    inv = {v: k for k, v in label_map.items()}
    ok = 0
    for r in recs:
        img = Image.open(r[key]).convert("RGB")
        prompts = [ext.prompt_template.format(label=l) for l in labels]
        inp = ext.proc(text=prompts, images=img, return_tensors="pt", padding=True)
        inp = {k: v.to(ext.device) for k, v in inp.items()}
        with ext.torch.no_grad():
            out = ext.model(**inp)
        top = labels[int(out.logits_per_image[0].argmax())]
        ok += (inv[top] == r["label"])
    print(os.path.basename(path), "n", len(recs), "top1 acc %.1f%%" % (100.0 * ok / max(len(recs), 1)))

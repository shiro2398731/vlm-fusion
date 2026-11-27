# -*- coding: utf-8 -*-
"""CLIP 嵌入 + 线性探针：检验语义分支是否可用。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from PIL import Image
from features.extractors import ClipExtractor

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
except Exception as e:
    print("sklearn missing:", e); raise SystemExit(1)

CASES = [
    ("coral", r"C:\vlmdisaster\LLaMA-Factory\data\train_coral_dhw_json.json",
     r"C:\vlmdisaster\LLaMA-Factory\data\train_coral_dhw_json_val.json"),
    ("snow_fsc", r"C:\vlmdisaster\LLaMA-Factory\data\train_snow_fsc_json.json",
     r"C:\vlmdisaster\LLaMA-Factory\data\train_snow_fsc_json_val.json"),
]

ext = ClipExtractor(); ext.load()

def embed(recs):
    X, y = [], []
    for r in recs:
        img = r.get("image")
        if not img or not os.path.exists(img):
            continue
        im = Image.open(img).convert("RGB")
        inp = ext.proc(images=im, return_tensors="pt")
        inp = {k: v.to(ext.device) for k, v in inp.items()}
        with ext.torch.no_grad():
            out = ext.model.get_image_features(pixel_values=inp["pixel_values"])
        X.append(out.pooler_output[0].detach().cpu().numpy())
        y.append(r.get("label"))
    return np.asarray(X), np.asarray(y)

for name, trp, vap in CASES:
    tr = json.load(open(trp, encoding="utf-8"))
    va = json.load(open(vap, encoding="utf-8"))
    Xtr, ytr = embed(tr)
    Xva, yva = embed(va)
    print(name, "train", Xtr.shape, "val", Xva.shape, flush=True)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yva, clf.predict(Xva))
    print("%s linear-probe acc %.1f%%" % (name, 100 * acc), flush=True)

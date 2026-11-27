# -*- coding: utf-8 -*-
"""为元素训练 CLIP 嵌入上的线性探针（语义支路）。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import joblib
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from features.extractors import ClipExtractor

OUT = r"C:\vlm-fusion\outputs\probes"
os.makedirs(OUT, exist_ok=True)
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"

ELEMENTS = sys.argv[1:] or ["coral_dhw", "snow_fsc"]
ext = ClipExtractor(); ext.load()

def embed(recs):
    X, y = [], []
    for r in recs:
        img = r.get("image") or (r.get("images") or [None])[0]
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

for el in ELEMENTS:
    trp = os.path.join(VAL, "train_%s_json.json" % el)
    vap = os.path.join(VAL, "train_%s_json_val.json" % el)
    if not (os.path.exists(trp) and os.path.exists(vap)):
        print(el, "missing json", flush=True); continue
    tr = json.load(open(trp, encoding="utf-8"))
    va = json.load(open(vap, encoding="utf-8"))
    Xtr, ytr = embed(tr)
    Xva, yva = embed(va)
    if len(Xtr) == 0 or len(Xva) == 0:
        print("%-16s skip: no images (%d/%d)" % (el, len(Xtr), len(Xva)), flush=True)
        continue
    clf = LogisticRegression(max_iter=3000)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yva, clf.predict(Xva))
    p = os.path.join(OUT, "%s_clip_probe.joblib" % el)
    pca = PCA(n_components=min(16, Xtr.shape[1]), random_state=0).fit(Xtr)
    joblib.dump({"model": clf, "labels": list(clf.classes_), "pca": pca}, p)
    print("%-16s train %d val %d  acc %.1f%%  -> %s" % (el, len(Xtr), len(Xva), 100 * acc, p), flush=True)

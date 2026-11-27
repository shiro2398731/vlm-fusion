# -*- coding: utf-8 -*-
"""训练 DINOv2 嵌入上的线性探针。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from features.extractors import DinoV2Extractor

OUT = r"C:\vlm-fusion\outputs\probes"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
ELS = sys.argv[1:] or ["snow_swe", "coral_dhw", "haze"]

ext = DinoV2Extractor(); ext.load()

def embed(recs):
    X, y = [], []
    for r in recs:
        img = r.get("image") or (r.get("images") or [None])[0]
        if not img or not os.path.exists(img):
            continue
        X.append(ext.embed(img)); y.append(r.get("label"))
    return np.asarray(X), np.asarray(y)

for el in ELS:
    trp = os.path.join(VAL, "train_%s_json.json" % el)
    vap = os.path.join(VAL, "train_%s_json_val.json" % el)
    if not (os.path.exists(trp) and os.path.exists(vap)):
        print(el, "missing json"); continue
    Xtr, ytr = embed(json.load(open(trp, encoding="utf-8")))
    Xva, yva = embed(json.load(open(vap, encoding="utf-8")))
    if len(Xtr) == 0 or len(Xva) == 0:
        print("%-14s skip: no images" % el); continue
    clf = LogisticRegression(max_iter=3000)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yva, clf.predict(Xva))
    p = os.path.join(OUT, "%s_dinov2_probe.joblib" % el)
    pca = PCA(n_components=min(16, Xtr.shape[1]), random_state=0).fit(Xtr)
    joblib.dump({"model": clf, "labels": list(clf.classes_), "pca": pca}, p)
    print("%-14s train %d val %d acc %.1f%% -> %s" % (el, len(Xtr), len(Xva), 100 * acc, p), flush=True)

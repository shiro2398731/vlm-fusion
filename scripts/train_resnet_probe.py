# -*- coding: utf-8 -*-
"""训练 ResNet50 嵌入上的线性探针（视觉支路）。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import joblib
import torch
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

OUT = r"C:\vlm-fusion\outputs\probes"
os.makedirs(OUT, exist_ok=True)
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
ELEMENTS = sys.argv[1:] or ["snow_swe", "coral_dhw"]

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).eval().to(dev)
feat = {}
model.layer4.register_forward_hook(lambda m, i, o: feat.__setitem__("m", o.detach()))
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def embed(recs):
    X, y = [], []
    for r in recs:
        img = r.get("image") or (r.get("images") or [None])[0]
        if not img or not os.path.exists(img):
            continue
        x = tf(Image.open(img).convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad():
            model(x)
        v = feat["m"][0].mean(dim=(1, 2)).cpu().numpy()
        X.append(v); y.append(r.get("label"))
    return np.asarray(X), np.asarray(y)

for el in ELEMENTS:
    trp = os.path.join(VAL, "train_%s_json.json" % el)
    vap = os.path.join(VAL, "train_%s_json_val.json" % el)
    if not (os.path.exists(trp) and os.path.exists(vap)):
        print(el, "missing"); continue
    Xtr, ytr = embed(json.load(open(trp, encoding="utf-8")))
    Xva, yva = embed(json.load(open(vap, encoding="utf-8")))
    if len(Xtr) == 0 or len(Xva) == 0:
        print("%-16s skip: no images (%d/%d)" % (el, len(Xtr), len(Xva)), flush=True)
        continue
    clf = LogisticRegression(max_iter=3000)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yva, clf.predict(Xva))
    p = os.path.join(OUT, "%s_resnet_probe.joblib" % el)
    joblib.dump({"model": clf, "labels": list(clf.classes_)}, p)
    print("%-14s train %d val %d acc %.1f%% -> %s" % (el, len(Xtr), len(Xva), 100 * acc, p), flush=True)

# -*- coding: utf-8 -*-
"""同一批样本上，比较探针单独判级的准确率（不经过 Qwen）。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import joblib
from PIL import Image
from features.extractors import ClipExtractor, ResNetExtractor

PROBES = r"C:\vlm-fusion\outputs\probes"
VAL = r"C:\vlmdisaster\LLaMA-Factory\data"

clip = ClipExtractor(); clip.load()
res = ResNetExtractor(output="vector"); res.load()

def clip_probs(path, probe):
    z = joblib.load(probe)
    img = Image.open(path).convert("RGB")
    inp = clip.proc(images=img, return_tensors="pt")
    inp = {k: v.to(clip.device) for k, v in inp.items()}
    with clip.torch.no_grad():
        out = clip.model.get_image_features(pixel_values=inp["pixel_values"])
    v = out.pooler_output[0].detach().cpu().numpy().reshape(1, -1)
    p = z["model"].predict_proba(v)[0]
    return z["labels"][int(p.argmax())]

def res_probs(path, probe):
    z = joblib.load(probe)
    v = res.extract(__import__("features.base", fromlist=["FeatureBundle"]).FeatureBundle(
        element="x", image_path=path)).visual_vector.reshape(1, -1)
    p = z["model"].predict_proba(v)[0]
    return z["labels"][int(p.argmax())]

for el in ("snow_swe", "coral_dhw", "sanddust", "volcano_so2"):
    vap = os.path.join(VAL, "train_%s_json_val.json" % el)
    recs = [r for r in json.load(open(vap, encoding="utf-8")) if os.path.exists(r.get("image", ""))]
    by = {}
    for r in recs:
        by.setdefault(r.get("label"), []).append(r)
    picked = [r for lab in sorted(by) for r in by[lab][:2]]
    cp = os.path.join(PROBES, "%s_clip_probe.joblib" % el)
    rp = os.path.join(PROBES, "%s_resnet_probe.joblib" % el)
    okc = okr = 0
    for r in picked:
        if os.path.exists(cp):
            okc += (clip_probs(r["image"], cp) == r["label"])
        if os.path.exists(rp):
            okr += (res_probs(r["image"], rp) == r["label"])
    print("%-14s n=%d  CLIP探针 %d  ResNet探针 %d" % (el, len(picked), okc, okr), flush=True)

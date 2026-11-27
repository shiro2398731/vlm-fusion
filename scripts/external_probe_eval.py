# -*- coding: utf-8 -*-
"""训练集外日期复核：snow_fsc 的外部样本上，探针与决策融合表现。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import joblib
from PIL import Image
from pipeline import ElementPipeline
from features.extractors import ClipExtractor, ResNetExtractor
from features.base import FeatureBundle

API = "http://127.0.0.1:8078"
PROBES = r"C:\vlm-fusion\outputs\probes"
DIRS = [r"C:\vlmdisaster\outputs\quality\snow_fsc_new",
        r"C:\vlmdisaster\outputs\quality\snow_fsc_external2"]

recs = []
for d in DIRS:
    p = os.path.join(d, "records.json")
    if not os.path.exists(p):
        continue
    for r in json.load(open(p, encoding="utf-8")):
        if r.get("product") != "intrinsic_fractional_snow_cover":
            continue
        r["image"] = os.path.join(d, "images", os.path.basename(r["image"]))
        recs.append(r)
print("external", len(recs), flush=True)

clip = ClipExtractor(); clip.load()
res = ResNetExtractor(output="vector"); res.load()
cz = joblib.load(os.path.join(PROBES, "snow_fsc_clip_probe.joblib"))
rz = joblib.load(os.path.join(PROBES, "snow_fsc_resnet_probe.joblib"))

okc = okr = 0
for r in recs:
    im = Image.open(r["image"]).convert("RGB")
    inp = clip.proc(images=im, return_tensors="pt")
    inp = {k: v.to(clip.device) for k, v in inp.items()}
    with clip.torch.no_grad():
        out = clip.model.get_image_features(pixel_values=inp["pixel_values"])
    v = out.pooler_output[0].detach().cpu().numpy().reshape(1, -1)
    pc = cz["model"].predict_proba(v)[0]
    topc = cz["labels"][int(pc.argmax())]
    okc += (topc == r["label"])

    b = FeatureBundle(element="snow_fsc", image_path=r["image"])
    b = res.extract(b)
    pr = rz["model"].predict_proba(b.visual_vector.reshape(1, -1))[0]
    topr = rz["labels"][int(pr.argmax())]
    okr += (topr == r["label"])
    print("%-40s truth %-9s clip %-9s resnet %-9s" % (
        os.path.basename(r["image"]), r["label"], topc, topr), flush=True)

print("external probe alone: CLIP %d/%d  ResNet %d/%d" % (okc, len(recs), okr, len(recs)), flush=True)

# 决策融合
cfg = {"element": "snow_fsc", "numeric": {"extractor": "none"},
       "visual": {"extractor": "none"},
       "semantic": {"extractor": "clip_probe",
                    "probe_path": os.path.join(PROBES, "snow_fsc_clip_probe.joblib")},
       "fusion": {"mode": ["decision"], "tau": 0.6, "source": "semantic"}}
p = ElementPipeline(cfg)
okd = 0
for r in recs:
    out = p.run(r["image"], API, "snow_fsc", {})
    pred = out["prediction"]
    okd += (pred.get("severity") == r["label"])
    print("%-40s truth %-9s decision %-9s src %s" % (
        os.path.basename(r["image"]), r["label"], pred.get("severity"),
        pred.get("severity_source")), flush=True)
print("external decision sev %d/%d" % (okd, len(recs)), flush=True)

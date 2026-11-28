# -*- coding: utf-8 -*-
"""日期干净元素的 hold 集外部验证：探针在训练集外日期上的准确率。"""
import json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from features.extractors import ClipExtractor, DinoV2Extractor

VAL = r"C:\vlmdisaster\LLaMA-Factory\data"
PROBES = r"C:\vlm-fusion\outputs\probes"
CAP = json.load(open(r"C:\vlm-fusion\elements_capability.json", encoding="utf-8"))

els = ["enso", "iod", "marine_heatwave", "flare"]
clip = ClipExtractor(); clip.load()
dino = DinoV2Extractor(); dino.load()

def scores_clip(img):
    import joblib
    z = joblib.load(os.path.join(PROBES, "%s_clip_probe.joblib" % el))
    from PIL import Image
    inp = clip.proc(images=Image.open(img).convert("RGB"), return_tensors="pt")
    inp = {k: v.to(clip.device) for k, v in inp.items()}
    with clip.torch.no_grad():
        out = clip.model.get_image_features(pixel_values=inp["pixel_values"])
    v = out.pooler_output[0].detach().cpu().numpy().reshape(1, -1)
    p = z["model"].predict_proba(v)[0]
    return z["labels"][int(p.argmax())]

rows_all = {}
for el in els:
    p = os.path.join(VAL, "train_%s_hold_json_val.json" % el)
    if not os.path.exists(p):
        print(el, "no hold"); continue
    recs = [r for r in json.load(open(p, encoding="utf-8")) if os.path.exists(r.get("image", ""))]
    if not recs:
        print(el, "no images"); continue
    ok = 0
    for r in recs:
        try:
            pred = scores_clip(r["image"])
        except Exception as e:
            print(el, "ERR", str(e)[:60]); break
        ok += pred == r.get("label")
    baseline = CAP.get(el, {}).get("holdout_sev")
    rows_all[el] = {"n": len(recs), "probe_ok": ok,
                    "probe_acc": round(100.0 * ok / len(recs), 1),
                    "baseline_holdout_sev": baseline}
    print("%-16s n=%d probe %.1f%%  baseline holdout sev %s" % (
        el, len(recs), 100.0 * ok / len(recs), baseline), flush=True)

json.dump(rows_all, open(r"C:\vlm-fusion\outputs\hold_probe_eval.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved")

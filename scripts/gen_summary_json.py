# -*- coding: utf-8 -*-
"""生成机器可读的全量汇总 JSON（供论文/表格二次加工）。"""
import glob, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import yaml

CAP = r"C:\vlm-fusion\elements_capability.json"
D = r"C:\vlm-fusion\outputs\four_groups"
CFG = r"C:\vlm-fusion\configs\fusion_best.yaml"
OUT = r"C:\vlm-fusion\docs\summary.json"

cap = json.load(open(CAP, encoding="utf-8"))
cfg = yaml.safe_load(open(CFG, encoding="utf-8"))

pat = re.compile(r"^(\S+)\s+train \d+ val \d+\s+acc ([\d.]+)%")
acc = {}
for path, kind in ((r"C:\vlm-fusion\outputs\probes_clip.log", "clip"),
                   (r"C:\vlm-fusion\outputs\probes_resnet.log", "resnet"),
                   (r"C:\vlm-fusion\outputs\probes_dinov2.log", "dinov2")):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            m = pat.match(line.strip())
            if m:
                acc.setdefault(m.group(1), {})[kind] = float(m.group(2))

base, dec = {}, {}
for f in glob.glob(os.path.join(D, "*_baseline.json")):
    el = os.path.basename(f)[:-len("_baseline.json")]
    base[el] = json.load(open(f, encoding="utf-8"))
for f in glob.glob(os.path.join(D, "*_decision_*.json")):
    name = os.path.basename(f)[:-5]
    el, src = name.split("_decision_", 1)
    j = json.load(open(f, encoding="utf-8"))
    if base.get(el, {}).get("n") != j.get("n"):
        continue
    if el not in dec or j["sev_ok"] > dec[el]["sev_ok"]:
        dec[el] = j

out = {}
deploy = json.load(open(r"C:\vlm-fusion\docs\config_eval.json", encoding="utf-8"))["elements"]
hold = json.load(open(r"C:\vlm-fusion\docs\hold_fusion.json", encoding="utf-8"))
for el in sorted(cap):
    c = cfg.get(el) or {}
    out[el] = {
        "baseline_det": cap[el].get("det"),
        "baseline_sev": cap[el].get("sev"),
        "probes": acc.get(el, {}),
        "decision_sev_ok": (dec.get(el) or {}).get("sev_ok"),
        "decision_n": (dec.get(el) or {}).get("n"),
        "deploy_ok": (deploy.get(el) or {}).get("ok"),
        "deploy_n": (deploy.get(el) or {}).get("n"),
        "hold_baseline": (hold.get(el) or {}).get("baseline"),
        "hold_fused": (hold.get(el) or {}).get("fused"),
        "fusion_enabled": bool(c.get("enabled")),
        "fusion_source": (c.get("fusion") or {}).get("source"),
    }
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT, "elements", len(out))

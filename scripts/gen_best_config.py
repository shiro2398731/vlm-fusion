# -*- coding: utf-8 -*-
"""生成逐元素“最优探针”可执行配置（fusion_best.yaml）。"""
import glob, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = r"C:\vlm-fusion\outputs\four_groups"
P = r"C:\vlm-fusion\outputs\probes"
OUT = r"C:\vlm-fusion\configs\fusion_best.yaml"

base = {}
for f in glob.glob(os.path.join(D, "*_baseline.json")):
    el = os.path.basename(f)[:-len("_baseline.json")]
    base[el] = json.load(open(f, encoding="utf-8"))

dec = {}
for f in glob.glob(os.path.join(D, "*_decision_*.json")):
    name = os.path.basename(f)[:-5]
    el, src = name.split("_decision_", 1)
    j = json.load(open(f, encoding="utf-8"))
    if base.get(el, {}).get("n") != j.get("n"):
        continue
    cur = dec.get(el)
    if cur is None or j["sev_ok"] > cur["json"]["sev_ok"]:
        dec[el] = {"src": src, "json": j}

lines = ["# 逐元素最优探针配置（由实测自动生成）",
         "# 默认 tau 0.6；enabled=false 表示保持 baseline",
         ""]
n_enabled = 0
for el in sorted(base):
    b = base[el]
    best = dec.get(el)
    enabled = bool(best and best["json"]["sev_ok"] > b["sev_ok"])
    if el == "snow_fsc":
        enabled = False   # 外部日期 baseline 8/8，融合会拉低
    src = best["src"] if best else "semantic"
    # 外部日期/原始链路证据优先：wind_gale 的 DINOv2 在外部 storm 上高置信误判，
    # CLIP 探针 4/4，改为 semanitic。
    if el == "wind_gale":
        src = "semantic"
    lines.append("%s:" % el)
    if not enabled:
        lines.append("  enabled: false")
    else:
        n_enabled += 1
        lines.append("  enabled: true")
        lines.append("  fusion: {mode: [decision], tau: 0.6, source: %s}" %
                     ("visual" if src in ("visual", "dinov2") else "semantic"))
        if src in ("visual", "dinov2"):
            vis = "dinov2_probe" if src == "dinov2" else "resnet_probe"
            lines.append("  visual: {extractor: %s, probe_path: outputs/probes/%s_%s.joblib}" %
                         (vis, el, vis))
        else:
            lines.append("  semantic: {extractor: clip_probe, probe_path: outputs/probes/%s_clip_probe.joblib}" % el)
    lines.append("")
lines.append("# enabled: %d / %d" % (n_enabled, len(base)))
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", OUT, "enabled", n_enabled, "/", len(base))

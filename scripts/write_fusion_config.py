# -*- coding: utf-8 -*-
"""由实测结果生成逐元素融合开关。"""
import glob, json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

D = r"C:\vlm-fusion\outputs\four_groups"
OVERRIDES = {"sanddust": "dinov2", "wind_gale": "dinov2", "wave": "dinov2"}
FORCE_DISABLE = {"snow_fsc"}   # 外部日期 baseline 8/8，融合会拉低
P = r"C:\vlm-fusion\outputs\probes"

base, dec = {}, {}
for f in glob.glob(os.path.join(D, "*_baseline.json")):
    el = os.path.basename(f)[:-len("_baseline.json")]
    base[el] = json.load(open(f, encoding="utf-8"))

# 每种决策源都读，逐元素取 sev_ok 最高者
best = {}
for f in glob.glob(os.path.join(D, "*_decision_*.json")):
    name = os.path.basename(f)[:-5]
    el, src = name.split("_decision_", 1)
    j = json.load(open(f, encoding="utf-8"))
    cur = best.get(el)
    if cur is None or j.get("sev_ok", 0) > cur["json"].get("sev_ok", 0):
        best[el] = {"src": src, "json": j}
dec = {el: v["json"] for el, v in best.items()}
DEC_SRC = {el: v["src"] for el, v in best.items()}

lines = ["# 逐元素融合开关（由实测生成）",
         "# enabled: true → Qwen 判读 + 探针高置信覆盖（tau 0.6）；false → 保持 baseline",
         "",
         "defaults:",
         "  mode: decision",
         "  tau: 0.6",
         "  source: semantic",
         "",
         "elements:"]
rows = []
for el in sorted(base):
    if el not in dec or base[el].get("n") != dec[el].get("n"):
        continue
    delta = dec[el]["sev_ok"] - base[el]["sev_ok"]
    if el in FORCE_DISABLE:
        en = "false"
        note = "external-baseline-perfect"
    elif delta > 0:
        en = "true"
        note = "+%d" % delta
    elif delta == 0:
        en = "false"
        note = "0"
    else:
        en = "false"
        note = "%d" % delta
    rows.append((el, en, note))
    src = DEC_SRC.get(el, "semantic")
    lines.append("  # base %d/%d -> fused %d/%d (%s)" % (
        base[el]["sev_ok"], base[el]["n"], dec[el]["sev_ok"], dec[el]["n"], src))
    lines.append("  %s:" % el)
    lines.append("    enabled: %s" % en)
    lines.append("    note: \"%s\"" % note)
    if src in ("visual", "dinov2"):
        lines.append("    source: visual")

# 外部日期已单独验证为获益
for el in ("wave", "earthquake_def"):
    if not any(r[0] == el for r in rows):
        lines.append("  %s:" % el)
        lines.append("    enabled: true")
        lines.append("    note: \"external-date\"")

open(r"C:\vlm-fusion\configs\fusion_selected.yaml", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("enabled:", sum(1 for r in rows if r[1] == "true"), "/", len(rows))
print("wrote configs/fusion_selected.yaml")

# -*- coding: utf-8 -*-
"""从训练集标签反推 snow_swe 数值等级边界。"""
import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN = r"C:\vlmdisaster\LLaMA-Factory\data\train_snow_swe_json.json"
OUT = r"C:\vlm-fusion\docs\criteria_snow_swe.json"

rows = json.load(open(TRAIN, encoding="utf-8"))
by = {}
for r in rows:
    try:
        truth = json.loads(r["conversations"][1]["value"])
    except Exception:
        continue
    v = truth.get("mean_swe_mm")
    if v is None:
        continue
    by.setdefault(r.get("label"), []).append(float(v))

stats = {}
for lab, vs in by.items():
    vs = sorted(vs)
    stats[lab] = {"n": len(vs), "min": vs[0], "max": vs[-1],
                  "p10": vs[int(0.1 * (len(vs) - 1))], "p90": vs[int(0.9 * (len(vs) - 1))]}
order = ["none", "light", "moderate", "heavy"]
edges = {}
for a, b in zip(order, order[1:]):
    if a in stats and b in stats:
        edges["%s->%s" % (a, b)] = round((stats[a]["p90"] + stats[b]["p10"]) / 2.0, 2)
json.dump({"stats": stats, "edges": edges}, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps({"stats": stats, "edges": edges}, ensure_ascii=False, indent=1))

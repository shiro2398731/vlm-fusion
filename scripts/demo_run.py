# -*- coding: utf-8 -*-
r"""
演示彩排：按三种采集方式各跑一个元素，确认整场演示顺畅。

    .\venv\Scripts\python.exe scripts\demo_run.py
    .\venv\Scripts\python.exe scripts\demo_run.py --py http://127.0.0.1:8078 --backend llm

流程：暖机 → 光学(洪水) → 热红外(热浪) → 重力(地下水) → 汇总研判
"""
import argparse
import io
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PICK = [("flood_opt", "光学·洪水"), ("heatwave_lst", "热红外·热浪"), ("groundwater_grace", "重力·地下水")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default="http://127.0.0.1:8078")
    ap.add_argument("--backend", default="llm", choices=["llm", "template"])
    a = ap.parse_args()
    base = a.py.rstrip("/")
    import requests

    t0 = time.time()
    cat = requests.get(base + "/api/catalog", timeout=60).json()
    by_el = {}
    for g in cat["groups"]:
        for e in g["elements"]:
            by_el[e["name"]] = e

    results = []
    for el, label in PICK:
        e = by_el.get(el)
        if not e or not e["samples"]:
            print("✗ %s 没有样例" % el)
            continue
        NEG = {"none", "normal", "quiet", "no_snow", "dry", "stable", "land", "blue", "calm", "clear", "open_water"}
        s = next((x for x in e["samples"] if str(x.get("label")) not in NEG), e["samples"][0])
        print("\n=== %s（%s，样例 %s）" % (label, el, s["tag"]))
        img = requests.get(base + s["url"], timeout=60).content
        t = time.time()
        r = requests.post(base + "/api/predict",
                          data={"element": el, "lat": "30.5", "lon": "114.3",
                                "report_backend": a.backend},
                          files={"file": ("s.png", io.BytesIO(img), "image/png")},
                          timeout=900)
        d = r.json()
        if r.status_code != 200 or not d.get("prediction"):
            print("✗ 失败：", str(d)[:160])
            continue
        p, rep = d["prediction"], d["report"]
        print("  检出=%s 等级=%s 风险=%s 用时 %.1fs" % (
            p.get("detected"), p.get("severity"), rep.get("risk"), time.time() - t))
        print("  研判：", (rep.get("summary") or "")[:70])
        results.append(p)

    if results:
        print("\n=== 汇总研判（%d 个元素）" % len(results))
        r = requests.post(base + "/api/report_aggregate",
                          json={"items": results, "lat": 30.5, "lon": 114.3, "backend": a.backend},
                          timeout=180)
        rep = (r.json() or {}).get("report", {})
        print("  综合风险：%s（后端 %s）" % (rep.get("risk"), rep.get("backend")))
        print("  综合研判：", (rep.get("summary") or "")[:120])
        for x in rep.get("actions", [])[:3]:
            print("   ·", x)

    print("\n总用时 %.0f 秒" % (time.time() - t0))


if __name__ == "__main__":
    main()

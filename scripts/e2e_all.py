# -*- coding: utf-8 -*-
r"""
全元素端到端自测：38 个元素各取一张样例，实际推理一遍，检查输出结构。

    .\venv\Scripts\python.exe scripts\e2e_all.py
    .\venv\Scripts\python.exe scripts\e2e_all.py --py http://127.0.0.1:8078 --backend template

产出：outputs/e2e_all_<时间>.json + 控制台汇总
"""
import argparse
import io
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REQ = ["disaster_type", "detected", "source", "physical_quantity", "severity", "criterion"]
LF_DATA = r"C:\vlmdisaster\LLaMA-Factory\data"


def required_fields(el):
    """按元素训练 prompt 里声明的字段来检查（不同元素 schema 不同）。"""
    for f in ("train_%s_json_val.json" % el, "train_%s_json.json" % el):
        p = os.path.join(LF_DATA, f)
        if not os.path.exists(p):
            continue
        try:
            recs = json.load(open(p, encoding="utf-8"))
            q = recs[0]["conversations"][0]["value"]
        except Exception:
            continue
        m = re.search(r"fields:\s*([^\.]+)", q)
        if m:
            fs = []
            for x in m.group(1).split(","):
                name = re.sub(r"\\(.*?\\)", "", x).strip().split(" ")[0]
                if name and re.match(r"^[a-z_]+$", name):
                    fs.append(name)
            if fs:
                return fs
    return REQ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default="http://127.0.0.1:8078")
    ap.add_argument("--backend", default="template", choices=["template", "llm"])
    ap.add_argument("--elements", default="")
    a = ap.parse_args()
    base = a.py.rstrip("/")
    import requests

    cat = requests.get(base + "/api/catalog", timeout=60).json()
    outdir = r"C:\vlmdisaster\outputs"
    stamp = time.strftime("%H%M%S")
    report_path = os.path.join(outdir, "e2e_all_%s.json" % stamp)

    rows = []
    t0 = time.time()
    n = 0
    total = sum(len(g["elements"]) for g in cat["groups"])
    only = [x for x in a.elements.split(",") if x]
    for g in cat["groups"]:
        for e in g["elements"]:
            if only and e["name"] not in only:
                continue
            n += 1
            if not e["samples"]:
                rows.append({"element": e["name"], "ok": False, "error": "无样例"})
                continue
            s = e["samples"][0]
            try:
                img = requests.get(base + s["url"], timeout=60).content
                t = time.time()
                r = requests.post(base + "/api/predict",
                                  data={"element": e["name"], "report_backend": a.backend},
                                  files={"file": ("s.png", io.BytesIO(img), "image/png")},
                                  timeout=600)
                dt = time.time() - t
                d = r.json()
                p = d.get("prediction") or {}
                missing = [k for k in required_fields(e["name"]) if k not in p]
                ok = r.status_code == 200 and not missing
                rows.append({
                    "element": e["name"], "zh": e["zh"], "ok": ok,
                    "missing": missing, "seconds": round(dt, 1),
                    "sample_tag": s.get("tag"), "sample_label": s.get("label"),
                    "detected": p.get("detected"), "severity": p.get("severity"),
                    "backend": (d.get("report") or {}).get("backend"),
                })
                flag = "✓" if ok else "✗"
                print("%s [%2d/%2d] %-22s %-6s det=%-5s sev=%-12s %.1fs" % (
                    flag, n, total, e["name"], s.get("tag"), p.get("detected"),
                    str(p.get("severity"))[:12], dt), flush=True)
            except Exception as ex:
                rows.append({"element": e["name"], "ok": False,
                             "error": "%s: %s" % (type(ex).__name__, str(ex)[:120])})
                print("✗ [%2d/%2d] %-22s %s" % (n, total, e["name"], type(ex).__name__), flush=True)

    ok_n = sum(1 for r in rows if r.get("ok"))
    json.dump({"rows": rows, "ok": ok_n, "total": len(rows),
               "seconds": round(time.time() - t0, 1)}, open(report_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n完成：%d/%d 通过，总用时 %.0f 秒" % (ok_n, len(rows), time.time() - t0))
    print("报告：", report_path)
    bad = [r["element"] for r in rows if not r.get("ok")]
    if bad:
        print("失败元素：", ", ".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
演示自检：明天上台前跑一条命令，确认演示链路每一环都正常。

用法：
    .\venv\Scripts\python.exe scripts\demo_check.py            # 只读检查（快）
    .\venv\Scripts\python.exe scripts\demo_check.py --warm      # 顺带暖机（跑一次推理）
    .\venv\Scripts\python.exe scripts\demo_check.py --py http://127.0.0.1:8078
"""
import argparse
import io
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

fails = []


def check(name, ok, detail=""):
    print(("✓ " if ok else "✗ ") + name + (("  → " + detail) if detail and not ok else ""))
    if not ok:
        fails.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default="http://127.0.0.1:8078")
    ap.add_argument("--warm", action="store_true")
    a = ap.parse_args()
    base = a.py.rstrip("/")

    import requests

    # 1. 服务
    try:
        st = requests.get(base + "/api/status", timeout=20).json()
        check("服务在线 /api/status", True)
    except Exception as e:
        check("服务在线 /api/status", False, "%s: %s" % (type(e).__name__, e))
        print("\n服务没起来：先运行 uvicorn scripts.serve:app --port 8078")
        sys.exit(1)

    check("元素数 = 38", st.get("elements") == 38, str(st.get("elements")))

    # 2. 目录与样例
    try:
        cat = requests.get(base + "/api/catalog", timeout=30).json()
        groups = cat.get("groups", [])
        check("采集方式分组 = 10", len(groups) == 10, str(len(groups)))
        total, unseen, empty = 0, 0, []
        for g in groups:
            for e in g["elements"]:
                if not e.get("samples"):
                    empty.append(e["name"])
                for s in e.get("samples", []):
                    total += 1
                    if s.get("tag") == "未见日期":
                        unseen += 1
        check("所有元素都有样例图", not empty, "缺样例: %s" % empty[:5])
        check("样例含未见日期图", unseen > 0, "unseen=%d" % unseen)
        print("   样例总数 %d，其中未见日期 %d" % (total, unseen))

        # 3. 每个分组抽一张样例，确认图片能打开
        bad = []
        for g in groups:
            e = g["elements"][0]
            if not e.get("samples"):
                bad.append(g["label"])
                continue
            u = e["samples"][0]["url"]
            try:
                r = requests.get(base + u, timeout=30)
                if r.status_code != 200 or len(r.content) < 1000:
                    bad.append("%s(%s)" % (g["label"], r.status_code))
            except Exception:
                bad.append(g["label"])
        check("每个采集方式至少一张样例可读", not bad, "异常: %s" % bad)
    except Exception as e:
        check("目录接口", False, "%s: %s" % (type(e).__name__, e))

    # 3b. 其余接口
    try:
        rd = requests.get(base + "/api/route?name=20240805_960_768_ndwi.png", timeout=20).json()
        c = (rd.get("candidates") or [{}])[0].get("element")
        check("文件名路由（ndwi → flood_opt）", c == "flood_opt", str(c))
    except Exception as e:
        check("文件名路由", False, str(e))
    try:
        ar = requests.get(base + "/api/artifacts", timeout=20).json()
        ok = len(ar.get("scripts", [])) >= 15 and len(ar.get("endpoints", [])) >= 6
        check("工程清单接口", ok, "scripts=%d endpoints=%d" % (len(ar.get("scripts", [])), len(ar.get("endpoints", [])))) 
    except Exception as e:
        check("工程清单接口", False, str(e))
    try:
        agg = requests.post(base + "/api/report_aggregate", json={"items": [
            {"element": "flood_opt", "disaster_type": "flood", "detected": True,
             "severity": "moderate", "criterion": "x"}], "backend": "template"}, timeout=60).json()
        rep = agg.get("report") or {}
        check("汇总研判接口", bool(rep.get("summary")), str(rep)[:80])
    except Exception as e:
        check("汇总研判接口", False, str(e))

    # 4. 大模型配置    try:
        llm = requests.get(base + "/api/llm", timeout=20).json()
        check("大模型已配置（可选）", llm.get("configured") is True,
              "未配置时研判会降级为模板，不影响演示")
        print("   模型：%s，key：%s" % (llm.get("model"), llm.get("api_key")))
    except Exception as e:
        check("大模型配置接口", False, str(e))

    # 5. 暖机
    if a.warm:
        try:
            cat = requests.get(base + "/api/catalog", timeout=30).json()
            sample = None
            for g in cat["groups"]:
                for e in g["elements"]:
                    if e["name"] == "flood_opt":
                        for s in e["samples"]:
                            if s.get("label") in ("flood", "waterlogging"):
                                sample = s
            if sample is None:
                check("暖机样例存在", False, "flood_opt 无正例样例")
            else:
                img = requests.get(base + sample["url"], timeout=60).content
                r = requests.post(base + "/api/predict",
                                  data={"element": "flood_opt", "lat": "30.5", "lon": "114.3",
                                        "report_backend": "template"},
                                  files={"file": ("w.png", io.BytesIO(img), "image/png")},
                                  timeout=900)
                d = r.json()
                ok = r.status_code == 200 and d.get("prediction") is not None
                check("暖机推理成功", ok, str(d)[:120])
                st2 = requests.get(base + "/api/status", timeout=20).json()
                check("模型已加载", st2.get("model_loaded") is True, str(st2))
                if ok:
                    p = d["prediction"]
                    print("   样例判定：detected=%s severity=%s" % (p.get("detected"), p.get("severity")))
        except Exception as e:
            check("暖机推理", False, "%s: %s" % (type(e).__name__, e))
    else:
        print("· 未做暖机（加 --warm 会跑一次推理，约 1–2 分钟）")

    print()
    if fails:
        print("有 %d 项未通过：%s" % (len(fails), ", ".join(fails)))
        sys.exit(1)
    print("全部通过，可以上台。")


if __name__ == "__main__":
    main()

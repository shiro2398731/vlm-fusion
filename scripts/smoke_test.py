# -*- coding: utf-8 -*-
r"""
自检（冒烟测试）：确认交付物的一致性与关键模块可用。

    .\venv\Scripts\python.exe scripts\smoke_test.py
"""
import json
import os
import re
import sys

ROOT = r"C:\vlmdisaster"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

fails = []


def check(name, cond, detail=""):
    print(("✓ " if cond else "✗ ") + name + (("  " + detail) if detail and not cond else ""))
    if not cond:
        fails.append(name)


def main():
    # 1. 附录 38 个元素，且 det 与评估结果一致
    cap_path = os.path.join(ROOT, "elements_capability.json")
    if not os.path.exists(cap_path):
        cap_path = os.path.join(ROOT, "处理方案", "元素能力_20261125.json")
    cap = json.load(open(cap_path, encoding="utf-8")) if os.path.exists(cap_path) else {}
    if cap:
        rows = {k: ("%.1f%%" % v["det"] if v.get("det") is not None else "—")
                for k, v in cap.items()}
    check("采用元素数 = 38", len(rows) == 38, "实际 %d" % len(rows))

    ev = {r["name"]: r for r in json.load(open(os.path.join(ROOT, "outputs", "eval_batch_result.json"),
                                                encoding="utf-8"))}
    bad = []
    for k, v in rows.items():
        e = ev.get(k)
        if not e or e.get("det") is None:
            continue
        try:
            av = float(v.rstrip("%"))
        except ValueError:
            continue
        if abs(av - e["det"] * 100) > 0.6:
            bad.append(k)
    check("附录 det 与评估结果一致", not bad, "不一致: %s" % bad[:5])

    # 2. 每个采用元素都有 adapter
    missing = [k for k in rows if not os.path.exists(
        os.path.join(ROOT, "outputs", "qwen3vl-lora-%s-json" % k, "adapter_config.json"))]
    check("38 个 adapter 齐全", not missing, "缺少: %s" % missing[:5])

    # 3. 逆地理编码
    from reverse_geocode import geocode
    g1 = geocode(39.9042, 116.4074)
    g2 = geocode(30.5, 114.3)
    g3 = geocode(0.0, 0.0)
    check("逆地理：北京", g1.get("province") == "北京市", str(g1))
    check("逆地理：武汉", (g2.get("province") or "").startswith("湖北"), str(g2))
    check("逆地理：范围外返回空", g3.get("country") is None, str(g3))

    # 4. 研判模板
    from report_llm import report
    r1 = report({"disaster_type": "flood", "detected": True, "severity": "heavy",
                 "physical_quantity": "ndwi", "criterion": "NDWI>=0.3", "lat": 30.5, "lon": 114.3})
    r2 = report({"disaster_type": "drought", "detected": False, "criterion": "x"})
    check("研判：检出高风险", r1.get("risk") == "高" and "洪涝" in r1.get("summary", ""), str(r1)[:120])
    check("研判：未检出低风险", r2.get("risk") == "低", str(r2)[:120])

    # 5. LLM 配置默认值
    from llm_config import load, mask
    c = load()
    check("LLM 配置有默认 base/model", bool(c.get("base_url") and c.get("model")), str(c))
    check("LLM key 掩码不泄露", mask("sk-abcdefghijklmn") == "sk-abc…klmn", mask("sk-abcdefghijklmn"))

    # 6. 服务代码可导入（不启动）
    try:
        import serve
        check("serve 可导入", hasattr(serve, "app"))
    except Exception as e:
        check("serve 可导入", False, str(e))

    print()
    if fails:
        print("失败 %d 项：%s" % (len(fails), ", ".join(fails)))
        sys.exit(1)
    print("全部通过（6 组检查）")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
智能体自检：不加载模型，把 8 个源的完整链路走一遍

## 检查什么

对每个源：
  1. 路由是否选出了正确的候选元素
  2. 判识结果能否汇总（用 mock 结果代替真实推理）
  3. 定位能否从 bbox 算出中心点
  4. 报告能否渲染

## 用法

  .\venv\Scripts\python.exe src\agent\selftest.py
"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent  # noqa: E402
import router  # noqa: E402

LF = os.path.join(router.ROOT, "LLaMA-Factory", "data")
BBOX = "103.5,30.0,104.7,31.0"      # 成都附近，只为验证中心点计算


def sample_image(el):
    """取该元素训练集第一条的图片路径"""
    info = json.load(open(router.INFO, encoding="utf-8"))
    key = "disaster_%s_json" % el
    if key not in info:
        return None
    recs = json.load(open(os.path.join(LF, info[key]["file_name"]), encoding="utf-8"))
    for r in recs:
        img = r.get("image") or (r.get("images") or [None])[0]
        if img and os.path.exists(str(img)):
            return str(img)
    return None


def main():
    table = router.load()
    print("=" * 78)
    print("  智能体自检（mock，不加载模型）")
    print("=" * 78)
    ok = fail = 0
    for cat, els in router.categories_in_use(table).items():
        cn = next((v[1] for k, v in router.CATEGORIES.items() if v[0] == cat), cat)
        el = sorted(els)[0]
        img = sample_image(el)
        if not img:
            print("[%s] %-10s ✗ 找不到样例图" % (cat, cn))
            fail += 1
            continue
        rep = agent.run(img, source=cat, bbox=BBOX, mock=True,
                        no_reverse=True, no_advice=True, verbose=False)
        if "error" in rep:
            print("[%s] %-10s ✗ %s" % (cat, cn, rep["error"]))
            fail += 1
            continue
        loc = rep["location"]
        good = (rep["candidates"] == sorted(els)
                and rep["detected"]
                and loc["lat"] is not None
                and not rep["failed"])
        print("[%s] %-10s %s 候选 %d 个，检出 %d 项，中心点 %.2f/%.2f"
              % (cat, cn, "✓" if good else "✗", len(rep["candidates"]),
                 len(rep["detected"]), loc["lat"], loc["lon"]))
        if good:
            ok += 1
        else:
            fail += 1
            print("      详情: candidates=%s failed=%s detected=%d"
                  % (rep["candidates"], rep["failed"], len(rep["detected"])))

    print("-" * 78)
    print("  通过 %d，失败 %d" % (ok, fail))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

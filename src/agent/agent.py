# -*- coding: utf-8 -*-
r"""
灾害识别智能体 · 主流程

## 分层

  路由  router.py      源 → 候选元素；问题模板从训练数据读
  执行  vlm_runner.py  一个基座 + 多 adapter 热切换
  定位  geo.py         输入端 bbox/经纬度 → 地名（不从像素反推）
  研判  advisor.py     结构化 JSON + 地名 → 处置建议

本文件只负责把它们串起来：run() 返回一份完整报告，main() 是命令行外壳。
界面（src/webui/app_v2.py）直接调 run()，不重复实现流程。

## 用法

  # 只看路由，不加载模型
  .\venv\Scripts\python.exe src\agent\agent.py --image X.png --source TIR --dry

  # 真跑
  .\venv\Scripts\python.exe src\agent\agent.py --image X.png --source TIR

  # 指定元素 + 坐标
  .\venv\Scripts\python.exe src\agent\agent.py --image X.png --elements wildfire --bbox 103.5,30.0,104.7,31.0

  # 全链路自检（不加载模型，验证汇总/定位/研判三段）
  .\venv\Scripts\python.exe src\agent\agent.py --image X.png --source TIR --mock
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import advisor  # noqa: E402
import geo  # noqa: E402
import router  # noqa: E402


def resolve_source(raw):
    """接受 OPT / 光学 / 01_optical_vnir 三种写法"""
    if not raw:
        return None
    up = raw.upper()
    for dirname, (code, cn) in router.CATEGORIES.items():
        if up == code or raw == dirname or raw == cn or raw.upper() == dirname.upper():
            return code
    return up


def pick_targets(table, source="", elements=""):
    """返回 (targets, source_label)；非法时抛 ValueError"""
    usable = router.available(table)
    if elements:
        targets = ([x.strip() for x in elements.split(",") if x.strip()]
                   if isinstance(elements, str) else list(elements))
        bad = [x for x in targets if x not in usable]
        if bad:
            raise ValueError("这些元素没有可用 adapter: %s" % ", ".join(bad))
        return targets, "手动指定"
    src = resolve_source(source)
    if not src:
        raise ValueError("必须给 source，或者用 elements 指定")
    targets = router.candidates(table, src)
    if not targets:
        raise ValueError("源 %s 下没有可用元素" % src)
    return targets, src


def _mock_predict(element, table):
    """不加载模型的假结果，仅用于验证汇总/定位/研判三段是否接对"""
    j = {
        "disaster_type": table[element].get("physical_quantity") or element,
        "detected": True,
        "source": "MOCK",
        "physical_quantity": table[element].get("physical_quantity"),
        "severity": "mock_severe",
        "criterion": "mock criterion",
        "mock": True,
    }
    return {"element": element, "raw": json.dumps(j, ensure_ascii=False), "json": j}


def run(image, source="", elements="", lat=None, lon=None, bbox="",
        max_new_tokens=200, no_reverse=False, no_advice=False, api_key="",
        mock=False, runner=None, verbose=True):
    """
    跑完整链路，返回 report dict。参数非法时返回 {"error": "..."}。
    runner 传入已加载的 VLMRunner 可避免每次重载基座（界面用）。
    """
    def log(*a):
        if verbose:
            print(*a, flush=True)

    if not os.path.exists(image):
        return {"error": "图片不存在: %s" % image}

    table = router.load()
    try:
        targets, src = pick_targets(table, source, elements)
    except ValueError as e:
        return {"error": str(e)}

    log("=" * 72)
    log("  影像: %s" % os.path.basename(image))
    log("  源:   %s" % src)
    log("  候选: %s" % ", ".join(targets))
    log("=" * 72)

    # ---- 1. 判识 ----
    own_runner = False
    results = []
    if mock:
        for el in targets:
            r = _mock_predict(el, table)
            results.append(r)
            log("  [mock] %s -> detected=%s" % (el, r["json"]["detected"]))
    else:
        if runner is None:
            from vlm_runner import VLMRunner
            runner = VLMRunner(targets, table)
            own_runner = True
        try:
            for el in targets:
                log("\n>>> %s" % el)
                if el not in runner.names:
                    results.append({"element": el, "raw": "", "json": None})
                    log("    ✗ 该 adapter 未加载")
                    continue
                r = runner.predict(el, image, max_new_tokens)
                results.append(r)
                if r["json"]:
                    log("    detected=%s severity=%s"
                        % (r["json"].get("detected"), r["json"].get("severity")))
                else:
                    log("    ✗ 输出无法解析为 JSON: %s" % r["raw"][:80])
        finally:
            if own_runner:
                runner.close()

    # ---- 2. 汇总 ----
    hits = [r for r in results if r["json"] and r["json"].get("detected")]
    detected = [r["json"] for r in hits]

    # ---- 3. 定位 ----
    pos = geo.resolve(bbox=bbox, lat=lat, lon=lon)
    if pos["ok"] and not no_reverse:
        place = geo.describe(pos["lat"], pos["lon"])
    elif pos["ok"]:
        place = "纬度 %.4f, 经度 %.4f" % (pos["lat"], pos["lon"])
    else:
        place = "位置未知（未提供 bbox 或经纬度）"
    location = {"lat": pos["lat"], "lon": pos["lon"], "place": place, "how": pos["how"]}

    # ---- 4. 研判 ----
    if no_advice:
        advice, advice_status = None, "已跳过（no_advice）"
    else:
        advice, advice_status = advisor.advise(
            detected, place, os.path.basename(image), api_key=api_key)

    return {
        "image": image,
        "source": src,
        "candidates": targets,
        "location": location,
        "detected": detected,
        "advice": advice,
        "advice_status": advice_status,
        "negative": [r["element"] for r in results
                     if r["json"] and not r["json"].get("detected")],
        "failed": [r["element"] for r in results if not r["json"]],
        "mock": mock,
        "raw": {r["element"]: r["raw"] for r in results},
    }


def render(report):
    """把 report 变成给人看的文本（命令行和界面共用）"""
    if "error" in report:
        return "✗ %s" % report["error"]
    lines = []
    if report["detected"]:
        lines.append("检出 %d 项：" % len(report["detected"]))
        for j in report["detected"]:
            extra = {k: v for k, v in j.items()
                     if k not in ("disaster_type", "detected", "severity", "criterion")}
            lines.append("  · %-24s %-24s %s"
                         % (str(j.get("physical_quantity") or j.get("disaster_type")),
                            str(j.get("severity")), extra))
    else:
        lines.append("未检出灾害（%d 个候选全部为 false）" % len(report["candidates"]))
    if report["negative"]:
        lines.append("判为无灾：%s" % ", ".join(report["negative"]))
    if report["failed"]:
        lines.append("解析失败：%s" % ", ".join(report["failed"]))
    loc = report["location"]
    lines.append("")
    lines.append("位置：%s（%s）" % (loc["place"], loc["how"]))
    if report["advice"]:
        lines.append("")
        lines.append("建议：")
        lines.extend("  " + ln for ln in report["advice"].splitlines())
    else:
        lines.append("建议：%s" % report["advice_status"])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="灾害识别智能体")
    ap.add_argument("--image", required=True, help="影像路径")
    ap.add_argument("--source", default="", help="TIR/OPT/SAR/PRECIP/MW/ALT/ATM/SPACE")
    ap.add_argument("--elements", default="", help="逗号分隔，绕过路由")
    ap.add_argument("--dry", action="store_true", help="只看路由，不加载模型")
    ap.add_argument("--mock", action="store_true", help="假结果，验证链路不加载模型")
    ap.add_argument("--out", default="", help="结果 JSON 路径")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--bbox", default="", help="minlon,minlat,maxlon,maxlat")
    ap.add_argument("--no-reverse", action="store_true", help="不调 Nominatim")
    ap.add_argument("--no-advice", action="store_true", help="不调用 LLM")
    ap.add_argument("--api-key", default="", help="DeepSeek key，缺省读 DEEPSEEK_API_KEY")
    a = ap.parse_args()

    if a.dry:
        if not os.path.exists(a.image):
            print("✗ 图片不存在: %s" % a.image)
            return 1
        table = router.load()
        try:
            targets, src = pick_targets(table, a.source, a.elements)
        except ValueError as e:
            print("✗ %s" % e)
            print("  当前可用源：%s" % ", ".join(sorted(router.categories_in_use(table))))
            return 1
        print("=" * 72)
        print("  影像: %s" % os.path.basename(a.image))
        print("  源:   %s" % src)
        print("  候选: %s" % ", ".join(targets))
        print("=" * 72)
        for el in targets:
            q = table[el]["question"].replace("<image>", "").strip()
            print("\n[%s]" % el)
            print("  问题: %s" % (q[:100] + ("..." if len(q) > 100 else "")))
            print("  训练: %d 条  验证: %d 条" % (table[el]["n_train"], table[el]["n_val"]))
        print("\n[dry] 未加载模型")
        return 0

    report = run(a.image, source=a.source, elements=a.elements, lat=a.lat, lon=a.lon,
                 bbox=a.bbox, max_new_tokens=a.max_new_tokens,
                 no_reverse=a.no_reverse, no_advice=a.no_advice,
                 api_key=a.api_key, mock=a.mock)

    print("\n" + "=" * 72)
    print(render(report))
    print("=" * 72)

    if "error" in report:
        return 1
    out = a.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "outputs", "agent_results", os.path.splitext(os.path.basename(a.image))[0] + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  结果已写 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

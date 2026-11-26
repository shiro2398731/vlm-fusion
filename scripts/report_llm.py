# -*- coding: utf-8 -*-
r"""
结构化结果 → 研判文字（LLM / 模板双后端）

输入（模型输出的 JSON，示例）：
    {"disaster_type": "flood", "detected": true, "source": "OPT",
     "physical_quantity": "ndwi", "severity": "moderate",
     "criterion": "NDWI >= 0.3", "lat": 30.5, "lon": 114.3}

输出：
    {"summary": "...", "risk": "中", "actions": ["..."], "location": {...}}

后端：
    - template（默认，离线，无需 API）
    - openai   需要环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL（OpenAI 兼容接口）

用法：
    .\venv\Scripts\python.exe scripts\report_llm.py --json '{"disaster_type":"flood","detected":true,"severity":"heavy","lat":30.5,"lon":114.3}'
    .\venv\Scripts\python.exe scripts\report_llm.py --file outputs/sample_result.json --backend openai
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\vlmdisaster\scripts")
from reverse_geocode import geocode  # noqa: E402

ZH = {
    "flood": "洪涝", "flood_water": "洪水淹没", "drought": "干旱", "heatwave": "热浪",
    "snowstorm": "暴雪", "snow_swe": "积雪", "snow_fsc": "积雪覆盖", "sea_ice": "海冰",
    "volcano": "火山活动", "volcano_so2": "火山 SO2", "volcano_thermal": "火山热异常",
    "earthquake": "地震形变", "ground_subsidence": "地面沉降",
    "landslide": "滑坡", "wildfire": "野火", "wildfire_smoke": "烟羽",
    "sanddust": "沙尘", "haze": "霾", "typhoon": "台风", "hurricane": "飓风",
    "coral_bleaching": "珊瑚白化", "bloom": "水华", "green_tide": "绿潮",
    "marine_heatwave": "海洋热浪", "enso": "ENSO", "iod": "印度洋偶极",
    "wave": "海浪", "wind_gale": "大风", "coastal": "海岸高水位",
    "geomagnetic": "地磁暴", "flare": "太阳耀斑", "ice_loss": "冰盖消融",
}

SEV = {"light": "轻", "moderate": "中", "heavy": "重", "severe": "严重",
       "minor": "轻", "major": "重", "none": "无", "burned": "已烧", "ice": "结冰",
       "waterlogging": "内涝", "inundation": "淹没", "flooded": "淹没", "open_water": "开放水域",
       "open_water": "开放水域", "drought": "干旱", "quiet": "静默",
       "c_flare": "C 级耀斑", "m_flare": "M 级耀斑", "x_flare": "X 级耀斑"}

ACTIONS = {
    "flood": ["核查受灾范围与人口", "关注上游降水预报", "准备排水与转移预案"],
    "drought": ["核查土壤墒情与水源", "评估农业受影响面积", "关注未来降水"],
    "heatwave": ["关注高温预警与用电负荷", "关注脆弱人群健康风险"],
    "wildfire": ["核实火线与风向", "关注周边居民疏散"],
    "sanddust": ["关注空气质量与能见度", "提醒交通与户外作业防护"],
    "sea_ice": ["关注航道结冰与破冰需求"],
    "ground_subsidence": ["核查沉降区基础设施", "关注地下水开采量"],
    "volcano_thermal": ["关注火山监测台网数据", "核查周边居民与航线"],
    "snowstorm": ["关注道路结冰与交通管制"],
}


def zh_name(dtype):
    return ZH.get(dtype, dtype or "未知")


def zh_sev(sev):
    return SEV.get(sev, sev or "未分级")


def template_report(d):
    dt = d.get("disaster_type", "")
    det = bool(d.get("detected"))
    sev = zh_sev(d.get("severity"))
    pq = d.get("physical_quantity", "")
    crit = d.get("criterion", "")
    loc = None
    if d.get("lat") is not None and d.get("lon") is not None:
        loc = geocode(d["lat"], d["lon"])
    where = ""
    if loc and loc.get("province"):
        where = "，位置约在%s%s" % (loc.get("province"), ("·" + loc["city"]) if loc.get("city") else "")
    if not det:
        summary = "未检出%s%s。判据：%s。" % (zh_name(dt), where, crit or "—")
        risk = "低"
    else:
        summary = "检出%s%s，等级为「%s」。判据：%s（物理量 %s）。" % (
            zh_name(dt), where, sev, crit or "—", pq or "—")
        risk = {"轻": "低", "中": "中", "重": "高", "严重": "高"}.get(sev, "中")
    acts = ACTIONS.get(dt, ["结合现场与气象资料复核"]) if det else ["保持例行监测"]
    return {
        "summary": summary,
        "risk": risk,
        "actions": acts,
        "location": loc,
        "backend": "template",
    }


def openai_report(d):
    from llm_config import load as load_llm
    cfg = load_llm()
    base = (cfg.get("base_url") or "").rstrip("/")
    key = cfg.get("api_key") or ""
    model = cfg.get("model") or ""
    if not (base and key and model):
        raise RuntimeError("未配置 LLM（scripts/llm_config.py set --api-key ...）")
    import urllib.request

    loc = None
    if d.get("lat") is not None and d.get("lon") is not None:
        loc = geocode(d["lat"], d["lon"])
    prompt = (
        "你是遥感灾害研判助手。根据以下结构化识别结果，用中文写一段 120 字以内的研判，"
        "并给出风险等级（低/中/高）与 2-3 条处置建议。只输出 JSON："
        '{"summary":"...","risk":"低|中|高","actions":["..."]}\n结果：'
        + json.dumps({**d, "location": loc}, ensure_ascii=False)
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    resp = json.loads(urllib.request.urlopen(req, timeout=90).read())
    text = resp["choices"][0]["message"]["content"]
    got = json.loads(text[text.find("{"):text.rfind("}") + 1])
    got.update({"location": loc, "backend": "openai:%s" % model})
    return got


def report(d, backend="template"):
    if backend == "openai":
        try:
            return openai_report(d)
        except Exception as e:
            r = template_report(d)
            r["backend"] = "template(fallback: %s)" % type(e).__name__
            return r
    return template_report(d)


def _agg_prompt(items, loc):
    lines = []
    for p in items:
        lines.append("- %s（%s）：检出=%s，等级=%s，物理量=%s，判据=%s" % (
            ZH.get(p.get("disaster_type"), p.get("disaster_type") or p.get("element") or "未知"),
            p.get("element", ""),
            "是" if p.get("detected") else "否",
            p.get("severity", "—"),
            p.get("physical_quantity", "—"),
            p.get("criterion", "—")))
    where = ""
    if loc and loc.get("province"):
        where = "位置约在%s%s。" % (loc["province"], ("·" + loc["city"]) if loc.get("city") else "")
    return ("你是遥感灾害研判助手。下面是同一地区的多个遥感元素识别结果，"
            "请综合判断该地区的主要灾害风险与次生风险，用中文写 150 字以内研判，"
            "并给出风险等级（低/中/高）与 3 条处置建议。只输出 JSON："
            '{"summary":"...","risk":"低|中|高","actions":["...","...","..."]}\n'
            + where + "要素结果：\n" + "\n".join(lines))


def aggregate_report(items, lat=None, lon=None, backend="template"):
    """多元素汇总研判：把多个元素的结构化结果交给 LLM 综合评估。"""
    loc = geocode(lat, lon) if (lat is not None and lon is not None) else None
    detected = [p for p in items if p.get("detected")]
    if backend == "openai":
        try:
            from llm_config import load as load_llm
            import urllib.request
            cfg = load_llm()
            base, key, model = (cfg.get("base_url") or "").rstrip("/"), cfg.get("api_key"), cfg.get("model")
            if not (base and key and model):
                raise RuntimeError("no llm config")
            body = json.dumps({"model": model, "temperature": 0.3,
                               "messages": [{"role": "user", "content": _agg_prompt(items, loc)}]}).encode()
            req = urllib.request.Request(base + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + key})
            resp = json.loads(urllib.request.urlopen(req, timeout=90).read())
            text = resp["choices"][0]["message"]["content"]
            got = json.loads(text[text.find("{"):text.rfind("}") + 1])
            got.update({"location": loc, "backend": "openai:%s" % model, "n": len(items)})
            return got
        except Exception as e:
            r = _agg_template(items, loc)
            r["backend"] = "template(fallback: %s)" % type(e).__name__
            return r
    return _agg_template(items, loc)


def _agg_template(items, loc):
    det = [p for p in items if p.get("detected")]
    where = ""
    if loc and loc.get("province"):
        where = "（%s%s）" % (loc["province"], ("·" + loc["city"]) if loc.get("city") else "")
    if not det:
        return {"summary": "本次 %d 个元素%s均未检出灾害信号。" % (len(items), where),
                "risk": "低", "actions": ["保持例行监测"], "location": loc,
                "backend": "template", "n": len(items)}
    names = "、".join(sorted({ZH.get(p.get("disaster_type"), p.get("element", "")) for p in det}))
    risk = "高" if any(p.get("severity") in ("heavy", "severe", "major", "waterlogging",
                                             "moderate", "m_flare", "x_flare") for p in det) else "中"
    acts = []
    for p in det[:3]:
        acts += ACTIONS.get(p.get("disaster_type"), [])[:1]
    return {"summary": "本次 %d 个元素%s检出 %s，需重点关注。" % (len(items), where, names),
            "risk": risk, "actions": acts or ["结合现场资料复核"], "location": loc,
            "backend": "template", "n": len(items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--file", default="")
    ap.add_argument("--backend", default="template", choices=["template", "openai"])
    a = ap.parse_args()
    d = json.loads(a.json) if a.json else json.load(open(a.file, encoding="utf-8"))
    out = report(d, a.backend)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

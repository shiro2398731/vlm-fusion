# -*- coding: utf-8 -*-
r"""
质量审计 v2：按每个元素的 prompt 自身定义检查（不再用通用假设）。

修正 v1 的两个误报来源：
  1. 字段完整率：用 prompt 里列出的字段，而不是 gold 的全部字段（gold 常带 *_unit 附加项）
  2. det/sev 自洽：每个元素的「无灾」类名不同（neutral/stable/quiet/blue/land/dry…），
     从 gold 里按 detected=False 的主导值自动识别，不再只认 none/normal

用法：
  .\venv\Scripts\python.exe scripts\quality_audit.py --limit 16 --batch 6
  .\venv\Scripts\python.exe scripts\quality_audit.py --elements wave,marine_heatwave --limit 16
"""
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\vlmdisaster\scripts")

from eval_batch import load_val  # noqa: E402
from eval_single import INFO, LF_DATA, MODEL, OUT_ROOT, metrics, parse_json  # noqa: E402

ROOT = r"C:\vlmdisaster"
OUT_MD = os.path.join(ROOT, "outputs", "quality_audit_report.md")
OUT_JSON = os.path.join(ROOT, "outputs", "quality_audit_v2.json")

FIXED = {"disaster_type", "detected", "source", "physical_quantity", "severity", "criterion"}


def adopted_elements():
    try:
        from elements_registry import adopted
        return adopted()
    except Exception:
        return []


def fields_of_prompt(q):
    """从单条 prompt 里解析字段名（每条样本可能不同）。"""
    m = re.search(r"fields:\s*(.*?)\.\s", (q or "") + " ")
    out = []
    if m:
        for part in m.group(1).split(","):
            nm = part.strip().split("(")[0].strip()
            if nm:
                out.append(nm)
    return out


def prompt_spec(name):
    """从训练数据里读 prompt 字段列表与 severity 词表。"""
    info = json.load(open(INFO, encoding="utf-8"))
    for k in info:
        if k.startswith("disaster_%s_json" % name):
            path = os.path.join(LF_DATA, info[k]["file_name"])
            if os.path.exists(path):
                recs = json.load(open(path, encoding="utf-8"))
                if not recs:
                    continue
                q = recs[0]["conversations"][0]["value"]
                m = re.search(r"fields:\s*(.*?)\.\s", q + " ")
                fields = []
                if m:
                    for part in m.group(1).split(","):
                        nm = part.strip().split("(")[0].strip()
                        if nm:
                            fields.append(nm)
                m2 = re.search(r"severity must be one of these exact level words:\s*([^.]*)", q)
                levels = []
                if m2:
                    levels = [x.strip() for x in re.split(r"[/,]", m2.group(1)) if x.strip()]
                return fields, levels
    return [], []


def none_classes(gold_list):
    """从 gold 中识别「无灾」类：detected=False 时占主导的 severity 值。"""
    from collections import Counter
    neg = Counter()
    tot = Counter()
    for g in gold_list:
        if not isinstance(g, dict):
            continue
        s = str(g.get("severity", "")).strip().lower()
        tot[s] += 1
        if g.get("detected") is False:
            neg[s] += 1
    out = set()
    for s, c in neg.items():
        if tot[s] and c / tot[s] >= 0.8:
            out.add(s)
    return out


def numeric_field(fields, gold_list):
    for k in fields:
        if k in FIXED:
            continue
        vals = [g.get(k) for g in gold_list if isinstance(g, dict)]
        if vals and sum(isinstance(v, (int, float)) for v in vals) >= max(3, len(vals) * 0.8):
            return k
    return None


def audit_element(name, limit, batch_size):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    rows = load_val(name, limit)
    if not rows:
        return {"name": name, "error": "no val rows"}
    fields0, levels = prompt_spec(name)

    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qc, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    model = PeftModel.from_pretrained(base, os.path.join(OUT_ROOT, "qwen3vl-lora-%s-json" % name),
                                      adapter_name="audit_adapter")
    model.eval()

    details = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        texts, images = [], []
        for q, g, img in chunk:
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": "file://%s" % img},
                {"type": "text", "text": q.replace("<image>", "").strip()}]}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            imgs, _ = process_vision_info(msgs)
            images.extend(imgs if imgs else [])
        inp = proc(text=texts, images=images, return_tensors="pt", padding=True).to(base.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=320, do_sample=False, repetition_penalty=1.05)
        decoded = proc.tokenizer.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                              skip_special_tokens=True)
        for (q, g, img), pt in zip(chunk, decoded):
            details.append({"gold": parse_json(g), "pred": parse_json(pt), "raw": pt,
                            "fields": fields_of_prompt(q)})

    del model, base
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    n = len(details)
    parse_ok = sum(1 for d in details if isinstance(d["pred"], dict))
    golds = [d["gold"] for d in details]
    nc = none_classes(golds)
    val_fields = next((d.get("fields") for d in details if d.get("fields")), None)
    nf = numeric_field(val_fields or fields0, golds)

    schema_ok = 0
    bad = []
    for d in details:
        p = d["pred"]
        req = d.get("fields") or fields0
        if isinstance(p, dict) and all(k in p for k in req):
            schema_ok += 1
        elif len(bad) < 3:
            missing = [k for k in req if not isinstance(p, dict) or k not in p]
            bad.append({"raw": (d.get("raw") or ""), "missing": missing,
                        "pred": (json.dumps(p, ensure_ascii=False)[:160] if isinstance(p, dict) else str(p)[:80])})

    sev_counter = {}
    for d in details:
        p = d["pred"]
        if isinstance(p, dict):
            s = str(p.get("severity", "")).strip().lower()
            sev_counter[s] = sev_counter.get(s, 0) + 1
    max_share = max(sev_counter.values()) / max(1, parse_ok) if sev_counter else 1.0

    ct = co = 0
    for d in details:
        p = d["pred"]
        if not isinstance(p, dict):
            continue
        sev = str(p.get("severity", "")).strip().lower()
        det = p.get("detected")
        if isinstance(det, bool) and sev:
            ct += 1
            if det == (sev not in nc):
                co += 1
    consist = co / ct if ct else 0.0

    in_range = num_total = num_missing = 0
    pairs = []
    if nf:
        gv = [d["gold"].get(nf) for d in details
              if isinstance(d["gold"], dict) and isinstance(d["gold"].get(nf), (int, float))]
        lo, hi = (min(gv), max(gv)) if gv else (None, None)
        pad = 0.3 * (hi - lo) if gv else 0
        for d in details:
            g, p = d["gold"], d["pred"]
            if not isinstance(g, dict) or not isinstance(p, dict):
                continue
            a, b = g.get(nf), p.get(nf)
            if not isinstance(a, (int, float)):
                continue
            if not isinstance(b, (int, float)):
                num_missing += 1      # 数值字段没答出来，单独统计，不算越界
                continue
            num_total += 1
            if lo is not None and lo - pad <= b <= hi + pad:
                in_range += 1
            pairs.append((a, b))
    in_range_rate = in_range / num_total if num_total else 0.0
    r = None
    if len(pairs) >= 5:
        import numpy as np
        a = np.array([x[0] for x in pairs], dtype="float64")
        b = np.array([x[1] for x in pairs], dtype="float64")
        if a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])

    m = metrics(name, details) or {}
    verdict = []
    if parse_ok / n < 0.98:
        verdict.append("解析率<98%")
    if schema_ok / n < 0.95:
        verdict.append("字段不全(%.0f%%)" % (100 * schema_ok / n))
    if max_share > 0.85:
        verdict.append("塌缩")
    neg_gold = sum(1 for g in golds if isinstance(g, dict) and str(g.get("severity", "")).strip().lower() in nc)
    if neg_gold >= 3 and ct and consist < 0.90:
        verdict.append("det/sev不自洽")
    if nf and (num_total + num_missing) and num_missing / (num_total + num_missing) > 0.05:
        verdict.append("数值缺失(%.0f%%)" % (100 * num_missing / (num_total + num_missing)))
    if num_total and in_range_rate < 0.90:
        verdict.append("数值越界")
    if nf and r is not None and r < 0.3:
        verdict.append("数值不相关")
    if (m.get("det") or 0) < 0.70:
        verdict.append("det<70%")

    return {
        "name": name, "n": n, "parse": parse_ok / n, "schema": schema_ok / n,
        "max_share": max_share, "consist": consist, "in_range": in_range_rate, "r": r,
        "det": m.get("det"), "sev": m.get("sev"), "mre": m.get("mre"),
        "numeric": nf, "none_class": sorted(nc), "bad": bad, "verdict": verdict or ["PASS"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--elements", default="")
    a = ap.parse_args()

    els = a.elements.split(",") if a.elements else adopted_elements()
    print("审计 %d 个元素，每个 %d 张" % (len(els), a.limit))

    results = []
    for i, e in enumerate(els, 1):
        print("\n[%d/%d] %s" % (i, len(els), e), flush=True)
        try:
            r = audit_element(e, a.limit, a.batch)
        except Exception as ex:
            r = {"name": e, "error": str(ex)[:160]}
        print("   ", r, flush=True)
        results.append(r)
        json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    lines = ["# 质量审计 v2（%d 个采用元素，每元素 %d 张）" % (len(els), a.limit), "",
             "> 判定：解析≥98%、字段（按 prompt）≥95%、不塌缩（单类≤85%）、det/sev 自洽≥90%、数值在 gold 30% 缓冲内、数值相关 r≥0.3、det≥70%", "",
             "| 元素 | n | det | sev | MRE | 解析 | 字段 | 单类 | 自洽 | 数值r | 判定 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        if "error" in r:
            lines.append("| %s | - | - | - | - | - | - | - | - | - | ✗ %s |" % (r["name"], r["error"]))
            continue
        lines.append("| %s | %d | %s | %s | %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %s | %s |" % (
            r["name"], r["n"],
            "%.1f%%" % (r["det"] * 100) if r.get("det") is not None else "-",
            "%.1f%%" % (r["sev"] * 100) if r.get("sev") is not None else "-",
            "%.1f%%" % (r["mre"] * 100) if r.get("mre") is not None else "-",
            r["parse"] * 100, r["schema"] * 100, r["max_share"] * 100, r["consist"] * 100,
            "%.2f" % r["r"] if r.get("r") is not None else "-",
            "✅" if r["verdict"] == ["PASS"] else "⚠️ " + "、".join(r["verdict"]),
        ))
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n报告:", OUT_MD)


if __name__ == "__main__":
    main()

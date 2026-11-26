# -*- coding: utf-8 -*-
r"""
单独评估指定元素（修 adapter 命名冲突）

## 问题

train_bill.py 一次加载多个 adapter 时，名字里带连字符（snow_fsc-json）
会让 peft 的 delete_adapter 处理不干净，导致后面的加载失败。

单独评估时只加载一个 adapter，不换，就不会冲突。

## 用法

  cd C:\vlmdisaster
  .\venv\Scripts\python.exe scripts\eval_single.py hurricane_opt,volcano_ndvi,gpm,hurricane_rain
"""

import json
import os
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r'C:\vlmdisaster\scripts')
from batch_to_json import SPECS

MODEL = r"C:\vlmdisaster\models\Qwen3-VL-4B\models\Qwen--Qwen3-VL-4B-Instruct\snapshots\master"
OUT_ROOT = r"C:\vlmdisaster\outputs"
LF_DATA = r"C:\vlmdisaster\LLaMA-Factory\data"
INFO = os.path.join(LF_DATA, "dataset_info.json")


def parse_json(t):
    if not t:
        return None
    t = str(t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            pass
    # 2026-11-20：模型偶尔会把同一段 JSON 输出两遍（或前后带解释文字），
    # 逐个尝试「第一个平衡花括号块」，取第一个能解析的对象。
    start = t.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        for k in range(start, len(t)):
            ch = t[k]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:k + 1])
                    except Exception:
                        break
        start = t.find("{", start + 1)
    return None


def rel_err(g, p):
    try:
        g, p = float(g), float(p)
    except (TypeError, ValueError):
        return None
    return abs(p - g) / abs(g) if abs(g) > 1e-9 else abs(p - g)


def extract(rec):
    q = g = None
    for t in rec.get("conversations") or []:
        if t.get("from") == "human" and q is None:
            q = t.get("value", "")
        elif t.get("from") == "gpt" and g is None:
            g = t.get("value", "")
    img = rec.get("images") or rec.get("image")
    if isinstance(img, list):
        img = img[0] if img else None
    return q, g, img


def eval_one(name, limit=80):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    info = json.load(open(INFO, encoding="utf-8"))
    vk = "disaster_%s_json_val" % name
    if vk not in info:
        print("  ✗ %s 验证集未注册" % name)
        return None
    recs = json.load(open(os.path.join(LF_DATA, info[vk]["file_name"]), encoding="utf-8"))
    if len(recs) > limit:
        random.seed(42)
        recs = random.sample(recs, limit)
    rows = [extract(r) for r in recs]
    rows = [(a, b, c) for a, b, c in rows if a and b and c]
    print("  验证集 %d 条" % len(rows), flush=True)

    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qc, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    lora = os.path.join(OUT_ROOT, "qwen3vl-lora-%s-json" % name)
    # 固定 adapter 名，避免连字符
    model = PeftModel.from_pretrained(base, lora, adapter_name="eval_adapter")
    model.eval()

    details = []
    for j, (q, g, img) in enumerate(rows, 1):
        if not os.path.exists(img):
            continue
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": "file://%s" % img},
            {"type": "text", "text": q.replace("<image>", "").strip()}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = proc(text=[text], images=imgs, videos=vids,
                   return_tensors="pt").to(base.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=320, do_sample=False,
                                 repetition_penalty=1.05)
        pt = proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
        details.append({"gold": parse_json(g), "pred": parse_json(pt)})
        if j % 20 == 0:
            print("    %d/%d" % (j, len(rows)), flush=True)

    del model
    del base
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return metrics(name, details)


def metrics(name, details):
    """从 details([{gold, pred}]) 算指标。eval_batch.py 复用本函数，口径必须一致。"""
    n = len(details)
    tp = fp = tn = fn = 0
    for d in details:
        if not d["gold"] or not d["pred"]:
            continue
        gv, pv = d["gold"].get("detected"), d["pred"].get("detected")
        if gv is None or pv is None:
            continue
        if gv and pv:
            tp += 1
        elif pv and not gv:
            fp += 1
        elif not gv and not pv:
            tn += 1
        else:
            fn += 1
    tot = tp + fp + tn + fn
    det = (tp + tn) / tot if tot else None

    sev_hit = sev_tot = 0
    for d in details:
        if not d["gold"] or not d["pred"]:
            continue
        gv, pv = d["gold"].get("severity"), d["pred"].get("severity")
        if gv is None or pv is None:
            continue
        sev_tot += 1
        if str(gv).lower() == str(pv).lower():
            sev_hit += 1
    sev = sev_hit / sev_tot if sev_tot else None

    # 数值
    spec = next((s for s in SPECS if s["out"] == name), None)
    mre = None
    num_key = None
    if spec and spec["nums"]:
        num_key = spec["nums"][0][1]
    else:
        # 不在 SPECS 里的元素（新增的、手工构建的），从 gold 里取第一个数值字段，
        # 否则 MRE 永远是 None，等于放弃了防塌缩的唯一量化证据。
        for d in details:
            if not d["gold"]:
                continue
            for k, v in d["gold"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    num_key = k
                    break
            if num_key:
                break
    if num_key:
        errs = []
        for d in details:
            if not d["gold"] or not d["pred"]:
                continue
            if num_key in d["gold"] and num_key in d["pred"]:
                e = rel_err(d["gold"][num_key], d["pred"][num_key])
                if e is not None:
                    errs.append(e)
        if errs:
            errs.sort()
            mre = errs[len(errs) // 2]

    return {"name": name, "n": n, "det": det, "sev": sev, "mre": mre,
            "mre_field": num_key,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else []
    if not names:
        print("用法: python eval_single.py name1,name2")
        return

    results = []
    for nm in names:
        nm = nm.strip()
        print("=" * 70)
        print("  %s" % nm)
        print("=" * 70, flush=True)
        r = eval_one(nm)
        if r:
            results.append(r)
            print("  格式 %d 条  detected %s  TP/FN %d/%d  severity %s  数值MRE %s"
                  % (r["n"],
                     "%.1f%%" % (r["det"] * 100) if r["det"] is not None else "-",
                     r["tp"], r["fn"],
                     "%.1f%%" % (r["sev"] * 100) if r["sev"] is not None else "-",
                     "%.1f%%" % (r["mre"] * 100) if r["mre"] is not None else "-"),
                  flush=True)

    print("\n" + "=" * 70)
    print("%-20s %8s %8s %10s %s" % ("元素", "detected", "severity", "数值MRE", "TP/FN"))
    print("-" * 70)
    for r in results:
        print("%-20s %7s %7s %9s %d/%d"
              % (r["name"],
                 "%.1f%%" % (r["det"] * 100) if r["det"] is not None else "-",
                 "%.1f%%" % (r["sev"] * 100) if r["sev"] is not None else "-",
                 "%.1f%%" % (r["mre"] * 100) if r["mre"] is not None else "-",
                 r["tp"], r["fn"]))

    out = os.path.join(OUT_ROOT, "eval_single_result.json")
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n已写 %s" % out)


if __name__ == "__main__":
    main()

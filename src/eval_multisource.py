# -*- coding: utf-8 -*-
r"""
三源融合 LoRA 验收脚本（SAR / 热红外 / 微波）。

用途：训练完，用 val 集逐条推理，比对答案，按**源分组**统计准确率。
      判断 loss 低是否过拟合（val 准不高 → 过拟合/记忆）。

数据：LLaMA-Factory/data/disaster_multisource_val.json（sharegpt 格式，images 绝对路径）
模型：Qwen3-VL-4B + LoRA adapter

用法：
  python src/eval_multisource.py --lora outputs/qwen3vl-multisource-sar-tir-mw --limit 60
  python src/eval_multisource.py --lora outputs/qwen3vl-multisource-sar-tir-mw
"""
import os
import re
import sys
import json
import argparse

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import PeftModel

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_MODEL = "C:/vlmdisaster/models/Qwen3-VL-4B/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
VAL_JSON = r"C:\vlmdisaster\LLaMA-Factory\data\disaster_multisource_val.json"

# 元素目录已改名（加 `)` 与 `★**` 前缀，用于演示定位），但 val json 里的旧路径没跟着改。
# 这里做旧→新路径重映射，保证验收脚本在新目录结构下仍能读到图。
_PATH_REMAP = [
    ("datasets\\03_thermal_infrared\\wildfire\\", "datasets\\03_thermal_infrared)\\★热红外_野火\\"),
    ("datasets\\04_passive_microwave\\drought\\", "datasets\\04_passive_microwave)\\★微波_土壤湿度\\"),
    ("datasets\\05_sar_insar\\earthquake_deformation\\", "datasets\\05_sar_insar)\\★SAR_地震形变\\"),
]


def remap_path(p):
    if not p:
        return p
    if os.path.exists(p):
        return p
    for old, new in _PATH_REMAP:
        if old in p:
            cand = p.replace(old, new)
            if os.path.exists(cand):
                return cand
            return cand  # 即便不存在也返回，报错时路径更可读
    return p


def classify_source(prompt, path):
    """按 prompt/图片特征归源"""
    p = (prompt or "").lower() + " " + (path or "").lower()
    if "insar" in p or "deformation" in p or "sar" in p:
        return "SAR"
    if "wildfire" in p or "fire" in p or "hotspot" in p:
        return "TIR"
    if "soil moisture" in p or "smap" in p:
        return "MW"
    return "OTHER"


def normalize(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def is_correct(pred, gold):
    """宽松判对：金标核心词出现在预测里。"""
    pred = normalize(pred); gold = normalize(gold)
    # 二分类/多分类的核心判别词
    keys = []
    for kw in ["yes", "no", "obvious", "slight", "stable", "cold wave", "frost",
               "heatwave", "normal", "dry", "wet", "drought"]:
        if kw in gold:
            keys.append(kw)
    if not keys:
        return None  # 无法判定
    # yes/no 需特别处理：出现在开头语气
    if "yes" in keys and "no" in keys:
        # gold 同时含两者时按主谓
        pass
    hits = 0
    # 对否定的 "no" 需避免被 "not/no significant" 误配
    for k in keys:
        if k == "no":
            if re.search(r"\bno\b", pred):
                hits += 1
        elif k == "yes":
            if re.search(r"\byes\b", pred):
                hits += 1
        else:
            if k in pred:
                hits += 1
    return hits >= len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 条（0=全部）")
    ap.add_argument("--val", default=VAL_JSON)
    args = ap.parse_args()

    val = json.load(open(args.val, encoding="utf-8"))
    if args.limit:
        val = val[:args.limit]
    print(f"val 条数: {len(val)}")

    print("加载模型...")
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.lora)
    model = model.to("cuda:0")
    model.eval()
    # 强制所有参数确在 GPU（防 peft 后 CPU 静默卸载）
    devs = {p.device.type for p in model.parameters()}
    print(f"模型所在设备: {devs}")
    if "cuda" not in devs:
        print("警告：模型未上 GPU！")
    proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    stats = {}   # src -> [correct, total, undecidable]
    detail = []
    for i, e in enumerate(val, 1):
        conv = e["conversations"]
        user = next((c["value"] for c in conv if c["from"] == "human"), "")
        gold = next((c["value"] for c in conv if c["from"] == "gpt"), "")
        img = remap_path(e["images"][0])
        prompt = user.replace("<image>", "").strip()
        src = classify_source(user, img)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = proc(text=[text], images=image_inputs, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        pred = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        ok = is_correct(pred, gold)
        s = stats.setdefault(src, [0, 0, 0])
        if ok is None:
            s[2] += 1
        else:
            s[1] += 1
            s[0] += 1 if ok else 0
        s_total = sum(x[1] for x in stats.values())
        print(f"[{i}/{len(val)}] {src:4s} {'OK ' if ok else 'X  ' if ok is not None else '?  '} "
              f"pred={pred.strip()[:60]}", flush=True)
        detail.append({"src": src, "correct": ok, "pred": pred, "gold": gold, "img": img})

    print("\n" + "=" * 60)
    print("  分源准确率")
    print("=" * 60)
    tot_c = tot_t = 0
    for src, (c, t, u) in sorted(stats.items()):
        acc = c / t * 100 if t else 0
        print(f"  {src:6s}: {c}/{t} = {acc:.1f}%   (无法判定 {u})")
        tot_c += c; tot_t += t
    if tot_t:
        print(f"  合计  : {tot_c}/{tot_t} = {tot_c/tot_t*100:.1f}%")

    out = args.lora.rstrip("/\\") + "_eval.json"
    json.dump(detail, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n明细: {out}")


if __name__ == "__main__":
    main()

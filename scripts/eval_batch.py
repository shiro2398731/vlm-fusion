# -*- coding: utf-8 -*-
r"""
批量评估（一次喂多张图），口径与 eval_single.py 完全一致

## 为什么不直接用 eval_single.py

eval_single.py 一张一张喂，实测约 2 分钟/张，80 张要 2.5 小时。
18 个元素要 45 小时，不可接受。

本脚本只改一件事：把样本按 batch_size 打包生成，后端仍是一个基座 + 一个 adapter，
贪心解码，结果与逐张推理一致。指标计算直接 import eval_single.metrics，保证口径不漂。

## 用法

  cd C:\vlmdisaster
  .\venv\Scripts\python.exe scripts\eval_batch.py marine_heatwave --limit 80 --batch 6

## 注意

- 结果按元素名合并写入 outputs\eval_batch_result.json（每跑完一个元素就落盘）
- 不覆盖 eval_single_result.json
- 训练和评估不能同时跑，8 GB 显存会 OOM
"""

import argparse
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\vlmdisaster\scripts")

from eval_single import (INFO, LF_DATA, MODEL, OUT_ROOT, extract,  # noqa: E402
                         metrics, parse_json)

OUT_JSON = os.path.join(OUT_ROOT, "eval_batch_result.json")


def load_val(name, limit):
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
    rows = [(a, b, c) for a, b, c in rows if a and b and c and os.path.exists(c)]
    return rows


def crop_rows(rows, top, tag, bottom=0):
    """裁掉图片边缘若干行。top>0 去掉顶部（标题/标注），bottom>0 用作对照组。"""
    import tempfile
    from PIL import Image
    out_dir = os.path.join(tempfile.gettempdir(), "ablate_%s" % (tag or "crop"))
    os.makedirs(out_dir, exist_ok=True)
    new = []
    for q, g, img in rows:
        dst = os.path.join(out_dir, os.path.basename(img))
        if not os.path.exists(dst):
            im = Image.open(img)
            w, h = im.size
            im.crop((0, top, w, h - bottom)).save(dst)
        new.append((q, g, dst))
    return new


def eval_one(name, limit, batch_size, crop_top=0, tag="", crop_bottom=0):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    rows = load_val(name, limit)
    if crop_top or crop_bottom:
        rows = crop_rows(rows, crop_top, tag, crop_bottom)
        print("  已裁剪：顶部 %d 行 / 底部 %d 行" % (crop_top, crop_bottom), flush=True)
    if not rows:
        return None
    print("  验证集 %d 条，batch=%d" % (len(rows), batch_size), flush=True)

    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qc, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    # 批量生成必须左填充，否则解码起点对不齐
    proc.tokenizer.padding_side = "left"

    lora = os.path.join(OUT_ROOT, "qwen3vl-lora-%s-json" % name)
    model = PeftModel.from_pretrained(base, lora, adapter_name="eval_adapter")
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

        inp = proc(text=texts, images=images, return_tensors="pt",
                   padding=True).to(base.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=320, do_sample=False,
                                 repetition_penalty=1.05)
        new_tokens = out[:, inp["input_ids"].shape[1]:]
        decoded = proc.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for (q, g, img), pt in zip(chunk, decoded):
            details.append({"gold": parse_json(g), "pred": parse_json(pt)})
        print("    %d/%d" % (min(start + batch_size, len(rows)), len(rows)), flush=True)

    del model
    del base
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return metrics(name, details)


def save(result):
    # 每天第一次写之前自动备份一次总账（同一天只备份一次，避免刷屏）
    import shutil
    import time
    if os.path.exists(OUT_JSON):
        bak = OUT_JSON + ".bak_" + time.strftime("%Y%m%d")
        if not os.path.exists(bak):
            try:
                shutil.copy2(OUT_JSON, bak)
            except Exception:
                pass
    data = []
    if os.path.exists(OUT_JSON):
        try:
            data = json.load(open(OUT_JSON, encoding="utf-8"))
        except Exception:
            data = []
    data = [d for d in data if d.get("name") != result["name"]]
    data.append(result)
    json.dump(data, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", type=str, help="逗号分隔，如 flare,iod")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--crop-top", type=int, default=0,
                    help="裁掉每张图顶部 N 行（消融：去掉标题里印的数值/标签）")
    ap.add_argument("--crop-bottom", type=int, default=0,
                    help="裁掉底部 N 行（对照组：验证掉落是否只是图像形变导致）")
    ap.add_argument("--tag", default="", help="结果名后缀，如 nocap（不覆盖原数字）")
    a = ap.parse_args()

    names = [x.strip() for x in a.names.split(",") if x.strip()]
    done = []
    for nm in names:
        print("=" * 70)
        print("  %s%s" % (nm, "（裁顶部 %d / 底部 %d）" % (a.crop_top, a.crop_bottom)
                          if (a.crop_top or a.crop_bottom) else ""))
        print("=" * 70, flush=True)
        try:
            r = eval_one(nm, a.limit, a.batch, a.crop_top, a.tag, a.crop_bottom)
        except RuntimeError as e:
            print("  ✗ 运行失败（可能是显存）: %s" % e, flush=True)
            break
        if not r:
            continue
        if a.tag:
            r["name"] = "%s@%s" % (r["name"], a.tag)
            r["base_name"] = nm
        save(r)
        done.append(r)
        print("  格式 %d 条  detected %s  TP/FN %d/%d  severity %s  数值MRE %s"
              % (r["n"],
                 "%.1f%%" % (r["det"] * 100) if r["det"] is not None else "-",
                 r["tp"], r["fn"],
                 "%.1f%%" % (r["sev"] * 100) if r["sev"] is not None else "-",
                 "%.1f%%" % (r["mre"] * 100) if r["mre"] is not None else "-"),
              flush=True)

    if done:
        print("\n" + "=" * 70)
        print("%-22s %9s %9s %9s %8s" % ("元素", "detected", "severity", "数值MRE", "TP/FN"))
        print("-" * 70)
        for r in done:
            print("%-22s %8s %8s %8s %d/%d"
                  % (r["name"],
                     "%.1f%%" % (r["det"] * 100) if r["det"] is not None else "-",
                     "%.1f%%" % (r["sev"] * 100) if r["sev"] is not None else "-",
                     "%.1f%%" % (r["mre"] * 100) if r["mre"] is not None else "-",
                     r["tp"], r["fn"]))
        print("\n已合并写入 %s" % OUT_JSON)


if __name__ == "__main__":
    main()

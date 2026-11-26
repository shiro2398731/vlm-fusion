# -*- coding: utf-8 -*-
r"""
安全重训：先删 output 目录，训完自动验收

## 为什么需要它

2026-11-13 踩过：LLaMA-Factory 会自动从 output_dir 找最新 checkpoint 续训。
旧 checkpoint 已到终点 -> 加载后立即 "Training completed" ->
把**一模一样的旧权重**重新保存一遍，文件时间戳变成当天，看着像重训过。

判断依据只有一个：`train_loss` 是否为 0。为 0 就是没训。

## 本脚本做的事

1. 删掉整个 outputs\qwen3vl-lora-<元素>-json\（含 checkpoint，这是根因）
2. 跑 src\train.py <yaml>
3. 只读**本次新增**的日志片段，检查：
   - 是否出现 "Resuming training from checkpoint"（出现即说明没删干净）
   - train_loss 是否为 0 / 缺失
4. 逐元素给出 ✅/❌，最后汇总

## 用法

  # 先看会做什么，不动手
  .\venv\Scripts\python.exe scripts\retrain_safe.py --elements lst,flare --dry-run

  # 真跑
  .\venv\Scripts\python.exe scripts\retrain_safe.py --elements lst,flare,hurricane_rain,volcano_ndvi,drought_tir,wildfire

  # 只验收，不训练
  .\venv\Scripts\python.exe scripts\retrain_safe.py --elements lst --verify-only
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\vlmdisaster"
LF = os.path.join(ROOT, "LLaMA-Factory")
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
YAML_DIR = os.path.join(LF, "examples", "train_qlora")
OUT_ROOT = os.path.join(ROOT, "outputs")
LOGDIR = os.path.join(OUT_ROOT, "train_logs")
INFO = os.path.join(LF, "data", "dataset_info.json")


def paths(el):
    return dict(
        yaml=os.path.join(YAML_DIR, "auto_%s_json.yaml" % el),
        outdir=os.path.join(OUT_ROOT, "qwen3vl-lora-%s-json" % el),
        log=os.path.join(LOGDIR, "json_%s.log" % el),
    )


def registered(el):
    info = json.load(open(INFO, encoding="utf-8"))
    return ("disaster_%s_json" % el) in info and ("disaster_%s_json_val" % el) in info


def last_run_text(path):
    """只取最后一次运行的日志片段。

    日志是追加写的（train_json_queue / 本脚本 / LLaMA-Factory 都往里追加），
    直接扫全文会把历史上那次 0 步空跑的 "Resuming" 误判成这一次，
    所以按分隔行切段，只看最后一段。
    """
    if not os.path.exists(path):
        return ""
    text = open(path, encoding="utf-8", errors="replace").read()
    parts = re.split(r"(?m)^=====.*$", text)
    return parts[-1] if parts else text


def verify(el, since=None):
    """检查本元素最近一次运行是否真的训过（只看最后一段日志）"""
    p = paths(el)
    adapter = os.path.join(p["outdir"], "adapter_model.safetensors")
    if not os.path.exists(adapter):
        return False, "没有 adapter 文件"
    if not os.path.exists(p["log"]):
        return False, "没有训练日志"
    text = last_run_text(p["log"])
    if not text.strip():
        return False, "日志最后一段是空的（本次没产生输出）"
    if "Resuming training from checkpoint" in text:
        return False, "出现续训（checkpoint 没删干净）"
    losses = re.findall(r"train_loss\s*=\s*([0-9.eE+-]+)", text)
    if not losses:
        return False, "日志里找不到 train_loss"
    last = losses[-1]
    if float(last) == 0.0:
        return False, "train_loss = 0（0 步空跑）"
    return True, "train_loss = %s" % last


def run_one(el, dry=False):
    p = paths(el)
    if not os.path.exists(p["yaml"]):
        return False, "缺 yaml: %s" % p["yaml"]
    if not registered(el):
        return False, "数据集未注册（disaster_%s_json / _val）" % el

    print("\n>>> %s" % el, flush=True)
    if os.path.exists(p["outdir"]):
        n_ckpt = len([d for d in os.listdir(p["outdir"]) if d.startswith("checkpoint-")])
        print("    删除旧目录（含 %d 个 checkpoint）" % n_ckpt, flush=True)
        if not dry:
            shutil.rmtree(p["outdir"], ignore_errors=True)
    else:
        print("    无旧目录", flush=True)

    if dry:
        print("    [dry] 将执行: %s src/train.py %s" % (PY, os.path.basename(p["yaml"])))
        return True, "dry-run"

    os.makedirs(LOGDIR, exist_ok=True)
    t0 = time.time()
    with open(p["log"], "a", encoding="utf-8", errors="replace") as logf:
        logf.write("\n===== retrain_safe %s =====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        logf.flush()
        proc = subprocess.run([PY, os.path.join("src", "train.py"), p["yaml"]],
                              cwd=LF, stdout=logf, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    ok, why = verify(el)
    print("    %.0f 分钟，退出码 %d，%s" % (dt, proc.returncode, why), flush=True)
    return ok, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", required=True, help="逗号分隔")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    els = [x.strip() for x in a.elements.split(",") if x.strip()]
    print("=" * 74)
    print("  安全重训：%d 个元素%s" % (len(els), "（dry-run）" if a.dry_run else ""))
    print("=" * 74)

    if a.verify_only:
        for el in els:
            ok, why = verify(el)
            print("  %s %-20s %s" % ("✅" if ok else "❌", el, why))
        return 0

    results = []
    for el in els:
        ok, why = run_one(el, a.dry_run)
        results.append((el, ok, why))

    print("\n" + "=" * 74)
    print("  汇总")
    print("=" * 74)
    for el, ok, why in results:
        print("  %s %-20s %s" % ("✅" if ok else "❌", el, why))
    bad = [el for el, ok, _ in results if not ok]
    if bad and not a.dry_run:
        print("\n  ⚠️ 这些元素没训好，别拿去评估：%s" % ", ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

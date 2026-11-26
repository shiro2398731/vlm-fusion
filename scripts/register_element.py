# -*- coding: utf-8 -*-
r"""
新元素一键注册：把渲染好的 train/_val JSON 变成「可以开训」的状态。

做四件事（都可 --dry-run 预览）：
  1. 复制 train_<el>_json.json / _val.json 到 LLaMA-Factory/data
  2. 备份 dataset_info.json，注册 disaster_<el>_json / _val 两个 key
  3. 由模板生成 auto_<el>_json.yaml（改 dataset / output_dir）
  4. 打印下一步命令（preflight → retrain）

用法：
  .\venv\Scripts\python.exe scripts\register_element.py --element wind_gale ^
      --train datasets/07_scatterometer/wind_gale/train_wind_gale_json.json ^
      --val   datasets/07_scatterometer/wind_gale/train_wind_gale_json_val.json --dry-run
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\vlmdisaster"
LF = os.path.join(ROOT, "LLaMA-Factory")
DATA = os.path.join(LF, "data")
INFO = os.path.join(DATA, "dataset_info.json")
YAML_DIR = os.path.join(LF, "examples", "train_qlora")
DEFAULT_TEMPLATE = os.path.join(YAML_DIR, "auto_sea_ice_amsr2_json.yaml")


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--element", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="已注册时覆盖（会备份）")
    a = ap.parse_args()
    el = a.element

    train = json.load(open(a.train, encoding="utf-8"))
    val = json.load(open(a.val, encoding="utf-8"))
    print("train %d / val %d" % (len(train), len(val)))
    missing = 0
    for r in train + val:
        v = r.get("image") or r.get("images")
        if isinstance(v, list):
            v = v[0] if v else None
        if not v or not os.path.exists(v):
            missing += 1
    if missing:
        print("✗ 图片缺失 %d 张，先修数据" % missing)
        return 1

    info = json.load(open(INFO, encoding="utf-8"))
    k1, k2 = "disaster_%s_json" % el, "disaster_%s_json_val" % el
    if (k1 in info or k2 in info) and not a.force:
        print("✗ %s 已注册（要覆盖加 --force，会先备份）" % el)
        return 1

    tname, vname = "train_%s_json.json" % el, "train_%s_json_val.json" % el
    yname = "auto_%s_json.yaml" % el
    target_yaml = os.path.join(YAML_DIR, yname)

    print("\n将执行：")
    print("  copy -> %s" % os.path.join(DATA, tname))
    print("  copy -> %s" % os.path.join(DATA, vname))
    print("  register %s / %s" % (k1, k2))
    print("  yaml    -> %s" % target_yaml)
    if a.dry_run:
        print("\n[dry-run] 未写盘")
        return 0

    for src, dst in ((a.train, os.path.join(DATA, tname)), (a.val, os.path.join(DATA, vname))):
        if os.path.exists(dst):
            shutil.copy2(dst, dst + ".bak_" + stamp())
        shutil.copy2(src, dst)

    shutil.copy2(INFO, INFO + ".bak_" + stamp())
    ent = {"file_name": tname, "formatting": "sharegpt",
           "columns": {"messages": "conversations", "images": "image"}}
    vent = dict(ent)
    vent["file_name"] = vname
    info[k1] = ent
    info[k2] = vent
    json.dump(info, open(INFO, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    if os.path.exists(target_yaml):
        shutil.copy2(target_yaml, target_yaml + ".bak_" + stamp())
    tpl = open(a.template, encoding="utf-8").read()
    # 模板里的 dataset / eval / output 行替换成新元素
    lines = []
    for line in tpl.splitlines():
        s = line.strip()
        if s.startswith("dataset:"):
            line = "dataset: %s" % k1
        elif s.startswith("eval_dataset:"):
            line = "eval_dataset: %s" % k2
        elif s.startswith("output_dir:"):
            line = "output_dir: C:/vlmdisaster/outputs/qwen3vl-lora-%s-json" % el
        lines.append(line)
    open(target_yaml, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print("\n✅ 注册完成。下一步：")
    print("  .\\venv\\Scripts\\python.exe scripts\\preflight_check.py --element %s" % el)
    print("  .\\venv\\Scripts\\python.exe scripts\\retrain_safe.py --elements %s" % el)
    return 0


if __name__ == "__main__":
    sys.exit(main())

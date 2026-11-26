# -*- coding: utf-8 -*-
r"""
CMEMS 风场 · 切窗 + 结构化 JSON（新元素：wind_gale_json）

与旧版 wind_gale_render.py 的区别
--------------------------------
旧版：答案写成自然语言（"This region has gale-force winds..."）
本版：结构化 JSON，带窗口最大风速（m/s）——防塌缩的读数证据；只出一张速度图。

物理链路（论文引用）
------------------------
  CMEMS L4 融合风场（0.125°，小时；融合散射计 + 辐射计 + 模式）
    → 10 m 风速 = sqrt(u²+v²)（m/s）
  判据：蒲福风级——>=17.2 m/s 为大风（8 级），>=24.5 m/s 为狂风（10 级）。
  注：原始散射计在强风下会饱和（~28 m/s 封顶），本产品为 L4 融合，链路卡需注明。

输出
----
  datasets/07_scatterometer/wind_gale/images/*.png
  datasets/07_scatterometer/wind_gale/train_wind_gale_json.json / _val.json

用法
----
  .\venv\Scripts\python.exe src\preprocess\wind_gale_render_json.py --keep 2   # 先验
  .\venv\Scripts\python.exe src\preprocess\wind_gale_render_json.py            # 全量
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wind_gale_render as W  # noqa: E402  复用读数据/切窗/渲图

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "07_scatterometer", "wind")
OUT = os.path.join(ROOT, "datasets", "07_scatterometer", "wind_gale")
IMG = os.path.join(OUT, "images")

GALE = 17.2
STORM = 24.5
MAX_PER_CLASS = 8
OUT_SIZE = 384
SEED = 42

PROMPT = ("This is a CMEMS 10-m ocean wind-speed field (m/s) with wind vectors. "
          "Output a JSON object with fields: disaster_type, detected (true/false), "
          "source, physical_quantity, severity, max_wind_ms, criterion. "
          "severity must be one of these exact words: normal / gale / storm.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0)
    ap.add_argument("--grid", type=int, default=30)
    ap.add_argument("--include", default="*", help="只处理文件名匹配该 glob 的文件，如 *202501*")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(RAW, "*.nc")))
    if args.include != "*":
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), args.include)]
    if args.keep:
        files = files[:args.keep]
    print("待渲染:", len(files), "| 网格", args.grid, "deg")
    if not files:
        return

    recs = []
    cnt = collections.Counter()
    maxspeeds = []
    for i, f in enumerate(files):
        m = re.search(r"(\d{10})", os.path.basename(f))
        stamp = m.group(1) if m else str(i)
        try:
            u, v, lat, lon = W.load(f)
        except Exception as e:
            print("  %s 失败 %s" % (stamp, str(e)[:60]))
            continue
        spd = np.sqrt(u ** 2 + v ** 2)
        cells = W.window_cells(spd, lat, lon, args.grid)
        by = {}
        for mx, la, lo in cells:
            by.setdefault(W.grade_of(mx), []).append((mx, la, lo))

        picked = []
        for lab, lst in by.items():
            lst.sort(key=lambda t: -t[0])
            picked += [(lab, t) for t in lst[:MAX_PER_CLASS]]

        for lab, (mx, la, lo) in picked:
            cnt[lab] += 1
            maxspeeds.append(mx)
            tag = "%s%d_%s%d" % ("N" if la >= 0 else "S", abs(int(la)),
                                 "E" if lo >= 0 else "W", abs(int(lo)))
            png = os.path.join(IMG, "%s_%s_speed.png" % (stamp, tag))
            W.render_cell(u, v, lat, lon, la, lo, args.grid, png, stamp, lab, False)

            ans = {
                "disaster_type": "wind_gale",
                "detected": bool(mx >= GALE),
                "source": "SCAT",
                "physical_quantity": "wind_speed_10m",
                "severity": lab,
                "max_wind_ms": round(float(mx), 2),
                "criterion": ("Beaufort scale: gale >= 17.2 m/s (force 8), "
                              "storm >= 24.5 m/s (force 10)"),
            }
            recs.append({
                "image": png,
                "label": lab,
                "stamp": stamp,
                "max_wind_ms": round(float(mx), 2),
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        if i % 10 == 0 or i == len(files) - 1:
            print("  [%d/%d] %s  累计 %d" % (i + 1, len(files), stamp, len(recs)))

    random.seed(SEED)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["label"]].append(r)
    val, train = [], []
    for lab in sorted(by):
        lst = by[lab][:]
        random.shuffle(lst)
        n = min(max(10, int(round(len(lst) * 0.2))), 30, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_wind_gale_json.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_wind_gale_json_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("分级分布:", dict(cnt))
    print("val 分布:", dict(collections.Counter(r["label"] for r in val)))
    ms = np.array(maxspeeds)
    print("窗最大风速: min %.1f  中位 %.1f  P90 %.1f  max %.1f m/s"
          % (ms.min(), np.median(ms), np.percentile(ms, 90), ms.max()))
    print("输出:", OUT)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
台风（雷达降水源）渲染：把 33 天 GPM 数据切窗口出样本

## 与现有 496 条的关系

现有 `datasets\06_radar_precipitation\hurricane\` 的 496 条来自 2023-07-27~31
一个台风事件（杜苏芮），格式是：

    字段: image / date / win / label / product / max_mm / conversations
    答案: "This is a daily precipitation field from GPM IMERG.
           The maximum in this window is 44.1 mm/day."
    产品: intrinsic_precip（降水场）+ feature_rain_grade（分级场）

**新脚本严格沿用同一套格式、同一套判据、同一套窗口大小。**

## 判据（沿用现有数据的实际分布，2026-11-07 实测）

    normal             最大 22.4 ~  49.3 mm/day
    rainstorm          最大 53.9 ~  95.3 mm/day
    heavy_rainstorm    最大 117.6 ~ 233.7 mm/day
    extreme_rainstorm  最大 291.7 ~ 416.8 mm/day

阈值取相邻类中点：50 / 100 / 250 mm/day。

## 窗口

现有数据 win 坐标最大到 3072，说明用的是 0.1° 全球格网（3600×1800），
窗口边长约 256 像素 = 25.6°。这里沿用 WIN=256。

## 用法

  cd C:\vlmdisaster
  .\venv\Scripts\python.exe src\preprocess\gpm_hurricane_render.py --probe
  .\venv\Scripts\python.exe src\preprocess\gpm_hurricane_render.py

输出到 `datasets\06_radar_precipitation\hurricane_multi\`，
**不动现有的 hurricane 目录**。
"""

import argparse
import collections
import glob
import io
import json
import os
import random
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

RAW = r"C:\vlmdisaster\data\raw\06_radar_precipitation\gpm"
OUT = r"C:\vlmdisaster\datasets\06_radar_precipitation\hurricane_multi"
IMG = os.path.join(OUT, "images")

WIN = 256                 # 窗口边长（0.1° 格网像元）
STRIDE = 128              # 半重叠滑动
# 2026-11-08 改：原来按降水降序取前 16 个，导致 extreme 占 44%，
# 而现有 496 条的 extreme 只有 1.6%。改成按类分层抽样。
MAX_PER_DAY = 24          # 每天总窗口上限
DRY_PER_DAY = 10          # 其中 normal（无雨）的配额
OTHER_PER_DAY = 12        # rainstorm / heavy / extreme 共用的配额
OUT_SIZE = 384
RAIN_MIN = 0.3            # 窗内均值下限，低于算 dry

# 判据：与现有 496 条一致（阈值取相邻类区间中点）
GRADE = [
    ("normal", 0.0, 50.0),
    ("rainstorm", 50.0, 100.0),
    ("heavy_rainstorm", 100.0, 250.0),
    ("extreme_rainstorm", 250.0, 1e9),
]

PROMPT = ("This is a daily precipitation field from GPM IMERG. "
          "Describe the maximum rainfall in this window.")


def grade_of(mm):
    for name, lo, hi in GRADE:
        if lo <= mm < hi:
            return name
    return GRADE[-1][0]


def answer_of(max_mm):
    return ("This is a daily precipitation field from GPM IMERG. "
            "The maximum in this window is %.1f mm/day." % max_mm)


def load_day(date_str):
    """合并一天 48 个半小时片成日累积降水（mm）"""
    import h5py
    fs = sorted(glob.glob(os.path.join(RAW, "*3IMERG.%s-*.HDF5" % date_str)))
    if not fs:
        return None, None, None
    lat = lon = None
    total = None
    for f in fs:
        try:
            with h5py.File(f, "r") as h:
                if lat is None:
                    lat = h["Grid"]["lat"][:]
                    lon = h["Grid"]["lon"][:]
                a = h["Grid"]["precipitation"][0].astype("float64")
        except Exception:
            continue
        a[a < -1000] = 0.0
        a = a * 0.5          # mm/hr × 0.5h
        total = a if total is None else total + a
    if total is None:
        return None, None, None
    return total, lat, lon


def render(sub, path, date, max_mm, kind):
    vmax = max(1.0, float(np.percentile(sub, 99)))
    fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
    if kind == "precip":
        m = ax.pcolormesh(sub, cmap="Blues", vmin=0, vmax=vmax, shading="auto")
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("precipitation (mm/day)", fontsize=7)
    else:
        # 分级场：0/1/2/3 四档
        g = np.zeros_like(sub)
        for i, (_, lo, hi) in enumerate(GRADE):
            g[(sub >= lo) & (sub < hi)] = i
        m = ax.pcolormesh(g, cmap="YlOrRd", vmin=0, vmax=3, shading="auto")
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04,
                          ticks=[0, 1, 2, 3])
        cb.ax.set_yticklabels(["normal", "rainstorm", "heavy", "extreme"],
                              fontsize=5)
        cb.set_label("rainfall grade", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("GPM IMERG %s   max %.1f mm/day" % (date, max_mm), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=96)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE),
                                              PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--max-days", type=int, default=0)
    args = ap.parse_args()

    dates = sorted(set(
        re.search(r"3IMERG\.(\d{8})-", os.path.basename(f)).group(1)
        for f in glob.glob(os.path.join(RAW, "*.HDF5"))))
    if args.max_days:
        dates = dates[:args.max_days]
    print("日期 %d 个: %s ~ %s" % (len(dates), dates[0], dates[-1]))

    if not args.probe:
        os.makedirs(IMG, exist_ok=True)
        for f in glob.glob(os.path.join(IMG, "*.png")):
            os.remove(f)

    records = []
    cnt = collections.Counter()

    for i, d in enumerate(dates, 1):
        total, lat, lon = load_day(d)
        if total is None:
            print("  · %s 无有效数据" % d)
            continue
        H, W = total.shape

        wet, dry = [], []
        for y in range(0, H - WIN + 1, STRIDE):
            for x in range(0, W - WIN + 1, STRIDE):
                sub = total[y:y + WIN, x:x + WIN]
                mean_mm = float(sub.mean())
                item = (mean_mm, y, x, sub, grade_of(float(sub.max())))
                (wet if mean_mm >= RAIN_MIN else dry).append(item)

        # 分层：按类别分桶，每桶内部按降水量排序，配额分配
        buckets = collections.defaultdict(list)
        for it in wet:
            buckets[it[4]].append(it)
        for k in buckets:
            buckets[k].sort(key=lambda t: -t[0])

        random.seed(hash(d) & 0xFFFF)
        picked = list(random.sample(dry, min(len(dry), DRY_PER_DAY))) if dry else []

        # 有雨的三个类按接近 1:1:0.4 的比例分配
        # （对应现有数据 rainstorm 116 / heavy 114 / extreme 8 的实际比例）
        quota = {"rainstorm": 5, "heavy_rainstorm": 5, "extreme_rainstorm": 2}
        for lab, q in quota.items():
            lst = buckets.get(lab, [])
            if not lst:
                continue
            pick_n = min(q, len(lst))
            picked += random.sample(lst, pick_n)

        for mean_mm, y, x, sub, _lab in picked:
            max_mm = float(sub.max())
            lab = grade_of(max_mm)
            cnt[lab] += 1
            for suffix, kind, prod in (("_precip", "precip", "intrinsic_precip"),
                                       ("_grade", "grade", "feature_rain_grade")):
                fn = "gpm_%s_%d_%d%s.png" % (d, y, x, suffix)
                p = os.path.join(IMG, fn)
                if not args.probe:
                    render(sub, p, d, max_mm, kind)
                records.append({
                    "image": p,
                    "date": "%s-%s-%s" % (d[:4], d[4:6], d[6:8]),
                    "win": [y, x],
                    "label": lab,
                    "product": prod,
                    "max_mm": round(max_mm, 1),
                    "conversations": [
                        {"from": "human", "value": "<image>" + PROMPT},
                        {"from": "gpt", "value": answer_of(max_mm)},
                    ],
                })
        print("  [%2d/%d] %s  窗口 %d  %s"
              % (i, len(dates), d, len(picked), dict(cnt)), flush=True)

    print("\n=== 汇总 ===")
    print("样本 %d 条，标签 %s" % (len(records), dict(cnt)))
    if args.probe:
        print("（probe，未写图）")
        return
    if not records:
        print("无样本")
        return

    bad = [r["image"] for r in records if not os.path.exists(r["image"])]
    print("图像缺失 %d" % len(bad))
    if bad:
        return

    json.dump(records, open(os.path.join(OUT, "train_gpm_multi.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    print("输出:", OUT)


if __name__ == "__main__":
    main()

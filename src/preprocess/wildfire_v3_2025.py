# -*- coding: utf-8 -*-
r"""
用 2025 年 FIRMS 数据渲染野火测试集（真正的泛化验证）

## 为什么这样做

wildfire adapter 在 2024 年的 val 上拿了 100%，
但 val 是从训练集同一批里切的，同源同分布，证明不了泛化。

这份脚本用 **2025 全年**（模型完全没见过）渲染同样格式的图，
用来做真验证。

## 铁律：判据一个字都不能改

本脚本的参数全部复制自 `wildfire_v3.py`，**不调不改**：

| 参数 | 值 | 来源 |
|---|---|---|
| 火点过滤 | type==2 剔除、confidence=='l' 剔除 | FIRMS 官方字段定义 |
| slight | 5 ~ 30 个火点 | v3 阈值 |
| moderate | 30 ~ 100 | v3 阈值 |
| severe | >= 100 | v3 阈值 |
| 窗口 | 5° × 5°，40×40 像元 | v3 |
| 输出 | 384×384 | v3 |
| 每天采样 | 每档最多 3 个窗口 | v3 |
| 产品 | ti4（亮温 K）+ frp（火功率 MW） | v3 |

**改任何一个都会让验证失效**，因为那就不是"同一套判据下的新数据"了。

## 用法

  cd C:\vlmdisaster
  .\venv\Scripts\python.exe src\preprocess\wildfire_v3_2025.py --probe
  .\venv\Scripts\python.exe src\preprocess\wildfire_v3_2025.py

输出到 `datasets\03_thermal_infrared\wildfire_2025\`，**不动 2024 的任何文件**。
"""

import argparse
import collections
import csv
import glob
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
from PIL import Image

# ---- 全部复制自 wildfire_v3.py，不调 ----
RAW = r"C:\vlmdisaster\data\raw\03_thermal_infrared\firms\chn2025"
OUT = r"C:\vlmdisaster\datasets\03_thermal_infrared\wildfire_2025"
IMG = os.path.join(OUT, "images")

OUT_SIZE = 384
MAX_PER_DAY = 3
T_SLIGHT = 5
T_MODERATE = 30
T_SEVERE = 100
W, S, E, N = 73.0, 18.0, 135.0, 54.0
GRID_N = 40

PROMPT = ("This is a satellite thermal-infrared active-fire detection map over a region "
          "in China. Classify: no_active_fire or active_fire.")


def load_pts(fp):
    out = []
    with open(fp, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                t = int(row.get("type", "0") or 0)
                conf = (row.get("confidence") or "").strip().lower()
                if t == 2:
                    continue
                if conf == "l":
                    continue
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                ti4 = float(row["bright_ti4"])
                frp = float(row["frp"] or 0)
                out.append((lat, lon, ti4, frp))
            except Exception:
                continue
    return out


def band_of(n):
    if n < T_SLIGHT:
        return "none"
    if n < T_MODERATE:
        return "slight"
    if n < T_SEVERE:
        return "moderate"
    return "severe"


def pts_in(pts, la, lo, g, n=GRID_N):
    ti4 = np.zeros((n, n), dtype=float)
    frp = np.zeros((n, n), dtype=float)
    cnt = np.zeros((n, n), dtype=int)
    for lat, lon, t, f_ in pts:
        if not (la <= lat < la + g and lo <= lon < lo + g):
            continue
        iy = int((lat - la) / g * n)
        ix = int((lon - lo) / g * n)
        if not (0 <= iy < n and 0 <= ix < n):
            continue
        ti4[iy, ix] += t
        frp[iy, ix] += f_
        cnt[iy, ix] += 1
    m = np.where(cnt > 0, ti4 / np.maximum(cnt, 1), np.nan)
    return m, frp, cnt


def draw(field, cnt, la, lo, g, path, title, kind):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=100)
    if kind == "ti4":
        im = ax.imshow(field, cmap="inferno")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("brightness temp (K)", fontsize=7)
    else:
        im = ax.imshow(field, cmap="YlOrRd")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("FRP (MW)", fontsize=7)
    if cnt.sum() == 0:
        ax.text(0.5, 0.5, "No fire hotspot", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888")
    ax.set_xlabel("Lon", fontsize=7)
    ax.set_ylabel("Lat", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE),
                                           Image.BILINEAR).save(path)


def answer_text(lab, sev, n, frp_sum):
    if lab == "no_active_fire":
        return "No active fire hotspots detected in this region."
    return ("Active fire detected: %d hotspot pixels, severity %s, "
            "total FRP %.0f MW." % (n, sev, frp_sum))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=5.0)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    fs_ = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    if args.max_days:
        fs_ = fs_[:args.max_days]
    print("天数 %d | 窗 %.1f° | 数据源 %s" % (len(fs_), args.grid, RAW))
    if not fs_:
        print("没有 csv")
        return

    if args.probe:
        print("\n=== 探测：只统计，不写文件 ===")
        bands = collections.Counter()
        for i, f in enumerate(fs_, 1):
            pts = load_pts(f)
            if i % 90 == 0:
                print("  已读 %d/%d" % (i, len(fs_)), flush=True)
        print("可读文件 %d 个" % len(fs_))
        return

    os.makedirs(IMG, exist_ok=True)
    for f in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(f)

    g = args.grid
    lats = np.arange(S, N - g + 1e-9, g)
    lons = np.arange(W, E - g + 1e-9, g)

    recs = []
    cnt = collections.Counter()
    window_hits = collections.Counter()

    for i, f in enumerate(fs_, 1):
        stamp = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        stamp = stamp.group(1) if stamp else str(i)
        pts = load_pts(f)

        by_band = collections.defaultdict(list)
        for la in lats:
            for lo in lons:
                ti4m, frp, c = pts_in(pts, la, lo, g)
                nn = int(c.sum())
                b = band_of(nn)
                score = nn if b != "none" else 0
                by_band[b].append((score, la, lo, ti4m, frp, c, nn))

        picked = []
        for b in ("severe", "moderate", "slight"):
            lst = sorted(by_band.get(b, []), key=lambda t: -t[0])
            picked += [(b, t) for t in lst[:MAX_PER_DAY]]
        lst = by_band.get("none", [])
        random.shuffle(lst)
        picked += [("none", t) for t in lst[:MAX_PER_DAY]]

        for b, (_, la, lo, ti4m, frp, c, nn) in picked:
            lab = "no_active_fire" if b == "none" else "active_fire"
            cnt[lab] += 1
            window_hits[(round(float(la), 3), round(float(lo), 3))] += 1
            tag = "%s%d%s%d" % ("N" if la >= 0 else "S", abs(int(la)),
                                "E" if lo >= 0 else "W", abs(int(lo)))
            frp_sum = float(frp.sum())
            for sfx, kind, prod in (("_ti4", "ti4", "intrinsic_brightness_temp"),
                                    ("_frp", "frp", "feature_frp")):
                png = os.path.join(IMG, "%s_%s_%s%s.png" % (stamp, tag, lab, sfx))
                draw(ti4m, c, la, lo, g, png,
                     "%s  %s  %s" % (stamp, lab, b), kind)
                recs.append({
                    "conversations": [
                        {"from": "human", "value": "<image>" + PROMPT},
                        {"from": "gpt", "value": answer_text(lab, b, nn, frp_sum)},
                    ],
                    "images": [png],
                    "product": prod,
                    "label": lab,
                    "severity": b,
                    "hotspots": nn,
                    "frp_total": round(frp_sum, 1),
                    "stamp": stamp,
                    "window": [round(float(la), 3), round(float(lo), 3)],
                })
        if i % 30 == 0:
            print("  %d/%d %s" % (i, len(fs_), dict(cnt)), flush=True)

    # 自检（与 v3 完全一致）
    print("\n=== 自检 ===")
    ok = True
    if not recs:
        print("❌ 无样本")
        return
    lab_cnt = collections.Counter(r["label"] for r in recs)
    sev_cnt = collections.Counter(r["severity"] for r in recs)
    print("标签:", dict(lab_cnt))
    print("严重度:", dict(sev_cnt))
    if lab_cnt.get("active_fire", 0) == 0:
        print("❌ 没有 active_fire")
        ok = False
    if lab_cnt.get("no_active_fire", 0) == 0:
        print("❌ 没有 no_active_fire")
        ok = False
    bad_sev = 0
    for r in recs:
        ans = r["conversations"][1]["value"]
        if "severity" in ans and r["severity"] not in ans:
            bad_sev += 1
    print("答案 severity 写错:", bad_sev)
    if bad_sev:
        ok = False
    missing = [r["images"][0] for r in recs
               if not os.path.exists(r["images"][0])]
    print("图像缺失:", len(missing))
    if missing:
        ok = False
    if not ok:
        print("\n自检未通过，不写 json。")
        return

    json.dump(recs, open(os.path.join(OUT, "train_wildfire_2025.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n=== 完成 ===")
    print("样本 %d 条" % len(recs))
    print("输出:", OUT)
    print("\n下一步：用这份数据测 adapter，见 scripts\\eval_wildfire_lora.py --data 2025")


if __name__ == "__main__":
    main()

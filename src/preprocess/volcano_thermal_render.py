# -*- coding: utf-8 -*-
r"""
MODVOLC 火山热异常渲染 → VLM 训练集（服务热红外 TIR 的"火山热异常"元素）。

输入：data\raw\03_thermal_infrared\modvolc\<volcano_id>.txt（download_modvolc.py 产物）
      25 字段 ASCII，首行是记录数；字段 1=UNIX时间, 8=经度, 9=纬度, 23=Excess辐射, 24=温度K
输出：datasets\03_thermal_infrared\volcano_thermal\images\*.png + json (+ val)

样本设计（关键：制造真负样本）：
  把每座火山的观测时间轴切成 N 天一段的滑窗，每段渲染一张图：
    - 窗口内显著热异常（Excess >= THRESH）像元数 >= MIN_HOT → Yes
    - 否则 → No（该火山该时段平静，是真负样本）
  同一火山既能出正也能出负，天然平衡，且贴合"识别火山热异常"任务。

用法：
  python src/preprocess/volcano_thermal_render.py --win 30 --step 30   # 30天窗
  python src/preprocess/volcano_thermal_render.py --win 30 --step 15
"""
import os
import re
import sys
import glob
import json
import argparse
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

RAW = r"C:\vlmdisaster\data\raw\03_thermal_infrared\modvolc"
OUT = r"C:\vlmdisaster\datasets\03_thermal_infrared\volcano_thermal"
IMG_DIR = os.path.join(OUT, "images")
OUT_JSON = os.path.join(OUT, "train_volcano_thermal.json")

FIG_SIZE = (4.5, 4.5)
DPI = 100
OUT_SIZE = 384
EXCESS_THRESH = 5.0      # 显著热异常阈值（W m-2 sr-1 um-1）
MIN_HOT = 3              # 窗口内显著热异常像元下限
VAL_PER_CLASS = 30
MAX_SAMPLES_PER_VOLCANO = 40

PROMPT = ("This is a satellite thermal-anomaly (hot-spot) map over a volcanic region. "
          "Is there an active volcanic thermal anomaly in this region? Answer Yes or No.")

PROMPT_TB = ("This is a satellite brightness-temperature field (kelvin) over a volcanic region, "
             "showing the background thermal state of the surface. Describe the brightness "
             "temperature distribution.")


def parse_points(path):
    """返回 [(unix, lon, lat, excess, temp)]"""
    pts = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # 首行记录数
            t = line.split()
            if len(t) < 25:
                continue
            try:
                pts.append((float(t[0]), float(t[7]), float(t[8]),
                            float(t[22]), float(t[23])))
            except (ValueError, IndexError):
                continue
    return pts


def render(pts, title, path):
    """特色场：Excess 热辐射散点（凸显热异常）"""
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    if pts:
        lons = np.array([p[1] for p in pts])
        lats = np.array([p[2] for p in pts])
        exc = np.array([p[3] for p in pts])
        vmax = max(float(np.percentile(exc, 95)), 1.0)
        sc = ax.scatter(lons, lats, c=exc, cmap="hot", s=30,
                        edgecolors="orange", linewidths=0.3, vmin=0, vmax=vmax)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("excess radiance", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No thermal anomaly", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    ax.set_facecolor("#101018")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def render_temp(pts, title, path):
    """本征场：亮温场（K）—— 热红外源的本征物理量"""
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    if pts:
        lons = np.array([p[1] for p in pts])
        lats = np.array([p[2] for p in pts])
        tmp = np.array([p[4] for p in pts])
        ok = np.isfinite(tmp) & (tmp > 100) & (tmp < 400)
        if ok.sum() > 0:
            sc = ax.scatter(lons[ok], lats[ok], c=tmp[ok], cmap="turbo", s=30,
                            edgecolors="none", vmin=250, vmax=340)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("brightness temperature (K)", fontsize=7)
        else:
            ax.text(0.5, 0.5, "no valid BT", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=11)
    else:
        ax.text(0.5, 0.5, "No observation", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    ax.set_facecolor("#101018")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=int, default=30, help="滑窗天数")
    ap.add_argument("--step", type=int, default=30, help="滑窗步长天数")
    args = ap.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*.txt")))
    files = [f for f in files if os.path.basename(f) != "volcano_list.tab"]
    print(f"火山文件: {len(files)}")
    WIN_SEC = args.win * 86400
    STEP_SEC = args.step * 86400

    records = []
    for f in files:
        vid = os.path.splitext(os.path.basename(f))[0]
        if not re.match(r"^\d+$", vid):
            continue
        pts = parse_points(f)
        if not pts:
            continue
        t0 = min(p[0] for p in pts)
        t1 = max(p[0] for p in pts)
        picked = []
        wt = t0
        while wt <= t1:
            seg = [p for p in pts if wt <= p[0] < wt + WIN_SEC]
            hot = [p for p in seg if p[3] >= EXCESS_THRESH]
            lab = "yes" if len(hot) >= MIN_HOT else "no"
            # 用段起始 wt 作为唯一段号，避免空段撞名
            picked.append((lab, seg, int(wt)))
            wt += STEP_SEC
        # 每座火山限量
        pos = [x for x in picked if x[0] == "yes"][:MAX_SAMPLES_PER_VOLCANO // 2]
        neg = [x for x in picked if x[0] == "no"][:MAX_SAMPLES_PER_VOLCANO // 2]
        for lab, seg, seg_id in pos + neg:
            # 特色场：Excess 热辐射
            fn = f"volc_{vid}_{seg_id}_excess.png"
            render(seg, f"Volcano {vid}  excess radiance", os.path.join(IMG_DIR, fn))
            records.append({
                "image": fn, "volcano": vid, "label": lab,
                "product": "feature_excess_radiance",
                "n_points": len(seg),
                "n_hot": sum(1 for p in seg if p[3] >= EXCESS_THRESH),
                "conversations": [
                    {"from": "human", "value": PROMPT},
                    {"from": "gpt",
                     "value": "Yes, there is an active volcanic thermal anomaly in this region."
                     if lab == "yes" else
                     "No, there is no active volcanic thermal anomaly in this region."},
                ],
            })
            # 本征场：亮温（K）
            fn_tb = f"volc_{vid}_{seg_id}_bt.png"
            render_temp(seg, f"Volcano {vid}  brightness temperature", os.path.join(IMG_DIR, fn_tb))
            records.append({
                "image": fn_tb, "volcano": vid, "label": lab,
                "product": "intrinsic_brightness_temp",
                "n_points": len(seg),
                "n_hot": sum(1 for p in seg if p[3] >= EXCESS_THRESH),
                "conversations": [
                    {"from": "human", "value": PROMPT_TB},
                    {"from": "gpt",
                     "value": "This is a brightness temperature field in kelvin over a volcanic "
                              "region. It shows the background surface thermal state."},
                ],
            })
        print(f"  {vid}: {len(pos)} 正 / {len(neg)} 负", flush=True)

    from collections import Counter, defaultdict
    import random as rnd
    rnd.seed(7)
    print(f"\n合计 {len(records)}")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        rnd.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 5))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(OUT_JSON.replace(".json", "_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} | val {len(val)} {dict(Counter(r['label'] for r in val))}")


if __name__ == "__main__":
    main()

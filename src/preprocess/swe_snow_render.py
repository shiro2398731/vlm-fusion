# -*- coding: utf-8 -*-
r"""
雪灾渲染（源 4 被动微波 · AMSR2 AU_DySno 雪水当量）

双产品（方案 C）
----------------
  本征场 = 雪水当量场 SWE（mm）—— 微波本征量，原样呈现，不做换算
  特色场 = 雪深分级（cm）—— 按 10:1 近似由 SWE 换算，套 GB/T 20482-2017《牧区雪灾等级》

换算与判据的全部出处
--------------------
SWE → 雪深：10 mm 雪深 ≈ 1 mm SWE（10:1，干雪/湿雪之间取中值近似）
  干雪 10:1、湿雪 3:1、极干粉雪 30:1，10:1 是常用中值，
  本脚本在图上明标"近似换算"，不隐去这一假设。

分级：GB/T 20482-2017《牧区雪灾等级》（中国国家标准）
  组内实测受害性畜积雪深度：
    牛      10 cm 以内开始受影响
    绵羊    10 ~ 20 cm
    马      20 ~ 30 cm
  据此取分级：
    < 10 cm      无雪灾
    10 ~ 20 cm   轻度（牛开始受影响）
    20 ~ 30 cm   中度（马也受影响）
    >= 30 cm     重度（牧区交通与畜群全面受阻）

重要限制（必须写进图内）
----------------------
  该分级只对牧区成立。同样 15 cm 积雪，在内蒙古牧区可致牲畜死亡，
  在城市只造成交通拥堵。因此本元素判据按"牧区口径"解读。

数据产品与标记值（见产品 comment）
--------------------------------
  SWE_NorthernDaily  0-240 为有效值（mm）
                     247 姿态异常 / 248 出地球 / 252 陆地不可能 / 254 冰 / 255 填充
  实际数据最大约 162 mm，200+ 为 0 像元 —— 需在图上如实反映。

输入：data\raw\04_passive_microwave\swe\*.he5
输出：datasets\04_passive_microwave\snowstorm\images\*.png + json

用法:
  python src/preprocess/swe_snow_render.py --keep 3    # 试渲
  python src/preprocess/swe_snow_render.py             # 全量
"""
import os
import re
import sys
import glob
import json
import argparse
import collections
import random

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from netCDF4 import Dataset
from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

RAW = r"C:\vlmdisaster\data\raw\04_passive_microwave\swe"
OUT = r"C:\vlmdisaster\datasets\04_passive_microwave\snowstorm"
IMG = os.path.join(OUT, "images")
OUT_SIZE = 384
VAL_PER_CLASS = 30

# SWE -> 雪深 换算比（10 mm 雪深 = 1 mm SWE）
SWE_TO_SNOW_CM = 10.0

# 窗口：8×8 格（25 km/格 → 约 200 km 见方）
WIN = 8
NWIN_PER_DAY = 24        # 每天抽取的窗口数

# GB/T 20482-2017《牧区雪灾等级》按牲畜受害的积雪深度分档（cm）
GRADE = [
    ("no_disaster", 0.0, 10.0),
    ("light", 10.0, 20.0),
    ("moderate", 20.0, 30.0),
    ("severe", 30.0, 1e9),
]

# 产品标记值
FILL_VALUES = {247, 248, 252, 253, 254, 255, 241}

PROMPT_SWE = ("This is a passive-microwave snow water equivalent (SWE) map in millimetres "
              "over the Northern Hemisphere. Describe the SWE distribution. Snow-covered "
              "land shows higher SWE than bare ground.")
PROMPT_GRADE = ("This is a snow-depth classification map over the Northern Hemisphere, "
                "graded by GB/T 20482-2017 pastoral snow-disaster standard "
                "(<10 cm none, 10-20 cm light, 20-30 cm moderate, >=30 cm severe). "
                "Classify the snow condition: no_disaster, light, moderate, or severe.")

ANS_GRADE = {
    "no_disaster": "This region has snow depth below 10 cm; no pastoral snow disaster.",
    "light": "This region has snow depth of 10 to 20 cm; cattle begin to be affected. "
             "Light pastoral snow disaster.",
    "moderate": "This region has snow depth of 20 to 30 cm; horses are also affected. "
                "Moderate pastoral snow disaster.",
    "severe": "This region has snow depth of 30 cm or more; livestock and traffic are "
              "severely blocked. Severe pastoral snow disaster.",
}


def _zh_font(size):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(p):
            return font_manager.FontProperties(fname=p, size=size)
    return font_manager.FontProperties(size=size)


def load(fp):
    """读 AMSR2 雪水当量（北半球）"""
    ds = Dataset(fp)
    g = ds.groups["HDFEOS"].groups["GRIDS"].groups["Northern Hemisphere"]
    df = g.groups["Data Fields"]
    swe = np.array(df.variables["SWE_NorthernDaily"][:], dtype="float64")
    flags = np.array(df.variables["Flags_NorthernDaily"][:], dtype="float64")
    lat = np.array(g.variables["lat"][:], dtype="float64")
    lon = np.array(g.variables["lon"][:], dtype="float64")
    ds.close()

    swe = np.ma.filled(swe, np.nan)
    # 标记值排除（>240 全非有效雪水当量）
    swe[(swe > 240) | (swe < 0)] = np.nan
    lat = np.ma.filled(lat, np.nan)
    lon = np.ma.filled(lon, np.nan)
    lat[(lat < -90) | (lat > 90)] = np.nan
    return swe, lat, lon


def grade_of_window(swe_win):
    """按单个窗口判档（GB/T 20482-2017 牧区口径）

    只在窗口内的有雪像元上统计，取 75 分位代表该区域成片积雪的厚度。
    窗口几乎无雪时返回 no_disaster（真负样本）。
    """
    v = swe_win[np.isfinite(swe_win)]
    if v.size == 0:
        return None, 0.0, 0.0
    snow = v[v > 0]
    if snow.size < v.size * 0.02:
        return "no_disaster", 0.0, 0.0
    q75 = float(np.percentile(snow, 75))
    snow_cm = q75 / SWE_TO_SNOW_CM * 10.0
    for name, lo, hi in GRADE:
        if lo <= snow_cm < hi:
            return name, snow_cm, q75
    return GRADE[-1][0], snow_cm, q75


def render_swe(swe, path, stamp, q75):
    """本征场：SWE 场（mm），原样呈现"""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=96)
    im = ax.imshow(swe, cmap="YlGnBu", vmin=0, vmax=160, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  SWE (mm)" % stamp, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("SWE (mm)", fontsize=7)
    fig.text(0.02, 0.02, "passive microwave AMSR2 25km  |  product max ~160 mm",
             fontsize=5.5, color="#666")
    fig.tight_layout()
    fig.savefig(path, dpi=96)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def render_grade(swe, path, stamp, lab, snow_cm, q75):
    """特色场：雪深分级（cm），10:1 近似换算，GB/T 20482-2017"""
    snow_depth = swe / SWE_TO_SNOW_CM * 10.0      # mm SWE -> cm 雪深
    fig = plt.figure(figsize=(4.6, 5.4), dpi=100)
    ax = fig.add_axes([0.04, 0.26, 0.80, 0.66])
    im = ax.imshow(snow_depth, cmap="Blues", vmin=0, vmax=40, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  %s" % (stamp, lab), fontsize=8)
    cax = fig.add_axes([0.86, 0.26, 0.035, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("snow depth (cm)", fontsize=7)
    for thr in (10, 20, 30):
        cb.ax.axhline(thr, color="r", lw=0.8, ls="--")

    fig.text(0.04, 0.20, "Legend: dark blue = deeper snow. Dashed lines 10/20/30 cm",
             fontsize=6.2)
    fig.text(0.04, 0.155, "Const: AMSR2 SWE -> snow depth by 10:1 approx. conversion",
             fontsize=6.2, color="#b00")
    fig.text(0.04, 0.11, "Criteria: GB/T 20482-2017 pastoral snow-disaster grade",
             fontsize=6.2)
    fig.text(0.04, 0.065, "q75 SWE %.0f mm -> snow depth ~%.0f cm" % (q75, snow_cm),
             fontsize=6.2)
    fig.text(0.04, 0.02, "Scope: pasture regions only (same depth differs in city)",
             fontsize=6.0, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*.he5")))
    if args.keep:
        files = files[:args.keep]
    print("待渲染:", len(files))
    if not files:
        print("无数据")
        return

    recs = []
    counter = collections.Counter()

    # 先在首日确定一批固定窗口位置（保证跨天可追踪同一区域）
    try:
        swe0, lat0_, lon0_ = load(files[0])
    except Exception:
        print("首日读取失败")
        return
    H, W = swe0.shape
    stp = max(1, WIN // 2)
    candidates = [(y, x) for y in range(0, H - WIN + 1, stp)
                  for x in range(0, W - WIN + 1, stp)
                  if np.isfinite(swe0[y:y + WIN, x:x + WIN]).mean() > 0.5]
    print("候选窗口: %d" % len(candidates))
    if not candidates:
        print("无有效候选窗口")
        return

    rng = random.Random(31)
    picks = rng.sample(candidates, NWIN_PER_DAY) if len(candidates) > NWIN_PER_DAY else candidates
    picks.sort()
    print("固定窗口: %d 个（跨天追踪同一区域）" % len(picks))

    for i, f in enumerate(files):
        m = re.search(r"(\d{8})", os.path.basename(f))
        stamp = m.group(1) if m else str(i)
        try:
            swe, lat, lon = load(f)
        except Exception as e:
            print("  %s 读取失败 %s" % (stamp, str(e)[:60]))
            continue

        for (y, x) in picks:
            sub = swe[y:y + WIN, x:x + WIN]
            if np.isfinite(sub).mean() < 0.5:
                continue
            lab, snow_cm, q75 = grade_of_window(sub)
            if lab is None:
                continue
            counter[lab] += 1

            tag = "%s_%03d_%03d" % (stamp, y, x)
            p1 = os.path.join(IMG, "%s_swe.png" % tag)
            render_swe(sub, p1, "%s [%d,%d]" % (stamp, y, x), q75)
            recs.append({"image": os.path.basename(p1), "stamp": stamp,
                         "win": [int(y), int(x)], "label": lab,
                         "product": "intrinsic_swe", "snow_cm": round(snow_cm, 1),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_SWE},
                             {"from": "gpt",
                              "value": "This is a snow water equivalent field in millimetres "
                                       "over a 200 km window. Snow-covered land shows higher SWE; "
                                       "the 75th percentile of snow area is about %.0f mm." % q75}]})

            p2 = os.path.join(IMG, "%s_grade.png" % tag)
            render_grade(sub, p2, "%s [%d,%d]" % (stamp, y, x), lab, snow_cm, q75)
            recs.append({"image": os.path.basename(p2), "stamp": stamp,
                         "win": [int(y), int(x)], "label": lab,
                         "product": "feature_snow_grade", "snow_cm": round(snow_cm, 1),
                         "q75_swe_mm": round(q75, 1),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_GRADE},
                             {"from": "gpt", "value": ANS_GRADE[lab]}]})

        if i % 20 == 0 or i == len(files) - 1:
            print("  [%d/%d] %s  累计样本 %d" % (i + 1, len(files), stamp, len(recs)))

    random.seed(29)
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["product"], r["label"])].append(r)
    val, train = [], []
    for k, lst in by.items():
        random.shuffle(lst)
        n = min(VAL_PER_CLASS // 4, max(3, len(lst) // 6))
        val.extend(lst[:n]); train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_snowstorm.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_snowstorm_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("窗口级分布:", dict(counter))
    print("产品×标签:")
    for k in sorted(by):
        print("   %-24s %-14s %d" % (k[0], k[1], len(by[k])))
    print("输出:", IMG)


if __name__ == "__main__":
    main()

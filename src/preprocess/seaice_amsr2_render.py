# -*- coding: utf-8 -*-
r"""
海冰/凌汛 渲染（源 4 被动微波 · AMSR2 AU_SI12 12.5km 第二源）

为什么改判据
------------
旧版用"整个北半球的冰覆盖占比"判冰情，结果 48 天全落在同一档（55%~59%）。
原因：北冰洋主体常年是冰，全图统计永远在中间。

新版改为**看冰在哪里**，而不是"有多少冰"：
  1. 冰缘位置（ice edge latitude）：冰密集度 >= 15% 的最南纬度
  2. 渤海冰情：中国近海唯一有实际海冰灾害意义的区域

双产品
------
  本征场 = 海冰密集度场（%）—— 被动微波本征物理量
  特色场 = 冰缘位置 + 渤海冰情分级

判据（WMO）
-----------
  海冰密集度 >= 15% 定义为冰区边界（WMO / OSI SAF 国际通用）
  冰缘向南推进 = 冰情加重
  渤海冰覆盖（>=15% 像元占比）：
      0        -> no_ice     无冰
      >0 ~ 30% -> partial    局部结冰，航道受限
      >30%     -> frozen     大面积封冻，港口与养殖受阻

数据标记值（产品 comment）
------------------------
  0            Open Water
  1~100        有效密集度
  110          missing / not calculated  -> 剔除
  120          Land                     -> 剔除

输入：data\raw\04_passive_microwave\seaice_amsr2\*.he5
输出：datasets\04_passive_microwave\sea_ice_amsr2\images\*.png + json

用法:
  python src/preprocess/seaice_amsr2_render.py --keep 3
  python src/preprocess/seaice_amsr2_render.py
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

PILImage.MAX_PIXELS = None
PILImage.MAX_IMAGE_PIXELS = None

RAW = r"C:\vlmdisaster\data\raw\04_passive_microwave\seaice_amsr2"
OUT = r"C:\vlmdisaster\datasets\04_passive_microwave\sea_ice_amsr2"
IMG = os.path.join(OUT, "images")
OUT_SIZE = 384
VAL_PER_GROUP = 30

ICE = 15.0        # WMO：密集度 >= 15% 为冰区

# 渤海海区（中国近海唯一有实际海冰灾害意义的区域）
BOH_LON = (117.0, 122.0)
BOH_LAT = (37.0, 41.0)

PROMPT_CONC = ("This is an AMSR2 passive-microwave sea-ice concentration map (percent) "
               "of the Northern Hemisphere. Describe the ice concentration distribution "
               "and where the ice edge lies.")
PROMPT_GRADE = ("This is a sea-ice condition map of the Northern Hemisphere showing the ice "
                "edge (concentration >= 15 percent, WMO definition) and the ice condition "
                "in the Bohai Sea. Classify the ice condition: no_ice, partial, or frozen.")

ANS_GRADE = {
    "no_ice": "The Bohai Sea is ice-free; no sea-ice hazard.",
    "partial": "The Bohai Sea has partial ice cover; navigation is restricted.",
    "frozen": "The Bohai Sea is largely frozen; ports and aquaculture are blocked.",
}


def _zh(size):
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(p):
            return font_manager.FontProperties(fname=p, size=size)
    return font_manager.FontProperties(size=size)


def load(fp):
    """读 AMSR2 12km 北半球：冰密集度 + 经纬度（已剔陆地与缺失）"""
    ds = Dataset(fp)
    g = ds.groups["HDFEOS"].groups["GRIDS"].groups["NpPolarGrid12km"]
    df = g.groups["Data Fields"]
    ice = np.ma.filled(np.array(df.variables["SI_12km_NH_ICECON_DAY"][:], dtype="float64"), np.nan)
    lat = np.ma.filled(np.array(g.variables["lat"][:], dtype="float64"), np.nan)
    lon = np.ma.filled(np.array(g.variables["lon"][:], dtype="float64"), np.nan)
    ds.close()

    ice[(ice == 110) | (ice == 120)] = np.nan     # 缺失 / 陆地
    ice[(ice < 0) | (ice > 100)] = np.nan
    lat[(lat < -90) | (lat > 90)] = np.nan
    return ice, lat, lon


def ice_edge_lat(ice, lat):
    """冰缘纬度：冰密集度 >= 15% 的最南纬度

    做法：按纬度从低到高扫，找到第一个"该纬度带有 >=1% 像元达 15%"的 0.5° 带。
    用 0.5% 占比阈值过滤零星噪点。
    """
    m = np.isfinite(ice) & np.isfinite(lat)
    if m.sum() < 1000:
        return None
    la = lat[m]
    ic = ice[m]
    for edge in np.arange(30.0, 90.0, 0.5):
        sel = (la >= edge) & (la < edge + 0.5)
        if sel.sum() < 50:
            continue
        frac = np.mean(ic[sel] >= ICE)
        if frac >= 0.005:        # 该纬带至少 0.5% 像元是冰
            return float(edge)
    return None


def bohai_frac(ice, lat, lon):
    """渤海冰覆盖：区内 >=15% 像元占比"""
    m = (np.isfinite(ice) & np.isfinite(lat) & np.isfinite(lon)
         & (lon >= BOH_LON[0]) & (lon < BOH_LON[1])
         & (lat >= BOH_LAT[0]) & (lat < BOH_LAT[1]))
    if m.sum() < 20:
        return None, 0
    v = ice[m]
    return float(np.mean(v >= ICE) * 100.0), int(m.sum())


def grade_of(bf):
    """渤海冰情分级（判据见文件头）"""
    if bf is None:
        return None
    if bf <= 0.0:
        return "no_ice"
    if bf <= 30.0:
        return "partial"
    return "frozen"


def render_conc(ice, path, stamp, edge, bf, lab):
    """本征场：冰密集度场（%）"""
    fig = plt.figure(figsize=(4.8, 4.8), dpi=100)
    ax = fig.add_axes([0.03, 0.10, 0.84, 0.84])
    im = ax.imshow(ice, cmap="Blues", vmin=0, vmax=100, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  sea-ice concentration" % stamp, fontsize=8)
    cax = fig.add_axes([0.89, 0.10, 0.035, 0.84])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ice concentration (%)", fontsize=7)
    cb.ax.axhline(ICE, color="r", lw=0.9, ls="--")
    fig.text(0.03, 0.045, "Sensor: AMSR2 12.5 km, NOAA/NSIDC AU_SI12", fontsize=6)
    fig.text(0.03, 0.012, "WMO: concentration >= 15% defines the ice zone", fontsize=6)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def render_grade(ice, path, stamp, edge, bf, lab):
    """特色场：冰缘线 + 渤海框 + 分级文字"""
    fig = plt.figure(figsize=(4.8, 5.4), dpi=100)
    ax = fig.add_axes([0.03, 0.20, 0.84, 0.74])
    mask = np.where(np.isfinite(ice) & (ice >= ICE), ice, np.nan)
    im = ax.imshow(mask, cmap="Blues", vmin=15, vmax=100, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  %s" % (stamp, lab), fontsize=8)
    cax = fig.add_axes([0.89, 0.20, 0.035, 0.74])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("concentration >= 15% (%)", fontsize=6.5)

    edge_s = "%.1f N" % edge if edge is not None else "n/a"
    bf_s = "%.1f %%" % bf if bf is not None else "n/a"
    fig.text(0.03, 0.155, "Ice edge (southernmost lat with >=15%% ice): %s" % edge_s,
             fontsize=6.5)
    fig.text(0.03, 0.115, "Bohai Sea ice cover (>=15%% pixels): %s" % bf_s, fontsize=6.5)
    fig.text(0.03, 0.075, "Grade: 0%%=no_ice / 0-30%%=partial / >30%%=frozen",
             fontsize=6.5)
    fig.text(0.03, 0.035, "Criteria: WMO ice-zone definition (15%)",
             fontsize=6.2, color="#b00")
    fig.text(0.03, 0.005, "Bohai box: 117-122E, 37-41N", fontsize=6.0, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(RAW, "*.he5")))
    if args.keep:
        fs = fs[:args.keep]
    print("待渲染:", len(fs))
    if not fs:
        print("无数据")
        return

    recs = []
    counter = collections.Counter()
    edges = []
    fracs = []

    for i, f in enumerate(fs):
        m = re.search(r"(\d{8})", os.path.basename(f))
        stamp = m.group(1) if m else str(i)
        try:
            ice, lat, lon = load(f)
        except Exception as e:
            print("  %s 读取失败 %s" % (stamp, str(e)[:60]))
            continue

        edge = ice_edge_lat(ice, lat)
        bf, npix = bohai_frac(ice, lat, lon)
        lab = grade_of(bf)
        if lab is None:
            print("  %s 渤海无有效像元" % stamp)
            continue

        counter[lab] += 1
        if edge is not None:
            edges.append(edge)
        fracs.append(bf)

        p1 = os.path.join(IMG, "%s_conc.png" % stamp)
        render_conc(ice, p1, stamp, edge, bf, lab)
        recs.append({"image": os.path.basename(p1), "stamp": stamp, "label": lab,
                     "product": "intrinsic_concentration",
                     "ice_edge_lat": round(edge, 2) if edge else None,
                     "bohai_frac": round(bf, 2),
                     "conversations": [
                         {"from": "human", "value": "<image>" + PROMPT_CONC},
                         {"from": "gpt",
                          "value": "This is an AMSR2 sea-ice concentration field. The ice edge "
                                   "(concentration 15%) lies near %.1f N, and the Bohai Sea shows "
                                   "%.1f%% ice-covered pixels." % (edge or 0, bf)}]})

        p2 = os.path.join(IMG, "%s_grade.png" % stamp)
        render_grade(ice, p2, stamp, edge, bf, lab)
        recs.append({"image": os.path.basename(p2), "stamp": stamp, "label": lab,
                     "product": "feature_ice_condition",
                     "ice_edge_lat": round(edge, 2) if edge else None,
                     "bohai_frac": round(bf, 2),
                     "conversations": [
                         {"from": "human", "value": "<image>" + PROMPT_GRADE},
                         {"from": "gpt", "value": ANS_GRADE[lab]}]})

        if i % 20 == 0 or i == len(fs) - 1:
            print("  [%d/%d] %s => %-8s 冰缘 %s  渤海 %.1f%%"
                  % (i + 1, len(fs), stamp, lab,
                     ("%.1fN" % edge) if edge else "n/a", bf))

    if not recs:
        print("无样本")
        return

    # 分层切分
    random.seed(41)
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["product"], r["label"])].append(r)
    val, train = [], []
    for k, lst in by.items():
        lst = lst[:]
        random.shuffle(lst)
        n = min(VAL_PER_GROUP, max(5, len(lst) // 5))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_sea_ice_amsr2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_sea_ice_amsr2_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("日级分布:", dict(counter))
    print("产品×标签:")
    for k in sorted(by):
        print("   %-26s %-10s %d" % (k[0], k[1], len(by[k])))
    if edges:
        print("冰缘纬度: 最南 %.1fN  最北 %.1fN  中位 %.1fN"
              % (min(edges), max(edges), float(np.median(edges))))
    if fracs:
        print("渤海冰覆盖: 最小 %.1f%%  最大 %.1f%%  中位 %.1f%%"
              % (min(fracs), max(fracs), float(np.median(fracs))))
    print("输出:", IMG)


if __name__ == "__main__":
    main()

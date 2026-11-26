# -*- coding: utf-8 -*-
r"""
07 散射计 · hurricane_wind（台风海面风场）

数据源
-----
  CMEMS L4 融合风场 cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H
  0.125 度约 14 km，已含 ASCAT 散射计观测
  与 ocean_gale（2266 条）同源，不重复下载

为什么这个元素不和别的撞
--------------------
  01 光学 hurricane     测云顶形态
  06 雷达降水 hurricane  测降水
  07 hurricane_wind    测海面 10 米风场   <- 物理量不同

判据
----
  按中国《热带气旋等级》GB/T 19201-2006，以底层中心附近最大平均风速分级：
    热带低压   10.8~17.1 m/s
    热带风暴   17.2~24.4 m/s
    强热带风暴 24.5~32.6 m/s
    台风       32.7~41.4 m/s
    强台风     41.5~50.9 m/s
    超强台风   >= 51.0 m/s

  注意：卫星给出的是 10 米高度风速（U10），与国标要求的
  "底层中心附近最大平均风速"定义不完全等同，图注中明标这一点。

双产品
------
  本征场  风速场（m/s），按热带气旋等级着色
  特色场  风速异常（相对该时段中位数）

用法
----
  python src/preprocess/render_hurricane_wind.py --probe
  python src/preprocess/render_hurricane_wind.py --all
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = r"C:\vlmdisaster\data\raw\07_scatterometer\wind"
OUT = r"C:\vlmdisaster\datasets\07_scatterometer\hurricane_wind"
IMG = os.path.join(OUT, "images")

# 台风 In-fa 海域
LON_MIN, LON_MAX = 130.0, 160.0
LAT_MIN, LAT_MAX = 10.0, 30.0
WIN = 96                 # 0.125 度像元，96 格约 12 度
STRIDE = 48
OUT_SIZE = 384

LEVELS = [
    (0.0, 10.8, "below_td", "#b9d8f0"),
    (10.8, 17.2, "td", "#7fb2e0"),
    (17.2, 24.5, "ts", "#2f6fd0"),
    (24.5, 32.7, "sts", "#f0c419"),
    (32.7, 41.5, "typhoon", "#f08019"),
    (41.5, 51.0, "severe", "#e04010"),
    (51.0, 999.0, "super", "#a000a0"),
]


def band_of(v):
    for lo, hi, name, _ in LEVELS:
        if lo <= v < hi:
            return name
    return "below_td"


def read_wind(path):
    """读 CMEMS L4 风场，返回 (u, v, lon, lat)

    变名：lat / lon（不是 latitude / longitude）
    netCDF4 已自动套用 scale_factor=0.01，不能再手乘
    """
    import netCDF4
    ds = netCDF4.Dataset(path, "r")
    vn = list(ds.variables.keys())

    def pick(names):
        for k in names:
            if k in vn:
                return k
        return None

    ku = pick(["eastward_wind", "u10", "wind_speed_u", "U"])
    kv = pick(["northward_wind", "v10", "wind_speed_v", "V"])
    if ku is None or kv is None:
        raise RuntimeError("找不到风场变量，现有: %s" % vn[:8])

    u = np.ma.filled(np.ma.asarray(ds.variables[ku][0], dtype="float64"), np.nan)
    v = np.ma.filled(np.ma.asarray(ds.variables[kv][0], dtype="float64"), np.nan)
    klon = pick(["lon", "longitude"])
    klat = pick(["lat", "latitude"])
    lon = np.ma.filled(np.ma.asarray(ds.variables[klon][:], dtype="float64"), np.nan)
    lat = np.ma.filled(np.ma.asarray(ds.variables[klat][:], dtype="float64"), np.nan)
    ds.close()
    return u, v, lon, lat


def render(field, path, tag, lab, kind, vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from PIL import Image as PILImage

    fig = plt.figure(figsize=(4.6, 4.9), dpi=100)
    ax = fig.add_axes([0.03, 0.16, 0.86, 0.78])

    if kind == "wind":
        bounds = [b[0] for b in LEVELS] + [LEVELS[-1][1]]
        cmap = ListedColormap([b[3] for b in LEVELS])
        norm = BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(field, cmap=cmap, norm=norm, interpolation="nearest")
        cax = fig.add_axes([0.90, 0.16, 0.03, 0.78])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Wind speed U10 (m/s)")
        handles = [Patch(facecolor=b[3], label="%s %.1f-" % (b[2], b[0]))
                   for b in LEVELS]
        ax.legend(handles=handles, loc="lower left", fontsize=5.8,
                  framealpha=0.92, ncol=2)
    else:
        im = ax.imshow(field, cmap="RdBu_r", vmin=-15, vmax=15,
                       interpolation="nearest")
        cax = fig.add_axes([0.90, 0.16, 0.03, 0.78])
        fig.colorbar(im, cax=cax, label="Wind anomaly (m/s)")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("%s  %s" % (tag, "wind speed" if kind == "wind" else "anomaly"),
                 fontsize=7.5)
    fig.text(0.03, 0.095,
             "CMEMS L4 blended wind (0.125 deg, includes ASCAT)",
             fontsize=5.8)
    fig.text(0.03, 0.055, "Reading: %s" % lab, fontsize=6.0)
    fig.text(0.03, 0.018,
             "U10 wind, not the 10-min mean at cyclone centre (GB/T 19201)",
             fontsize=5.6, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE),
                                              PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(RAW, "*.nc")))
    print("文件 %d 个" % len(fs))
    if not fs:
        return

    if args.probe:
        u, v, lon, lat = read_wind(fs[0])
        spd = np.sqrt(u ** 2 + v ** 2)
        print("shape", spd.shape)
        print("lon %.2f~%.2f  lat %.2f~%.2f" % (lon.min(), lon.max(), lat.min(), lat.max()))
        good = spd[np.isfinite(spd)]
        print("风速 min %.2f  中位 %.2f  max %.2f m/s" % (good.min(), np.median(good), good.max()))
        return

    recs = []
    for i, fp in enumerate(fs, 1):
        tag = os.path.basename(fp).split("_")[-3]
        try:
            u, v, lon, lat = read_wind(fp)
        except Exception as e:
            print("  x %s %s" % (tag, str(e)[:60]))
            continue
        spd = np.sqrt(u ** 2 + v ** 2)
        H, W = spd.shape

        # 找目标区子矩阵
        li = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]
        la = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
        if li.size < WIN or la.size < WIN:
            print("  - %s 海域覆盖不足" % tag)
            continue
        y0, y1 = int(la.min()), int(la.max())
        x0, x1 = int(li.min()), int(li.max())
        sub = spd[y0:y1, x0:x1]
        base = float(np.nanmedian(sub))

        kept = 0
        for y in range(0, max(1, sub.shape[0] - WIN), STRIDE):
            for x in range(0, max(1, sub.shape[1] - WIN), STRIDE):
                w = sub[y:y + WIN, x:x + WIN]
                if np.isfinite(w).mean() < 0.9:
                    continue
                mx = float(np.nanmax(w))
                lab = band_of(mx)
                t = "%s_%d_%d" % (tag, y, x)

                p1 = os.path.join(IMG, "%s_wind.png" % t)
                render(w, p1, t, lab, "wind")
                recs.append({"image": os.path.basename(p1), "stem": t,
                             "label": lab, "max_ms": round(mx, 1),
                             "product": "intrinsic_wind_speed"})

                anom = w - base
                p2 = os.path.join(IMG, "%s_anom.png" % t)
                render(anom, p2, t, lab, "anom")
                recs.append({"image": os.path.basename(p2), "stem": t,
                             "label": lab, "max_ms": round(mx, 1),
                             "product": "feature_wind_anomaly"})
                kept += 1
        print("  v %s  窗口 %d  累计 %d" % (tag, kept, len(recs)))

    if not recs:
        print("无样本")
        return

    json.dump(recs, open(os.path.join(OUT, "train_hurricane_wind.json"),
                         "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print("\n样本 %d" % len(recs))
    print("分级:", dict(Counter(r["label"] for r in recs)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
台风 In-fa · 显著波高面状场拼图

输入：outputs/infa_wave/swh_points.npz（read_infa_swh.py 生成的轨道点）
输出：outputs/infa_wave/*.png

两种产品
--------
  本征场  有效波高（Hs）面状分布，按《风暴潮、海浪、海冰和海啸灾害
          应急预案》四级预警色分级（蓝 2.5 / 黄 3.5 / 橙 4.5 / 红 6.0 m）
          同时在图上标注"开阔海域有效波高，非沿岸预警值"

  特色场  波高异常 = 各格点 Hs 减去该格点所在纬度带的平静基准（中位数）
          用来突出"哪里比常态高得异常"

网格
----
  0.25° 等经纬度网格。每格取中位数（抗离群），
  同时记录格内点数，点数 < 3 的格子不画（避免单点噪声）

用法
----
  python src/preprocess/render_infa_wave.py
  python src/preprocess/render_infa_wave.py --cell 0.5
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INP = r"C:\vlmdisaster\outputs\infa_wave\swh_points.npz"
OUT = r"C:\vlmdisaster\outputs\infa_wave"

# 《风暴潮、海浪、海冰和海啸灾害应急预案》海浪警报
LEVELS = [
    (0.0, 1.25, "calm", "#b9d8f0"),
    (1.25, 2.5, "moderate", "#7fb2e0"),
    (2.5, 3.5, "blue", "#2f6fd0"),
    (3.5, 4.5, "yellow", "#f0c419"),
    (4.5, 6.0, "orange", "#f08019"),
    (6.0, 99.0, "red", "#d02020"),
]


def grid_stat(lat, lon, swh, cell, min_pts=3):
    """按格聚合，返回 (lon_edges, lat_edges, med2d, cnt2d)"""
    lon_min = np.floor(lon.min() / cell) * cell
    lon_max = np.ceil(lon.max() / cell) * cell
    lat_min = np.floor(lat.min() / cell) * cell
    lat_max = np.ceil(lat.max() / cell) * cell

    nx = int(round((lon_max - lon_min) / cell))
    ny = int(round((lat_max - lat_min) / cell))
    ix = np.clip(((lon - lon_min) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((lat - lat_min) / cell).astype(int), 0, ny - 1)

    flat = iy * nx + ix
    n = nx * ny
    cnt = np.bincount(flat, minlength=n).astype("float64")
    ssum = np.bincount(flat, weights=swh, minlength=n)

    # 中位数：逐格算（格数不多，够快）
    med = np.full(n, np.nan)
    order = np.argsort(flat, kind="stable")
    fs = flat[order]
    vs = swh[order]
    bounds = np.searchsorted(fs, np.arange(n + 1))
    for k in range(n):
        a, b = bounds[k], bounds[k + 1]
        if b - a >= min_pts:
            med[k] = np.median(vs[a:b])

    med2d = med.reshape(ny, nx)
    cnt2d = cnt.reshape(ny, nx)
    # 经度轴自西向东
    lon_edges = np.linspace(lon_min, lon_min + nx * cell, nx + 1)
    lat_edges = np.linspace(lat_min, lat_min + ny * cell, ny + 1)
    return lon_edges, lat_edges, med2d, cnt2d


def render_field(lon_edges, lat_edges, field, cnt, path, title, kind,
                 vmin=None, vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(9.6, 7.6), dpi=110)
    ax = fig.add_axes([0.075, 0.13, 0.86, 0.74])

    if kind == "hs":
        bounds = [b for band in LEVELS for b in (band[0],)] + [LEVELS[-1][1]]
        colors = [band[3] for band in LEVELS]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)
    else:
        cmap = "RdBu_r"
        norm = None

    pcm = ax.pcolormesh(lon_edges, lat_edges, field, cmap=cmap, norm=norm,
                        vmin=vmin, vmax=vmax, shading="flat")
    ax.set_aspect("equal", adjustable="box")

    total = int(np.isfinite(field).sum())
    valid_cells = int(np.isfinite(field).sum())
    ax.set_title("%s\n%d cells with >=3 measurements (grid %.2f deg)"
                 % (title, valid_cells,
                    lon_edges[1] - lon_edges[0]), fontsize=10)
    ax.set_xlabel("Longitude (deg E)", fontsize=9)
    ax.set_ylabel("Latitude (deg N)", fontsize=9)

    cax = fig.add_axes([0.945, 0.13, 0.022, 0.74])
    cb = fig.colorbar(pcm, cax=cax)
    if kind == "hs":
        cb.set_label("Significant wave height Hs (m)", fontsize=9)
        handles = [Patch(facecolor=b[3],
                         label="%s  %.2f-%.2f m" % (b[2], b[0], b[1]))
                   for b in LEVELS]
        ax.legend(handles=handles, loc="lower left", fontsize=7.5,
                  framealpha=0.9, ncol=2)
    else:
        cb.set_label("Hs anomaly (m)", fontsize=9)

    fig.text(0.075, 0.055,
             "Sensor: Sentinel-3A/B SRAL L2 (20 Hz swh_ocean_ku), median per cell",
             fontsize=7.6)
    fig.text(0.075, 0.022,
             "Open-ocean Hs, NOT the coastal warning value of the national plan",
             fontsize=7.6, color="#b00")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("  写入:", os.path.basename(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=float, default=0.25)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    d = np.load(INP)
    lat, lon, swh = d["lat"], d["lon"], d["swh"]
    print("点 %d  经度 %.1f~%.1f  纬度 %.1f~%.1f"
          % (len(swh), lon.min(), lon.max(), lat.min(), lat.max()))

    lon_e, lat_e, med, cnt = grid_stat(lat, lon, swh, args.cell)
    ny, nx = med.shape
    print("网格 %dx%d  有效格 %d (%.1f%%)"
          % (ny, nx, int(np.isfinite(med).sum()),
             100.0 * np.isfinite(med).sum() / (nx * ny)))

    render_field(lon_e, lat_e, med, cnt,
                 os.path.join(OUT, "infa_hs_field.png"),
                 "Typhoon In-fa (2026-10-27..29) significant wave height",
                 "hs")

    # 纬度带基准：每个纬度行有效格的中位数
    base = np.full(ny, np.nan)
    for j in range(ny):
        row = med[j][np.isfinite(med[j])]
        if row.size >= 5:
            base[j] = np.median(row)
    anom = med - base[:, None]

    render_field(lon_e, lat_e, anom, cnt,
                 os.path.join(OUT, "infa_hs_anomaly.png"),
                 "Typhoon In-fa Hs anomaly vs latitude-band baseline",
                 "anom", vmin=-4, vmax=4)

    # 统计
    fin = med[np.isfinite(med)]
    print("\n格值统计: 中位 %.2f  P95 %.2f  最大 %.2f"
          % (np.median(fin), np.percentile(fin, 95), fin.max()))
    for lo, hi, name, _ in LEVELS:
        c = int(((fin >= lo) & (fin < hi)).sum())
        print("  %-10s %4d 格  %.1f%%" % (name, c, 100.0 * c / fin.size))

    np.savez_compressed(os.path.join(OUT, "infa_grid.npz"),
                        lon_edges=lon_e, lat_edges=lat_e,
                        med=med, cnt=cnt, anom=anom)
    print("\n网格已存:", os.path.join(OUT, "infa_grid.npz"))


if __name__ == "__main__":
    main()

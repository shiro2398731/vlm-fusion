# -*- coding: utf-8 -*-
r"""
台风 In-fa · 巨浪 · 轨道条带图渲染（第一层产品）

一张图 = 一条 Sentinel-3 轨道的沿轨有效波高剖面。

为什么用条带而不是面
------------------
  测高是沿轨细条带，宽度只有几公里。60 条轨道撒在 10°x35° 海域，
  0.25° 网格下只有 2.9% 的格子有 >=3 个观测。
  插值补洞 = 编数据，违反"宁可样本少也不造假"。

  所以本征场直接呈现沿轨剖面：横轴沿轨距离，纵轴 Hs。
  这是观测本身的形状，不做任何虚构。

分级
----
  按《风暴潮、海浪、海冰和海啸灾害应急预案》海浪警报：
    蓝 2.5~3.5 / 黄 3.5~4.5 / 橙 4.5~6.0 / 红 >= 6.0 m

  注意：预案阈值针对近岸/沿岸，卫星测的是开阔海域有效波高，
  能量在传向岸边时会衰减，两类数值不可直接等同。图上明标。

双产品
------
  本征场 沿轨剖面 + 波高分档色带（看的物理量就是 Hs 本身）
  特色场 波高相对该纬度正常海况的偏离（异常值，突出"哪里异常高"）

用法
----
  python src/preprocess/render_infa_tracks.py
  python src/preprocess/render_infa_tracks.py --min-pts 200
  python src/preprocess/render_infa_tracks.py --limit 5
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = r"C:\vlmdisaster\data\raw\08_altimetry\infa"
OUT = r"C:\vlmdisaster\datasets\08_altimetry\wave"
IMG_DIR = os.path.join(OUT, "images")
TMP = r"C:\vlmdisaster\outputs\infa_wave\_tmp_tracks"

ERUPT_LEVELS = [
    (2.5, 3.5, "blue", "#2f6fd0"),
    (3.5, 4.5, "yellow", "#f0c419"),
    (4.5, 6.0, "orange", "#f08019"),
    (6.0, 99.0, "red", "#d02020"),
]

PROMPT = ("This is a Sentinel-3 SRAL altimeter along-track significant wave height "
          "profile across the western North Pacific during Typhoon In-fa (2026). "
          "The horizontal axis is along-track distance, the vertical axis is Hs in "
          "metres. Describe the sea state shown.")
ANS = {
    "red": "Significant wave height reaches the red warning level (>= 6.0 m) along "
           "part of this track, indicating a violent sea associated with the typhoon.",
    "orange": "Significant wave height reaches the orange warning level "
              "(4.5-6.0 m), a rough to very rough sea.",
    "yellow": "Significant wave height reaches the yellow warning level "
              "(3.5-4.5 m), a rough sea.",
    "blue": "Significant wave height reaches the blue warning level "
            "(2.5-3.5 m), a moderate to rough sea.",
    "calm": "Significant wave height stays below the blue warning level "
            "(< 2.5 m); no wave warning would be issued for this track.",
}


def read_var(ds, name):
    """netCDF4 默认已自动套用 scale_factor，不能再手乘"""
    if name not in ds.variables:
        return None
    return np.ma.filled(np.ma.asarray(ds.variables[name][:], dtype="float64"),
                        np.nan)


def load_track(zp):
    """解压一个产品，返回 (lat, lon, swh, t_sec)"""
    import zipfile
    import netCDF4

    tmp = os.path.join(TMP, os.path.basename(zp).replace(".zip", ""))
    os.makedirs(tmp, exist_ok=True)
    if not os.listdir(tmp):
        zipfile.ZipFile(zp).extractall(tmp)

    std_path = None
    for dp, _, fns in os.walk(tmp):
        for fn in fns:
            if fn.endswith(".nc") and "_STD__" in fn:
                std_path = os.path.join(dp, fn)
                break
        if std_path:
            break
    if not std_path:
        return None

    ds = netCDF4.Dataset(std_path, "r")
    lat = read_var(ds, "lat_20_ku")
    lon = read_var(ds, "lon_20_ku")
    swh = read_var(ds, "swh_ocean_20_ku")
    qual = read_var(ds, "swh_ocean_qual_20_ku")
    tm = None
    for cand in ("time_20_ku", "time_20_plrm_ku"):
        if cand in ds.variables:
            tm = read_var(ds, cand)
            break
    ds.close()

    if lat is None or lon is None or swh is None:
        return None

    n = min(len(lat), len(lon), len(swh))
    lat, lon, swh = lat[:n], lon[:n], swh[:n]
    if qual is not None:
        qual = qual[:n]

    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(swh)
    if qual is not None:
        ok &= (qual == 0)
    ok &= (swh > 0) & (swh < 25)

    return lat[ok], lon[ok], swh[ok], (tm[:n][ok] if tm is not None else None)


def path_distance(lat, lon):
    """累计大圆距离（km）"""
    R = 6371.0
    dlat = np.radians(np.diff(lat))
    dlon = np.radians(np.diff(lon))
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat[:-1])) * np.cos(np.radians(lat[1:]))
         * np.sin(dlon / 2) ** 2)
    d = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return np.concatenate([[0.0], np.cumsum(d)])


def band_of(v):
    for lo, hi, name, _ in ERUPT_LEVELS:
        if lo <= v < hi:
            return name
    return "calm"


def render(dist, swh, lat, lon, meta, path, kind, anomaly=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(9.2, 5.0), dpi=105)
    ax = fig.add_axes([0.075, 0.17, 0.885, 0.68])

    if kind == "anomaly":
        ax.fill_between(dist, 0, anomaly, color="#888", lw=0)
        ax.plot(dist, anomaly, color="#333", lw=0.9)
        ax.axhline(0, color="#000", lw=0.8)
        ax.set_ylabel("Hs anomaly (m)", fontsize=9)
        title = "Hs anomaly vs 30-day latitude climatology"
    else:
        # 分档色带
        for lo, hi, name, col in ERUPT_LEVELS:
            ax.axhspan(lo, hi, color=col, alpha=0.20, lw=0)
        ax.plot(dist, swh, color="#123", lw=1.1)
        ax.set_ylim(0, max(8.0, float(np.nanmax(swh)) * 1.12))
        ax.set_ylabel("Significant wave height Hs (m)", fontsize=9)
        title = "Along-track Hs profile"

    ax.set_xlabel("Along-track distance (km)", fontsize=9)
    ax.set_title("%s\n%s\nlon %.1f-%.1f E, lat %.1f-%.1f N, max Hs %.2f m"
                 % (title, meta["short"], lon.min(), lon.max(),
                    lat.min(), lat.max(), float(np.nanmax(swh))), fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

    if kind != "anomaly":
        handles = [Patch(facecolor=c, alpha=0.5,
                         label="%s %.1f-%.1f m" % (n, lo, hi))
                   for lo, hi, n, c in ERUPT_LEVELS]
        ax.legend(handles=handles, loc="upper left", fontsize=7,
                  framealpha=0.92, ncol=2)

    fig.text(0.075, 0.075, "Sensor: Sentinel-3 SRAL L2, 20 Hz swh_ocean_ku, "
                           "open-ocean values", fontsize=7.2)
    fig.text(0.075, 0.032,
             "Warning levels from the national marine disaster plan (coastal "
             "thresholds, NOT directly comparable to open-ocean Hs)",
             fontsize=7.2, color="#b00")
    fig.savefig(path, dpi=105)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pts", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--segment", type=int, default=2,
                    help="每条轨道切成几段出图（默认 2）")
    args = ap.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(RAW, "*.zip")))
    if args.limit:
        zips = zips[:args.limit]
    print("产品 %d 个" % len(zips))

    # 纬度带气候基准（用全部轨道的中位数按 2° 分箱）
    all_lat, all_swh = [], []
    tracks = []
    for zp in zips:
        try:
            r = load_track(zp)
        except Exception as e:
            print("  ✗ %s %s" % (os.path.basename(zp)[18:40], str(e)[:50]))
            continue
        if r is None or len(r[0]) < args.min_pts:
            continue
        tracks.append((zp, r))
        all_lat.append(r[0])
        all_swh.append(r[2])

    if not tracks:
        print("没有可用轨道")
        return
    all_lat = np.concatenate(all_lat)
    all_swh = np.concatenate(all_swh)
    print("可用轨道 %d 条，点 %d" % (len(tracks), len(all_swh)))

    lat_bins = np.arange(0, 52, 2.0)
    clim = np.full(len(lat_bins) - 1, np.nan)
    for j in range(len(lat_bins) - 1):
        m = (all_lat >= lat_bins[j]) & (all_lat < lat_bins[j + 1])
        if m.sum() >= 500:
            clim[j] = np.median(all_swh[m])

    recs = []
    cnt = {"calm": 0, "blue": 0, "yellow": 0, "orange": 0, "red": 0}
    for zp, (lat, lon, swh, tm) in tracks:
        name = os.path.basename(zp).replace(".zip", "")
        short = "%s %s" % (name[:3], name[18:32])
        dist = path_distance(lat, lon)
        L = dist[-1]
        if L <= 0:
            continue

        seg = max(1, args.segment)
        edges = np.linspace(0, L, seg + 1)
        for k in range(seg):
            a, b = edges[k], edges[k + 1]
            m = (dist >= a) & (dist < b)
            if m.sum() < args.min_pts // 2:
                continue
            d = dist[m] - a
            s = swh[m]
            la, lo = lat[m], lon[m]
            mx = float(np.nanmax(s))
            lab = band_of(mx)
            cnt[lab] += 1

            meta = {"short": "%s seg%d/%d" % (short, k + 1, seg)}
            fn1 = "%s_seg%d_hs.png" % (name[18:32], k + 1)
            render(d, s, la, lo, meta, os.path.join(IMG_DIR, fn1), "hs")
            recs.append({"image": fn1, "label": lab, "orbit": name,
                         "segment": k + 1,
                         "product": "intrinsic_hs_profile",
                         "hs_max": round(mx, 2),
                         "hs_median": round(float(np.nanmedian(s)), 2),
                         "lon_range": [round(float(lo.min()), 2),
                                       round(float(lo.max()), 2)],
                         "lat_range": [round(float(la.min()), 2),
                                       round(float(la.max()), 2)],
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT},
                             {"from": "gpt", "value": ANS[lab] +
                              " Peak Hs on this segment is %.1f m." % mx}]})

            # 异常场
            bi = np.clip(((la - lat_bins[0]) / 2.0).astype(int),
                         0, len(clim) - 1)
            base = clim[bi]
            anom = s - base
            if np.isfinite(anom).sum() < 50:
                continue
            fn2 = "%s_seg%d_anom.png" % (name[18:32], k + 1)
            render(d, s, la, lo, meta, os.path.join(IMG_DIR, fn2),
                   "anomaly", anomaly=anom)
            recs.append({"image": fn2, "label": lab, "orbit": name,
                         "segment": k + 1,
                         "product": "feature_hs_anomaly",
                         "hs_max": round(mx, 2),
                         "anom_median": round(float(np.nanmedian(anom)), 2),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT},
                             {"from": "gpt", "value": ANS[lab] +
                              " Median departure from climatology is %+.1f m."
                              % float(np.nanmedian(anom))}]})

    json.dump(recs, open(os.path.join(OUT, "train_wave.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 %d 条" % len(recs))
    print("按最高档分布:", cnt)
    print("图片目录:", IMG_DIR)
    print("台账:", os.path.join(OUT, "train_wave.json"))


if __name__ == "__main__":
    main()

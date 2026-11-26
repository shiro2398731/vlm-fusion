# -*- coding: utf-8 -*-
r"""
地面塌陷 · Sentinel-1 GRD 双时相 Δσ⁰ 变化检测

与上一版的根本区别
------------------
上一版走 SLC + InSAR 解缠，致命问题：
  1. snaphu 反复卡死（4 次，每次十几分钟白跑）
  2. 位移场 37% 是 FLOAT32 填充值（3.4e38），不是物理量
  3. 中位数判据失效，5740 张全判 stable
本版走 GRD 后向散射幅度，不解缠、不碰相位。

物理依据
--------
地面塌陷 vs 地面沉降，分水岭是"变化速率与空间尺度"：
  沉降：每年几毫米到几厘米，几十~几百公里，梯度平缓
  塌陷：几秒到几天，几米~几百米，边界陡峭、坑形规整

SAR 看到的是后向散射 σ⁰（dB）：
  塌陷坑内 → 积水或裸土/碎石 → 镜面反射或漫散射改变 → σ⁰ 显著下降
  （柳州 2026-05 案例：塌陷坑"多呈圆形、椭圆形，形态规整"）

定标（真 σ⁰，非相对强度）
------------------------
  σ⁰ = DN² / (calibrationVector)²        单位：线性功率
  σ⁰[dB] = 10·log10(σ⁰)
calibrationVector 从 annotation/calibration/*-vv-*.xml 读取，
按 (line, pixel) 双线性插值到全图。

判据
----
  Δσ⁰ = σ⁰_后 − σ⁰_前  (dB)
  塌陷 = Δσ⁰ < μ − 3σ（用同期"未变化"区域的 Δσ⁰ 分布定 μ、σ）
         且 |Δσ⁰| > 5 dB（CEOS《SAR 解读指南》C 波段 VV 典型值 −17.5±5 dB
                         的波动范围，超出即"显著变化"）
注意：5 dB 只作"显著变化"的物理下限，塌陷的定义来自 3σ 异常，不来自这个数。

AOI
---
广西桂林临桂主城区：110.20°E, 25.25°N, 半径 0.25°
依据：江思义等，2020，《中国地质灾害与防治学报》
      "近 40 年桂林市及近郊 300 km² 发生 654 个岩溶塌陷"
      "多发生于地下水位波动大的 5~8 月，集中在临桂主城区"

用法
----
  python src/preprocess/grd_collapse_dsigma.py --probe          # 只看量级
  python src/preprocess/grd_collapse_dsigma.py                  # 全量出图
"""
import os
import io
import sys
import re
import json
import glob
import zipfile
import argparse
import collections
import random

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRD_ROOT = r"C:\vlmdisaster\data\raw\05_sar_insar\grd\ground_collapse"
OUT = r"C:\vlmdisaster\datasets\05_sar_insar\ground_collapse"
IMG = os.path.join(OUT, "images")

# AOI（江思义等 2020：临桂主城区）
# 注意：AOI 必须落在两帧重叠区中间。
#   帧 A 覆盖 23.5467~25.4706，帧 B 覆盖 25.0561~26.9766
#   重叠区 25.0561~25.4706（宽 0.41°），中心 25.263
#   半径收至 0.15 留边，否则 LinearNDInterpolator 落在边界会出 NaN
AOI_LON, AOI_LAT, AOI_R = 110.20, 25.26, 0.15
GRID = 512          # AOI 重采样后的网格边长
WIN = 128           # 切窗
STRIDE = 64
OUT_SIZE = 384
SIGMA_K = 3.0       # 3σ 异常
MIN_DB = 5.0        # CEOS：C-VV 典型 −17.5±5 dB，超此即显著变化
VALID_Q = 0.02      # σ⁰ 有效下限（dB），低于此视为无回波


def log(m):
    print(m, flush=True)


def find_in_zip(names, kind, pol="vv"):
    """在 zip 条目里找指定类型的文件"""
    k = kind.lower()
    out = []
    for n in names:
        low = n.lower()
        if k == "tiff":
            if "/measurement/" in low and pol in low and low.endswith(".tiff"):
                out.append(n)
        elif k == "calibration":
            if "/annotation/calibration/" in low and "calibration" in low and pol in low and low.endswith(".xml"):
                out.append(n)
        elif k == "annotation":
            if ("/annotation/" in low and pol in low and low.endswith(".xml")
                    and "calibration" not in low and "noise" not in low and "rfi" not in low):
                out.append(n)
    return out


def parse_calibration(xml_text):
    """解析 calibrationVectorList → (lines, pixels, sigmaNought 二维数组)"""
    vecs = re.findall(r"<calibrationVector>(.*?)</calibrationVector>", xml_text, re.S)
    rows, line_ids = [], []
    pix_axis = None
    for v in vecs:
        m_line = re.search(r"<line>(\d+)</line>", v)
        if not m_line:
            continue
        line_ids.append(int(m_line.group(1)))
        m_pix = re.search(r'<pixel count="\d+">([^<]+)</pixel>', v)
        m_sig = re.search(r'<sigmaNought[^>]*>([^<]+)</sigmaNought>', v)
        if not (m_pix and m_sig):
            continue
        px = np.array([float(x) for x in m_pix.group(1).split()])
        sg = np.array([float(x) for x in m_sig.group(1).split()])
        if pix_axis is None:
            pix_axis = px
        # 对齐到统一像素轴
        n = min(len(pix_axis), len(sg))
        rows.append(sg[:n])
    if not rows:
        return None
    w = min(len(r) for r in rows)
    arr = np.vstack([r[:w] for r in rows])
    return np.array(line_ids[:len(arr)]), pix_axis[:w], arr


def parse_geolocation(xml_text):
    """解析 geolocationGridPoint → (line, pixel, lat, lon)"""
    gps = re.findall(r"<geolocationGridPoint>(.*?)</geolocationGridPoint>", xml_text, re.S)
    out = []
    for g in gps:
        def gv(tag):
            m = re.search(r"<" + tag + r">([^<]+)</" + tag + r">", g)
            return m.group(1) if m else None
        try:
            out.append((int(gv("line")), int(gv("pixel")),
                        float(gv("latitude")), float(gv("longitude"))))
        except (TypeError, ValueError):
            continue
    return out


def bilinear(field_lines, field_pixs, field_vals, line, pix):
    """在 (line, pixel) 上双线性插值标定场"""
    ni = np.clip(np.searchsorted(field_lines, line) - 1, 0, len(field_lines) - 2)
    nj = np.clip(np.searchsorted(field_pixs, pix) - 1, 0, len(field_pixs) - 2)
    y0, y1 = field_lines[ni], field_lines[ni + 1]
    x0, x1 = field_pixs[nj], field_pixs[nj + 1]
    wy = 0.0 if y1 == y0 else (line - y0) / (y1 - y0)
    wx = 0.0 if x1 == x0 else (pix - x0) / (x1 - x0)
    v00 = field_vals[ni, nj]
    v01 = field_vals[ni, nj + 1]
    v10 = field_vals[ni + 1, nj]
    v11 = field_vals[ni + 1, nj + 1]
    return ((1 - wy) * ((1 - wx) * v00 + wx * v01)
            + wy * ((1 - wx) * v10 + wx * v11))


def load_sigma0(zip_path):
    """读一景 GRD → 返回 (sigma0_dB 全图, geoloc 点列表, 全图尺寸)"""
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    tif = find_in_zip(names, "tiff")[0]
    cal = find_in_zip(names, "calibration")[0]
    ann = find_in_zip(names, "annotation")[0]

    im = PILImage.open(io.BytesIO(z.read(tif)))
    dn = np.array(im, dtype="float32")
    im.close()
    nrow, ncol = dn.shape

    cal_txt = z.read(cal).decode("utf-8", errors="replace")
    ann_txt = z.read(ann).decode("utf-8", errors="replace")
    z.close()

    parsed = parse_calibration(cal_txt)
    if parsed is None:
        raise RuntimeError("定标向量解析失败")
    cal_lines, cal_pixs, cal_vals = parsed

    # 标定场按行分块插值（避免整图 4 亿次插值）
    s0 = np.full(dn.shape, np.nan, dtype="float32")
    for i in range(nrow):
        # 该行所在的两个标定行之间插值，先按行得到每列系数
        if cal_lines[0] <= i <= cal_lines[-1]:
            ni = np.clip(np.searchsorted(cal_lines, i) - 1, 0, len(cal_lines) - 2)
            y0, y1 = cal_lines[ni], cal_lines[ni + 1]
            wy = 0.0 if y1 == y0 else (i - y0) / (y1 - y0)
            row_cal = (1 - wy) * cal_vals[ni] + wy * cal_vals[ni + 1]
        else:
            # 超出标定范围，用最近行
            row_cal = cal_vals[0] if i < cal_lines[0] else cal_vals[-1]
        # 像素方向插值
        coef = np.interp(np.arange(ncol), cal_pixs, row_cal)
        s0[i] = (dn[i] ** 2) / np.where(coef > 0, coef, np.nan) ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(s0 > 0, s0, np.nan))
    db[~np.isfinite(db)] = np.nan
    db[db < -45] = np.nan          # 无回波/边框

    geo = parse_geolocation(ann_txt)
    return db, geo, (nrow, ncol)


def crop_aoi(db, geo, lon0, lat0, r, grid=GRID):
    """按经纬度把 AOI 重采样到规则网格（格网向量化反查）

    geolocationGrid 是 lines×pixels 的规则网格（IW GRD 通常 10×21）。
    把 210 个控制点的 (lat, lon) → (line, pixel) 做一次 Delaunay 线性插值，
    向量化处理整块 AOI，能跟随地表起伏与经度方向的非线性，比全局仿射准很多。
    """
    if not geo:
        return None, None, None

    from scipy.interpolate import LinearNDInterpolator

    lats = np.array([g[2] for g in geo], dtype="float64")
    lons = np.array([g[3] for g in geo], dtype="float64")
    lines = np.array([g[0] for g in geo], dtype="float64")
    pixs = np.array([g[1] for g in geo], dtype="float64")

    # 经纬边界检查（避免插太多 NaN）
    if (lon0 - r < lons.min() - 0.01 or lon0 + r > lons.max() + 0.01
            or lat0 - r < lats.min() - 0.01 or lat0 + r > lats.max() + 0.01):
        # AOI 超出该景覆盖，直接返回 None
        return None, None, None

    # 目标经纬网格
    tlon = np.linspace(lon0 - r, lon0 + r, grid)
    tlat = np.linspace(lat0 - r, lat0 + r, grid)
    TLON, TLAT = np.meshgrid(tlon, tlat)

    pts = np.c_[lons, lats]
    f_line = LinearNDInterpolator(pts, lines)
    f_pix = LinearNDInterpolator(pts, pixs)

    tl = np.asarray(f_line(TLON.ravel(), TLAT.ravel())).reshape(grid, grid)
    tp = np.asarray(f_pix(TLON.ravel(), TLAT.ravel())).reshape(grid, grid)

    valid = np.isfinite(tl) & np.isfinite(tp)
    yi = np.where(valid, np.round(tl), 0).astype("int32")
    xi = np.where(valid, np.round(tp), 0).astype("int32")
    np.clip(yi, 0, db.shape[0] - 1, out=yi)
    np.clip(xi, 0, db.shape[1] - 1, out=xi)

    out = np.where(valid, db[yi, xi], np.nan).astype("float32")
    return out, TLON, TLAT


def two_dates(zip_a, zip_b):
    """同一轨道相邻日期两景 → Δσ⁰"""
    db_a, geo_a, _ = load_sigma0(zip_a)
    db_b, geo_b, _ = load_sigma0(zip_b)
    a, _, _ = crop_aoi(db_a, geo_a, AOI_LON, AOI_LAT, AOI_R)
    b, _, _ = crop_aoi(db_b, geo_b, AOI_LON, AOI_LAT, AOI_R)
    if a is None or b is None:
        return None
    with np.errstate(invalid="ignore"):
        d = b - a
    d[~np.isfinite(d)] = np.nan
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只看 Δσ⁰ 量级，不出图")
    args = ap.parse_args()

    zips = sorted(glob.glob(os.path.join(GRD_ROOT, "*.zip")))
    log("GRD 文件: %d 景" % len(zips))
    if len(zips) < 2:
        log("不足两景")
        return

    # 按日期分组：同一日期的两帧是相邻帧，取与 AOI 重叠更多的那一帧
    by_day = collections.defaultdict(list)
    for z in zips:
        m = re.search(r"_(\d{8})T\d{6}_", os.path.basename(z))
        if m:
            by_day[m.group(1)].append(z)
    days = sorted(by_day)
    log("日期: %d 个 → %s" % (len(days), ", ".join(days)))

    if args.probe:
        za = by_day[days[0]][0]
        zb = by_day[days[1]][0]
        log("\n探测 %s → %s" % (days[0], days[1]))
        d = two_dates(za, zb)
        if d is None:
            log("无法配准")
            return
        v = d[np.isfinite(d)]
        log("  AOI 网格 %dx%d" % d.shape)
        log("  有效像元 %.1f%%" % (v.size / d.size * 100))
        log("  Δσ⁰: min %.2f  1%% %.2f  50%% %.2f  99%% %.2f  max %.2f  dB"
            % (np.min(v), np.percentile(v, 1), np.median(v),
               np.percentile(v, 99), np.max(v)))
        log("  std %.2f dB" % np.std(v))
        log("  超出 ±5 dB 的像元 %.2f%%" % (np.mean(np.abs(v) > MIN_DB) * 100))
        return

    # 全量：相邻日期成对
    os.makedirs(IMG, exist_ok=True)
    records = []
    pairs_done = 0
    for i in range(len(days) - 1):
        d0, d1 = days[i], days[i + 1]
        za = by_day[d0][0]
        zb = by_day[d1][0]
        log("\n[%d/%d] %s → %s" % (i + 1, len(days) - 1, d0, d1))
        try:
            d = two_dates(za, zb)
        except Exception as e:
            log("  失败: %s" % str(e)[:80])
            continue
        if d is None:
            log("  配准失败")
            continue

        v = d[np.isfinite(d)]
        if v.size < 1000:
            log("  有效像元不足")
            continue
        mu, sd = float(np.median(v)), float(np.std(v))
        thr = mu - SIGMA_K * sd
        log("  μ=%.2f  σ=%.2f  σ异常阈值=%.2f dB" % (mu, sd, thr))

        for y in range(0, max(1, d.shape[0] - WIN + 1), STRIDE):
            for x in range(0, max(1, d.shape[1] - WIN + 1), STRIDE):
                sub = d[y:y + WIN, x:x + WIN]
                sv = sub[np.isfinite(sub)]
                if sv.size < WIN * WIN * 0.5:
                    continue
                # 塌陷判据：显著下降（3σ 异常且超 5dB）
                frac_drop = float(np.mean((sub < thr) & (sub < -MIN_DB)))
                frac_rise = float(np.mean((sub > -thr) & (sub > MIN_DB)))
                if frac_drop >= 0.10:
                    lab = "collapse"
                elif frac_drop >= 0.02:
                    lab = "possible_collapse"
                else:
                    lab = "stable"
                records.append({"pair": "%s_%s" % (d0, d1), "y": y, "x": x,
                                "label": lab, "frac_drop": round(frac_drop, 4),
                                "frac_rise": round(frac_rise, 4),
                                "_sub": sub})
        pairs_done += 1

    log("\n成对处理: %d / %d" % (pairs_done, len(days) - 1))
    log("标签分布:", dict(collections.Counter(r["label"] for r in records)))

    # 出图（双产品）
    log("\n渲染双产品 ...")
    # 计算 σ⁰ 的显示范围（用所有 sub 的 2/98 分位）
    allv = np.concatenate([r["_sub"][np.isfinite(r["_sub"])].ravel() for r in records[:200]])
    vmax = max(5.0, float(np.percentile(np.abs(allv), 98)))

    out_recs = []
    # 两产品：特色场 Δσ⁰（变化）、本征场 σ⁰（后景绝对后向散射）
    for r in records:
        sub = r["_sub"]
        fn_d = "gcl_%s_%d_%d_dsigma.png" % (r["pair"], r["y"], r["x"])
        p_d = os.path.join(IMG, fn_d)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
        im = ax.imshow(sub, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("%s  %s" % (r["pair"], r["label"]), fontsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Δσ⁰ (dB)", fontsize=7)
        fig.tight_layout(); fig.savefig(p_d, dpi=96); plt.close(fig)
        PILImage.open(p_d).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(p_d)

        out_recs.append({
            "image": fn_d, "pair": r["pair"], "label": r["label"],
            "product": "feature_dsigma",
            "frac_drop": r["frac_drop"], "frac_rise": r["frac_rise"],
            "conversations": [
                {"from": "human",
                 "value": "<image>This is a two-date Sentinel-1 SAR backscatter change map "
                          "(Δσ⁰, dB) over a karst region. Blue = backscatter decreased "
                          "(possible collapse or waterlogging), red = increased. "
                          "Is there a localized collapse signature in this window? Answer Yes or No."},
                {"from": "gpt",
                 "value": {"collapse": "Yes, there is a localized collapse signature.",
                           "possible_collapse": "There is a weak localized change, possibly a collapse precursor.",
                           "stable": "No, this window shows no localized collapse signature."}[r["label"]]},
            ]})

    json.dump(out_recs, open(os.path.join(OUT, "train_ground_collapse.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("已写 %d 条 → %s" % (len(out_recs), OUT))
    log("注意：σ⁰ 本征场待补（需再存一份后景 σ⁰ 窗）")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
海冰 × SAR（Sentinel-1 GRD VV）切窗 + 结构化 JSON（新元素：seaice_sar）

物理链路
--------
  Sentinel-1 C 波段 SAR → 后向散射系数 σ0（dB）
  冰面粗糙、后向散射强；开放水面镜面反射、后向散射弱
真值
----
  NSIDC 海冰密集度 CDR（25km 日产品）；WMO 判据：密集度 ≥ 15% 为冰区
任务
----
  detected：窗口海冰密集度是否 ≥ 15%（真值）
  数值字段：mean_sigma0_db（只能从图上读）
  severity：none<0.15 / light 0.15-0.40 / moderate 0.40-0.70 / heavy ≥0.70（密集度分数）

用法
----
  .\venv\Scripts\python.exe src\preprocess\seaice_sar_render.py --limit-scenes 1
  .\venv\Scripts\python.exe src\preprocess\seaice_sar_render.py
"""
import argparse
import collections
import datetime
import glob
import io
import json
import os
import random
import re
import sys
import tempfile
import zipfile

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from PIL import Image as PILImage
import tifffile

ROOT = r"C:\vlmdisaster"
SAR_DIR = os.path.join(ROOT, "data", "raw", "05_sar_insar", "grd")
ICE_NC = os.path.join(ROOT, "data", "raw", "04_passive_microwave", "ice_conc",
                      "nh_ice_conc_2020-12_2021-03.nc")
LATLON_NC = os.path.join(ROOT, "data", "raw", "04_passive_microwave", "ice_conc",
                         "nh_grid_latlon.nc")
OUT = os.path.join(ROOT, "datasets", "05_sar_insar", "seaice_sar")
IMG = os.path.join(OUT, "images")

WIN = 512
STRIDE = 512
POS_PER_SCENE = 30
NEG_PER_SCENE = 30
SEED = 42
OUT_SIZE = 384
ICE_MIN = 0.15
LATB = (37.0, 42.0)
LONB = (117.0, 124.0)

PROMPT = ("This is a Sentinel-1 SAR backscatter image (VV, dB) over a sea-ice region. "
          "Ice surfaces are brighter (higher backscatter) than open water. Output a JSON "
          "object with fields: disaster_type, detected (true/false), source, "
          "physical_quantity, severity, mean_sigma0_db, criterion. severity must be one "
          "of these exact words: none / light / moderate / heavy.")


def severity_of_db(v):
    """severity 用图上的后向散射分档（不依赖外部真值）"""
    if v < -16.0:
        return "none"
    if v < -14.0:
        return "light"
    if v < -12.0:
        return "moderate"
    return "heavy"


def scene_date(zip_path):
    m = re.search(r"_(\d{8})T", os.path.basename(zip_path))
    return datetime.date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:]))


def parse_geo(xml_text):
    pts = re.findall(
        r"<line>(\d+)</line>\s*<pixel>(\d+)</pixel>\s*<latitude>([-\d.eE+]+)</latitude>"
        r"\s*<longitude>([-\d.eE+]+)</longitude>", xml_text)
    a = np.array([[int(p[0]), int(p[1]), float(p[2]), float(p[3])] for p in pts])
    lines = np.unique(a[:, 0])
    pixels = np.unique(a[:, 1])
    lat = np.full((len(lines), len(pixels)), np.nan)
    lon = np.full((len(lines), len(pixels)), np.nan)
    li = {v: i for i, v in enumerate(lines)}
    pi = {v: i for i, v in enumerate(pixels)}
    for (l, p, la, lo) in a:
        lat[li[l], pi[p]] = la
        lon[li[l], pi[p]] = lo
    return lines.astype("float64"), pixels.astype("float64"), lat, lon


def parse_calib(xml_text):
    vec_lines, vec_pix, vec_lut = [], [], []
    for m in re.finditer(r"<calibrationVector>(.*?)</calibrationVector>", xml_text, re.S):
        blk = m.group(1)
        line = int(re.search(r"<line>(\d+)</line>", blk).group(1))
        pix = np.array([float(x) for x in re.search(r"<pixel[^>]*>(.*?)</pixel>", blk, re.S).group(1).split()])
        lut = np.array([float(x) for x in re.search(r"<sigmaNought[^>]*>(.*?)</sigmaNought>", blk, re.S).group(1).split()])
        vec_lines.append(line)
        vec_pix.append(pix)
        vec_lut.append(lut)
    return np.array(vec_lines, dtype="float64"), vec_pix, vec_lut


def sigma0_db(dn, calib, r0, r1, x0, x1):
    """对 crop 区域算 dB（DN²/A²，A 由最近标定向量的 LUT 沿距离向插值）"""
    lines, pix_list, lut_list = calib
    h, w = dn.shape
    x = np.arange(x0, x1, dtype="float64")
    # 找最近的两条向量，按行插值
    idx = np.searchsorted(lines, [r0, r1])
    idx = [int(np.clip(i, 0, len(lines) - 1)) for i in idx]
    luts = []
    for i in idx:
        luts.append(np.interp(x, pix_list[i], lut_list[i]))
    lut = np.stack(luts)  # (2, w)
    rows = np.arange(r0, r1, dtype="float64")
    la = lines[idx[0]]
    lb = lines[idx[0] if idx[0] == idx[1] else idx[1]]
    t = 0.0 if lb == la else np.clip((rows - la) / (lb - la), 0, 1)[:, None]
    A = luts[0][None, :] * (1 - t) + luts[1][None, :] * t
    sub = dn[r0:r1, x0:x1].astype("float32")
    sigma0 = (sub * sub) / np.maximum(A, 1e-6) ** 2
    return 10.0 * np.log10(np.maximum(sigma0, 1e-6))


def load_ice():
    ds = Dataset(ICE_NC)
    conc = np.ma.filled(np.array(ds.variables["cdr_seaice_conc"][:], dtype="float32"), 2.54)
    times = np.array(ds.variables["time"][:], dtype="float64")
    units = ds.variables["time"].units
    ds.close()
    ds = Dataset(LATLON_NC)
    la = np.array(ds.variables["latitude"][:])
    lo = np.array(ds.variables["longitude"][:])
    ds.close()
    conc[conc > 1.5] = np.nan
    conc[conc < 0] = np.nan
    base = datetime.datetime(1970, 1, 1)
    dates = [(base + datetime.timedelta(seconds=float(t))).date() for t in times]
    return conc, dates, la, lo


def ice_at(la, lo, a, la_g, lo_g):
    """在 2D 浓度切片 a 上找最近的有效格点（跳过陆地 NaN）"""
    d = (la_g - la) ** 2 + ((lo_g - lo) * np.cos(np.deg2rad(la))) ** 2
    dd = np.where(np.isfinite(a), d, np.inf)
    if not np.isfinite(dd).any():
        return None
    i = int(np.argmin(dd))
    r, c = np.unravel_index(i, dd.shape)
    if not np.isfinite(a[r, c]):
        return None
    return r, c


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="gray", vmin=-20, vmax=-8, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("sigma0 (dB)", fontsize=7)
    fig.text(0.04, 0.075, "Sentinel-1 GRD VV (calibrated)", fontsize=6)
    fig.text(0.04, 0.035, "bright = ice/rough surface", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-scenes", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    conc, dates, la_g, lo_g = load_ice()
    print("NSIDC 浓度 %s，日期 %s ~ %s" % (conc.shape, dates[0], dates[-1]))

    zips = sorted(f for f in glob.glob(os.path.join(SAR_DIR, "S1*.zip"))
                  if "GRDH" in os.path.basename(f))
    print("SAR 场景 %d 个" % len(zips))
    if a.limit_scenes:
        zips = zips[:a.limit_scenes]

    records = []
    random.seed(SEED)
    for zi, z in enumerate(zips, 1):
        d = scene_date(z)
        if d not in dates:
            print("  跳过（无对应浓度日）", d)
            continue
        ti = dates.index(d)
        a_slice = conc[ti]
        zf = zipfile.ZipFile(z)
        names = zf.namelist()
        ann = [n for n in names if n.endswith(".xml") and "-vv-" in n
               and "calibration" not in n and "noise" not in n and "annotation/" in n][0]
        cal = [n for n in names if "calibration-s1" in n and "-vv-" in n][0]
        tif = [n for n in names if n.endswith(".tiff") and "-vv-" in n][0]
        geo = parse_geo(zf.read(ann).decode("utf-8", "replace"))
        calib = parse_calib(zf.read(cal).decode("utf-8", "replace"))
        tmp = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
        tmp.write(zf.read(tif))
        tmp.close()
        dn = tifffile.imread(tmp.name)
        h, w = dn.shape
        lines_g, pixels_g, latg, longg = geo
        # 找渤海范围的 crop（按格点包围盒）
        sel = (latg > LATB[0]) & (latg < LATB[1]) & (longg > LONB[0]) & (longg < LONB[1])
        if sel.sum() < 3:
            os.unlink(tmp.name)
            print("  跳过（格点不在渤海）", z)
            continue
        r0 = int(max(0, lines_g[0] if sel[0, :].any() else 0) - 200)
        rows_idx = np.where(sel.any(axis=1))[0]
        cols_idx = np.where(sel.any(axis=0))[0]
        r0 = int(max(0, lines_g[rows_idx[0]] - 300))
        r1 = int(min(h, lines_g[rows_idx[-1]] + 300))
        x0 = int(max(0, pixels_g[cols_idx[0]] - 300))
        x1 = int(min(w, pixels_g[cols_idx[-1]] + 300))
        db = sigma0_db(dn, calib, r0, r1, x0, x1)
        os.unlink(tmp.name)

        # 每个窗口中心的经纬度：用格点网格线性映射（行/列 -> 格点索引 -> 双线性）
        def latlon_of(rr, cc):
            # 归一到格点坐标
            fr = np.interp(rr, lines_g, np.arange(len(lines_g)))
            fc = np.interp(cc, pixels_g, np.arange(len(pixels_g)))
            i0 = int(np.clip(np.floor(fr), 0, len(lines_g) - 2))
            j0 = int(np.clip(np.floor(fc), 0, len(pixels_g) - 2))
            tr = fr - i0
            tc = fc - j0
            la = (latg[i0, j0] * (1 - tr) * (1 - tc) + latg[i0 + 1, j0] * tr * (1 - tc) +
                  latg[i0, j0 + 1] * (1 - tr) * tc + latg[i0 + 1, j0 + 1] * tr * tc)
            lo = (longg[i0, j0] * (1 - tr) * (1 - tc) + longg[i0 + 1, j0] * tr * (1 - tc) +
                  longg[i0, j0 + 1] * (1 - tr) * tc + longg[i0 + 1, j0 + 1] * tr * tc)
            return float(la), float(lo)

        pos, neg = [], []
        for r in range(0, db.shape[0] - WIN + 1, STRIDE):
            for c in range(0, db.shape[1] - WIN + 1, STRIDE):
                win = db[r:r + WIN, c:c + WIN]
                if np.isfinite(win).mean() < 0.5:
                    continue
                la, lo = latlon_of(r0 + r + WIN // 2, x0 + c + WIN // 2)
                if not (LATB[0] < la < LATB[1] and LONB[0] < lo < LONB[1]):
                    continue
                hit = ice_at(la, lo, a_slice, la_g, lo_g)
                if hit is None:
                    continue
                ri, ci = hit
                cv = float(a_slice[ri, ci])
                if not np.isfinite(cv):
                    continue
                item = (r, c, la, lo, float(np.nanmean(win)), cv)
                (pos if cv >= ICE_MIN else neg).append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        stamp = d.strftime("%Y%m%d")
        for det, (r, c, la, lo, mdb, cv) in [(True, x) for x in pos[:POS_PER_SCENE]] + \
                [(False, x) for x in neg[:NEG_PER_SCENE]]:
            sev = severity_of_db(mdb)
            nm = "seaice_%s_r%d_c%d.png" % (stamp, r0 + r, x0 + c)
            p = os.path.join(IMG, nm)
            render(db[r:r + WIN, c:c + WIN], p, stamp, la, lo)
            ans = {
                "disaster_type": "sea_ice",
                "detected": bool(det),
                "source": "SAR",
                "physical_quantity": "sigma0_db",
                "severity": sev,
                "mean_sigma0_db": round(mdb, 1),
                "criterion": ("ice concentration from NSIDC CDR; WMO 15% defines the ice zone; "
                              "backscatter calibrated from S1 GRD VV"),
            }
            records.append({
                "image": p, "label": sev, "stamp": stamp, "mean_sigma0_db": round(mdb, 1),
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        print("  [%d/%d] %s  冰窗 %d 水窗 %d  累计 %d" % (zi, len(zips), stamp, len(pos), len(neg), len(records)))

    seen = set()
    uniq = []
    for r in records:
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        uniq.append(r)
    random.seed(SEED)
    by = collections.defaultdict(list)
    for r in uniq:
        by[r["label"]].append(r)
    val, train = [], []
    for k in sorted(by):
        lst = by[k][:]
        random.shuffle(lst)
        n = min(max(2, int(round(len(lst) * 0.2))), 30, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_seaice_sar_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_seaice_sar_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()

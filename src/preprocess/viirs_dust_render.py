# -*- coding: utf-8 -*-
r"""
光学沙尘暴 · VIIRS VJ102MOD 渲染（源 1 光学 · 沙尘暴）—— 网格采样版

为什么重做
----------
  2026-10-29 版每景只切一个窗口（取目标区中心），36 景只出了 32 张，
  dust 仅 6 张，样本严重不足。
  根因不在判据，在采样密度。本版改为在目标区内按网格铺窗口，
  每景独立判定 4~10 个窗口。判据保持不变。

路线说明（为什么不用亮温差判据）
------------------------------
  最初想用 11-12 μm 亮温差为负 + 1.6 μm 反射率低这套经典双通道判据
  （Li et al. 2021 综述 / NOAA NESDIS ATBD）。在 2023-03 这场沙尘暴上实测：
      框内（华北）候选像元 36.7%   框外 57.4%    ← 框外反而更高
  判据失效。原因是该判据在业务中始终与大气订正/AOD 产品配合使用，
  裸用两个原始通道会把干燥裸地、薄卷云、斜视路径一并判为沙尘。

  故改走可见光形态判识：沙尘羽流在真彩色上呈黄褐色、边界弥散。

判据（形态性，不给数值等级）
--------------------------
  dust_frac = 窗口内 M10(1.6um) 反射率 < 0.15 的像元占比
  dust_frac >= 0.25 → dust，否则 clear

  这是弱监督标签，不是国标等级。GB/T 20480-2025《沙尘天气等级》
  以水平能见度分级，卫星测不到能见度，图上明标"形态判读，非能见度等级"。

双产品
------
  本征场 = 真彩色（M05/M04/M03，Rayleigh + 太阳天顶角订正）
  特色场 = 沙尘增强合成（M10-M05 亮度差 / M04:M05 比值 / M05）

用法
----
  python src/preprocess/viirs_dust_render.py --keep 2     # 试渲 2 景
  python src/preprocess/viirs_dust_render.py              # 全跑
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

ROOT = r"C:\vlmdisaster\data\raw\01_optical_vnir\viirs_dust"
L1B = os.path.join(ROOT, "l1b")
GEO = os.path.join(ROOT, "geo")
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\sand_dust"
IMG = os.path.join(OUT, "images")

# 目标区：华北 + 沙源（与下载脚本一致）
TARGET = (105.0, 125.0, 30.0, 45.0)
WIN = 384
STRIDE = 192
OUT_SIZE = 384
VAL_PER_GROUP = 40
MAX_WIN_PER_SCENE = 24
MIN_VALID = 0.30          # 窗口内有效像元下限
DUST_FRAC = 0.25          # 沙尘判据阈值（与旧版一致，未放宽）


def find_geo(l1b_name):
    parts = l1b_name.split(".")
    stamp = parts[1] + "." + parts[2]
    g = glob.glob(os.path.join(GEO, "VJ103MOD.%s*.nc" % stamp))
    return g[0] if g else None


def load(l1b_path, geo_path):
    with h5py.File(l1b_path, "r") as f:
        od = f["observation_data"]
        ch, sf, off, fv = {}, {}, {}, {}
        for k in ("M03", "M04", "M05", "M10"):
            ds = od[k]
            ch[k] = np.array(ds[:], dtype="uint16")
            sf[k] = float(np.atleast_1d(ds.attrs.get("scale_factor", 1.0))[0])
            off[k] = float(np.atleast_1d(ds.attrs.get("add_offset", 0.0))[0])
            fv[k] = int(np.atleast_1d(ds.attrs.get("_FillValue", 65535))[0])

    with h5py.File(geo_path, "r") as f:
        g = f["geolocation_data"]

        def rg(k):
            ds = g[k]
            s = float(np.atleast_1d(ds.attrs.get("scale_factor", 1.0))[0])
            o = float(np.atleast_1d(ds.attrs.get("add_offset", 0.0))[0])
            return np.array(ds[:], dtype="float32") * s + o

        lat = rg("latitude")
        lon = rg("longitude")
        sza = rg("solar_zenith")
        lwm = np.array(g["land_water_mask"][:], dtype="uint8")

    h = min(ch["M03"].shape[0], lat.shape[0])
    w = min(ch["M03"].shape[1], lat.shape[1])
    data = {k: ch[k][:h, :w].astype("float32") * sf[k] + off[k] for k in ch}
    valid = np.ones((h, w), dtype=bool)
    for k in ch:
        valid &= ch[k][:h, :w] != fv[k]
    valid &= np.isfinite(lat[:h, :w]) & (np.abs(lat[:h, :w]) <= 90)
    valid &= sza[:h, :w] < 88

    return data, valid, lat[:h, :w], lon[:h, :w], sza[:h, :w], lwm[:h, :w]


def to_rgb(d, valid, sza):
    """真彩色：Rayleigh 订正 + 太阳天顶角订正，再拉伸到 0~1"""
    cosz = np.cos(np.deg2rad(np.clip(sza, 0, 88)))
    cosz = np.maximum(cosz, 0.15)
    RAY = {"M05": 0.0, "M04": 0.055, "M03": 0.105}

    chans = []
    for k in ("M05", "M04", "M03"):
        v = d[k] / cosz - RAY[k]
        v = np.where(valid, v, np.nan)
        chans.append(v)
    stack = np.dstack(chans)

    good = stack[np.isfinite(stack)]
    hi = np.percentile(good, 99) if good.size else 1.0
    lo = np.percentile(good, 2) if good.size else 0.0
    hi = max(hi, lo + 1e-3)
    out = np.clip((np.nan_to_num(stack, nan=lo) - lo) / (hi - lo), 0, 1)
    out = np.power(out, 0.85)
    return (out * 255).astype("uint8")


def to_dust_rgb(d, valid):
    """沙尘增强 RGB（EUMETSAT Dust RGB 思路的 VIIRS 近似），定性用"""
    r = d["M10"] - d["M05"]
    g = np.divide(d["M04"], np.maximum(d["M05"], 1e-6))
    b = d["M05"]
    for a in (r, g, b):
        a[~valid] = np.nan

    def nz(a, lo_p=2, hi_p=98):
        v = a[np.isfinite(a)]
        if v.size < 100:
            return np.zeros_like(a)
        lo, hi = np.percentile(v, lo_p), np.percentile(v, hi_p)
        if hi - lo < 1e-6:
            return np.zeros_like(a)
        return np.clip(np.nan_to_num((a - lo) / (hi - lo), nan=0.0), 0, 1)

    out = np.dstack([1.0 - nz(r), nz(g), nz(b)])
    return (out * 255).astype("uint8")


def windows(valid, lat, lon):
    """在目标区内按网格铺窗口，返回 [(y0, x0), ...]"""
    lon0, lon1, lat0, lat1 = TARGET
    inb = valid & (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    if inb.sum() < WIN * WIN * 0.05:
        return []

    H, W = valid.shape
    ys, xs = np.nonzero(inb)
    y_lo, y_hi = int(ys.min()), int(ys.max())
    x_lo, x_hi = int(xs.min()), int(xs.max())

    out = []
    for y0 in range(max(0, y_lo - WIN // 4), min(H - WIN, y_hi), STRIDE):
        for x0 in range(max(0, x_lo - WIN // 4), min(W - WIN, x_hi), STRIDE):
            sub = inb[y0:y0 + WIN, x0:x0 + WIN]
            if sub.size == 0:
                continue
            if sub.mean() < MIN_VALID:
                continue
            out.append((y0, x0))

    if not out:
        return []
    # 均匀抽稀，避免单调区域刷屏
    if len(out) > MAX_WIN_PER_SCENE:
        step = len(out) / float(MAX_WIN_PER_SCENE)
        out = [out[int(i * step)] for i in range(MAX_WIN_PER_SCENE)]
    return out


PROMPT_RGB = ("This is a VIIRS true-colour image (M05 red, M04 green, M03 blue, "
              "750 m) over northern China during a dust event. Describe what you see: "
              "whether a yellowish-brown dust plume is present, how it obscures the "
              "surface, and how it differs from clouds.")
PROMPT_DUST = ("This is a VIIRS dust-enhancement composite over northern China. "
               "Describe the spatial structure: where the dust layer is, its extent, "
               "and how it is separated from cloud.")

ANS = {
    "dust": "A yellowish-brown dust plume covers the surface with soft, diffuse edges; "
            "terrain outlines remain faintly visible through it and there is no cloud "
            "shadow, which distinguishes it from cloud.",
    "clear": "The surface is visible with normal contrast and no yellowish veil. "
             "No dust plume is evident in this window.",
}


def render(rgb, path, tag, lab, kind):
    fig = plt.figure(figsize=(4.6, 4.9), dpi=100)
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.78])
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("%s  %s" % (tag, "true colour" if kind == "tc" else "dust composite"),
                 fontsize=8)
    if kind == "tc":
        fig.text(0.03, 0.095,
                 "Bands: M05 0.67um (R) / M04 0.55um (G) / M03 0.48um (B)",
                 fontsize=6.2)
    else:
        fig.text(0.03, 0.095,
                 "Composite for dust enhancement (qualitative, not a physical index)",
                 fontsize=6.2)
    fig.text(0.03, 0.055, "Sensor: VIIRS/JPSS-1 VJ102MOD, 750 m", fontsize=6.2)
    fig.text(0.03, 0.018,
             "Morphological reading: %s (not a visibility class)" % lab,
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE),
                                              PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0,
                    help="只渲染前 N 景（快速试渲）")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(L1B, "*.nc")))
    if args.keep:
        fs = fs[:args.keep]
    print("L1B 待渲染: %d 景" % len(fs))

    recs = []
    counter = collections.Counter()
    n_win_total = 0

    for i, fp in enumerate(fs):
        name = os.path.basename(fp)
        stamp = ".".join(name.split(".")[1:3])
        gp = find_geo(name)
        if not gp:
            print("  %s 无地理定位" % stamp)
            continue
        try:
            d, valid, lat, lon, sza, lwm = load(fp, gp)
        except Exception as e:
            print("  %s 读取失败 %s" % (stamp, str(e)[:70]))
            continue

        wins = windows(valid, lat, lon)
        if not wins:
            print("  %s 目标区内有效像元不足" % stamp)
            continue

        got_d = got_c = 0
        for (y0, x0) in wins:
            sl = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))
            d_w = {k: v[sl] for k, v in d.items()}
            v_w = valid[sl]
            s_w = sza[sl]
            if v_w.size == 0 or v_w.mean() < MIN_VALID:
                continue

            r16 = d_w["M10"]
            good16 = r16[v_w]
            dust_frac = float(np.mean(good16 < 0.15)) if good16.size else 0.0
            lab = "dust" if dust_frac >= DUST_FRAC else "clear"
            if lab == "dust":
                got_d += 1
            else:
                got_c += 1
            n_win_total += 1

            tc = to_rgb(d_w, v_w, s_w)
            ct = to_dust_rgb(d_w, v_w)
            tag = "%s_%d_%d" % (stamp, y0, x0)

            p1 = os.path.join(IMG, "%s_tc.png" % tag)
            render(tc, p1, tag, lab, "tc")
            recs.append({"image": os.path.basename(p1), "stem": tag, "label": lab,
                         "product": "intrinsic_true_colour",
                         "dust_frac": round(dust_frac, 4),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_RGB},
                             {"from": "gpt", "value": ANS[lab]}]})

            p2 = os.path.join(IMG, "%s_dust.png" % tag)
            render(ct, p2, tag, lab, "dust")
            recs.append({"image": os.path.basename(p2), "stem": tag, "label": lab,
                         "product": "intrinsic_dust_composite",
                         "dust_frac": round(dust_frac, 4),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_DUST},
                             {"from": "gpt", "value": ANS[lab]}]})

            counter[lab] += 1

        print("  [%d/%d] %s  窗口 %d  dust %d / clear %d  累计样本 %d"
              % (i + 1, len(fs), stamp, len(wins), got_d, got_c, len(recs)))

    if not recs:
        print("无样本")
        return

    random.seed(53)
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

    json.dump(train, open(os.path.join(OUT, "train_sand_dust.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_sand_dust_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n=== 结果 ===")
    print("窗口总数 %d   样本 train %d | val %d" % (n_win_total, len(train), len(val)))
    print("标签分布:", dict(counter))
    for k in sorted(by):
        print("   %-30s %-8s %d" % (k[0], k[1], len(by[k])))
    print("输出:", IMG)


if __name__ == "__main__":
    main()

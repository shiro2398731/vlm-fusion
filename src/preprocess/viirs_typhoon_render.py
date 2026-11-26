# -*- coding: utf-8 -*-
r"""
光学台风 · VIIRS VJ102IMG 渲染（源 1 光学 · 飓风/台风）

为什么是 VIIRS
--------------
  JPSS-1 VIIRS 幅宽约 3000 km，特大台风（直径 1000+ km）能完整覆盖；
  MODIS 幅宽 2330 km 会切边。I 波段 375 m，足以看清眼壁与螺旋云带。

数据结构（已实测）
----------------
  VJ102IMG 为 L1B swath，只含扫描时间 + 辐射率，**无经纬度**。
  observation_data 有 I01~I05 五个波段：
    I01  0.64 μm  可见光红
    I02  0.865 μm 近红外
    I03  1.61 μm  短波红外
    I04  3.74 μm  中红外  （带 brightness_temperature_lut）
    I05  11.45 μm 热红外  （带 brightness_temperature_lut）
  I01~I03 为反射率类，attrs 给 scale_factor；I04/I05 为辐射率，需查 LUT 得亮温。

没有经纬度怎么定位
------------------
  不依赖地理定位。66 条均按台风海域 bounding_box 检索，每条都覆盖台风区。
  台风中心直接用**云顶最冷区**的重心定位 —— 比按经纬度切更贴物理，
  与国际 Dvorak 技术"按云型定强度"的思路一致（但本脚本不声称给出 T 数）。

双产品
------
  本征场 = I05 亮温场（K）—— 热红外本征物理量，云顶温度
  特色场 = I1/I2/I3 假彩色 —— 云系结构可视化

判据（云顶亮温，卫星气象通行分级）
--------------------------------
  >= 260 K     普通云
  220 ~ 260 K  对流云
  200 ~ 220 K  强对流
  < 200 K      眼壁 / 深对流 —— 成熟台风的核心特征
  分级取窗口内"<=200 K 像元占比"：
      >= 8%   mature       成熟台风（有眼壁深对流）
      2~8%    developing   发展期（强对流成片）
      < 2%    weak         弱系统 / 非台风云系

窗口
----
  1024×1024 原始像元（375 m → 约 384 km 见方），缩到 384 px 输出。
  每景取 1 个以最冷区为中心的窗口 + 2 个偏移窗口（制造同一景内的
  强对流 / 弱对流对比，天然的正负样本）。

用法
----
  python src/preprocess/viirs_typhoon_render.py --keep 2
  python src/preprocess/viirs_typhoon_render.py
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

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\viirs_typhoon"
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\hurricane"
IMG = os.path.join(OUT, "images")

WIN = 1024              # 原始像元窗口
OUT_SIZE = 384
VAL_PER_GROUP = 30
FILL = 65535

# 云顶亮温分级（K）
T_COLD = 200.0          # 眼壁/深对流
T_CONV = 220.0          # 强对流
T_CLOUD = 260.0         # 普通云

# 分级：窗口内 <=T_COLD 像元占比（经验阈值，由 2023 年三场台风实测分布定）
# 实测：眼壁窗口 1.47%，全图平均 0.35~1.0%
G_MATURE = 0.010
G_DEVELOPING = 0.003

PROMPT_BT = ("This is a VIIRS 11.45 micrometre brightness temperature field (kelvin, 375 m) "
             "over a tropical cyclone. Cold values indicate high, deep convective cloud tops. "
             "Describe the cloud-top temperature structure.")
PROMPT_RGB = ("This is a VIIRS false-colour composite (I1 red, I2 green, I3 blue) of a "
              "tropical cyclone. Describe the cloud pattern: spiral bands, central dense "
              "overcast, and eye structure.")

ANS = {
    "mature": "The window shows a mature tropical cyclone: a cold central dense overcast "
              "with a large fraction of cloud tops below 200 K, indicating an eyewall of "
              "deep convection.",
    "developing": "The window shows organized deep convection with a moderate fraction of "
                  "cloud tops below 200 K. The system appears to be developing.",
    "weak": "The window shows ordinary cloud cover with few very cold tops; no organized "
            "tropical cyclone structure is evident.",
}


def _lut_bt(dn, lut):
    """DN → 亮温。LUT 末尾有 8 个 -999.9 无效槽，一并剔掉"""
    dn = np.clip(dn, 0, len(lut) - 1)
    bt = lut[dn]
    bt[bt < 100.0] = np.nan      # -999.9 等无效值
    return bt


def load_scene(fp):
    """返回 (bt_K, rgb_uint8)  —— bt 为 I05 亮温（K），rgb 为假彩色"""
    with h5py.File(fp, "r") as f:
        od = f["observation_data"]
        i05 = np.array(od["I05"][:], dtype="uint16")
        lut = np.array(od["I05_brightness_temperature_lut"][:], dtype="float32")
        bt = _lut_bt(i05, lut)
        bt[i05 == FILL] = np.nan
        # I1/I2/I3 反射率（scale_factor 定标）
        chans = []
        for k in ("I01", "I02", "I03"):
            raw = np.array(od[k][:], dtype="float32")
            sf = od[k].attrs.get("scale_factor", 1.0)
            sf = float(np.atleast_1d(sf)[0])      # netCDF4 存成 shape=(1,) 的数组
            v = raw * sf
            v[raw == FILL] = np.nan
            chans.append(v)

    r, g, b = chans
    # 反射率 → 0~1 的显示拉伸（经验值：0~0.6 映射到 0~1）
    def stretch(a, lo=0.0, hi=0.6):
        x = (a - lo) / (hi - lo)
        return np.clip(np.nan_to_num(x, nan=0.0), 0, 1)

    rgb = np.dstack([stretch(r), stretch(g), stretch(b)])
    return bt, (rgb * 255).astype("uint8")


def find_cold_center(bt, thr=T_CONV):
    """云顶最冷区重心（行, 列）"""
    m = np.isfinite(bt) & (bt <= thr)
    if m.sum() < 200:
        m = np.isfinite(bt) & (bt <= np.nanpercentile(bt, 2))
    if m.sum() < 50:
        return None
    ys, xs = np.nonzero(m)
    return int(np.mean(ys)), int(np.mean(xs))


def crop(bt, rgb, cy, cx, size=WIN, shift=(0, 0)):
    H, W = bt.shape
    y0 = int(np.clip(cy + shift[0] - size // 2, 0, max(0, H - size)))
    x0 = int(np.clip(cx + shift[1] - size // 2, 0, max(0, W - size)))
    return bt[y0:y0 + size, x0:x0 + size], rgb[y0:y0 + size, x0:x0 + size], (y0, x0)


def grade_of(bt_win):
    v = bt_win[np.isfinite(bt_win)]
    if v.size < 5000:
        return None, 0.0
    frac = float(np.mean(v <= T_COLD))
    if frac >= G_MATURE:
        return "mature", frac
    if frac >= G_DEVELOPING:
        return "developing", frac
    return "weak", frac


def render_bt(bt_win, path, tag_trm, lab, frac):
    fig = plt.figure(figsize=(4.6, 5.2), dpi=100)
    ax = fig.add_axes([0.04, 0.22, 0.80, 0.70])
    im = ax.imshow(bt_win, cmap="turbo_r", vmin=180, vmax=300,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  %s" % (tag_trm, lab), fontsize=8)
    cax = fig.add_axes([0.86, 0.22, 0.035, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("cloud-top BT (K)", fontsize=7)
    for t in (T_COLD, T_CONV, T_CLOUD):
        cb.ax.axhline(t, color="w", lw=0.7, ls="--")
    fig.text(0.04, 0.165, "Sensor: VIIRS/JPSS-1 I05 11.45 um, 375 m", fontsize=6.2)
    fig.text(0.04, 0.120, "Criteria: cloud-top BT 260/220/200 K (deep convection)",
             fontsize=6.2)
    fig.text(0.04, 0.075, "Fraction <=200 K in window: %.1f %%" % (frac * 100), fontsize=6.2)
    fig.text(0.04, 0.030, "Grade: %s   (no geolocation; located by cold cloud top)" % lab,
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def render_rgb(rgb_win, path, tag_trm):
    fig = plt.figure(figsize=(4.6, 4.8), dpi=100)
    ax = fig.add_axes([0.03, 0.14, 0.94, 0.80])
    ax.imshow(rgb_win, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  false colour I1/I2/I3" % tag_trm, fontsize=8)
    fig.text(0.03, 0.075, "Bands: I1 0.64um (R) / I2 0.865um (G) / I3 1.61um (B)",
             fontsize=6.2)
    fig.text(0.03, 0.030, "Sensor: VIIRS/JPSS-1 VJ102IMG, 375 m", fontsize=6.2)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(RAW, "*.nc")))
    if args.keep:
        fs = fs[:args.keep]
    print("待渲染:", len(fs))
    if not fs:
        print("无数据")
        return

    recs = []
    counter = collections.Counter()

    for i, f in enumerate(fs):
        m = re.search(r"A(\d{7})\.(\d{4})", os.path.basename(f))
        stamp = ("%s_%s" % (m.group(1), m.group(2))) if m else str(i)
        try:
            bt, rgb = load_scene(f)
        except Exception as e:
            print("  %s 读取失败 %s" % (stamp, str(e)[:60]))
            continue

        c = find_cold_center(bt)
        if c is None:
            print("  %s 找不到冷云区" % stamp)
            continue
        cy, cx = c

        # 三窗口：中心 + 两个偏移（制造同一景内的强弱对比）
        wins = [(cy, cx, (0, 0)),
                (cy, cx, (0, int(WIN * 0.9))),
                (cy, cx, (int(WIN * 0.9), 0))]
        for (yy, xx, sh) in wins:
            bt_w, rgb_w, (y0, x0) = crop(bt, rgb, yy, xx, WIN, shift=sh)
            if bt_w.shape[0] < WIN or bt_w.shape[1] < WIN:
                continue
            lab, frac = grade_of(bt_w)
            if lab is None:
                continue
            counter[lab] += 1
            tag = "%s_%d_%d" % (stamp, y0, x0)

            p1 = os.path.join(IMG, "%s_bt.png" % tag)
            render_bt(bt_w, p1, tag, lab, frac)
            recs.append({"image": os.path.basename(p1), "stem": tag, "label": lab,
                         "product": "intrinsic_cloud_top_bt",
                         "frac_cold": round(frac, 4),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_BT},
                             {"from": "gpt",
                              "value": ANS[lab] + " Fraction of cloud tops at or below "
                                       "200 K is %.1f percent." % (frac * 100)}]})

            p2 = os.path.join(IMG, "%s_rgb.png" % tag)
            render_rgb(rgb_w, p2, tag)
            recs.append({"image": os.path.basename(p2), "stem": tag, "label": lab,
                         "product": "intrinsic_false_colour",
                         "frac_cold": round(frac, 4),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_RGB},
                             {"from": "gpt", "value": ANS[lab]}]})

        if i % 5 == 0 or i == len(fs) - 1:
            print("  [%d/%d] %s  冷云中心 (%d,%d)  累计样本 %d"
                  % (i + 1, len(fs), stamp, cy, cx, len(recs)))

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

    json.dump(train, open(os.path.join(OUT, "train_hurricane.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_hurricane_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("分级分布:", dict(counter))
    print("产品×标签:")
    for k in sorted(by):
        print("   %-30s %-12s %d" % (k[0], k[1], len(by[k])))
    print("输出:", IMG)


if __name__ == "__main__":
    main()

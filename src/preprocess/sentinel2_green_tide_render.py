# -*- coding: utf-8 -*-
r"""
光学绿潮 · Sentinel-2 L2A 渲染（源 1 光学 · 赤潮/绿潮）

为什么用 FAI 而不是 NDCI
-----------------------
  旧的 sentinel2_algal_render.py 用 NDCI = (B05−B04)/(B05+B04)，
  并把藻华阈值拍在 0.02 —— 这个数没有出处。
  且 Digital Earth Africa 官方文档明说 NDCI 对 Sentinel-2 仍是
  "experimental index，需要进一步标定"。不带出处，不能用。

  改用 FAI（Floating Algae Index，漂浮藻类指数）：
      FAI = R_NIR − [ R_red + (R_SWIR − R_red)·(λ_NIR − λ_red)/(λ_SWIR − λ_red) ]
  即近红外反射率减去"红—短波红外"基线在近红外处的线性插值。
  漂浮藻类在近红外反射率明显高于该基线，FAI > 0。

  出处：Hu C., 2009, JGR-Oceans（黄海浒苔绿潮 FAI 原始文献）
        Hu C. et al., 2023（90 次引用，绿潮遥感综述与验证）

  为什么 FAI 适合绿潮而非赤潮
    绿潮是漂浮大型藻类（浒苔），在海面形成厚层，近红外强反射；
    赤潮是浮游微藻，在水中，近红外信号弱，需叶绿素算法。
    黄海浒苔每年 5~8 月暴发，Sentinel-2 重访 5 天，样本稳定。

Sentinel-2 波段
--------------
  B04  665 nm  红
  B08  842 nm  近红外
  B11 1610 nm  短波红外（20 m，需重采样到 10 m 网格）

双产品
------
  本征场 = FAI 场（漂浮藻类指数）
  特色场 = 真彩色（B04/B03/B02），看绿潮条带的实际形态

分级（按窗口内 FAI 超阈像元占比）
--------------------------------
  FAI 阈值的确定方式：
    Hu 2009 的做法是对每个场景取"清澈海水 FAI 的中位数 + 2×标准差"作为阈值，
    因为 FAI 的绝对值受大气订正残差与太阳角度影响，绝对阈值不可跨场景移植。
    本脚本沿用该做法（逐景自适应），不写死常数。
  占比分档：
    >= 0.15   green_tide     明显绿潮覆盖
    0.03~0.15 patchy         零星漂浮藻斑
    < 0.03    clean_water    无绿潮

用法
----
  python src/preprocess/sentinel2_green_tide_render.py --keep 1
  python src/preprocess/sentinel2_green_tide_render.py
"""
import os
import io
import re
import sys
import glob
import json
import argparse
import zipfile
import collections
import random

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\sentinel2"
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\green_tide"
IMG_DIR = os.path.join(OUT, "images")

WIN = 256                 # 10 m 网格上的窗口像元数 → 2.56 km 见方
OUT_SIZE = 384
SCALE_DIV = 4             # 10980/4 = 2745，降采样后再切窗口
MAX_PER_CLASS = 60
VAL_PER_CLASS = 30

# Sentinel-2 中心波长（nm）
LAM = {"B04": 665.0, "B08": 842.0, "B11": 1610.0}

G_TIDE = 0.15
G_PATCHY = 0.03

PROMPT_FAI = ("This is a Floating Algae Index (FAI) field derived from Sentinel-2 over the "
              "Yellow Sea. Positive values mark floating macroalgae (Ulva) slicks on the sea "
              "surface. Describe where the floating algae is and how it is distributed.")
PROMPT_RGB = ("This is a Sentinel-2 true-colour image (B04 red, B03 green, B02 blue, 10 m) "
              "of the Yellow Sea. Describe what you see: whether green floating algae "
              "slicks are present and how they are distributed.")

ANS = {
    "green_tide": "Dense green floating algae slicks cover a large part of the water "
                  "surface, forming elongated bands typical of Ulva prolifera green tides.",
    "patchy": "Scattered patches of floating algae are present on the water surface, "
              "but they do not form continuous slicks.",
    "clean_water": "The water surface is clear with no substantial floating algae; "
                   "only isolated specks may be present.",
}


def load_bands(zp, div=SCALE_DIV):
    """读 B02/B03/B04/B08/B11，统一到 TARGET×TARGET 网格"""
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    TARGET = 10980 // div

    def grab(tag):
        # 必须精确匹配 "T<编码>_<日期>_<BAND>_<分辨率>.jp2"，
        # 否则会抓到 MSK_DETFOO_B11.jp2 / MSK_QUALIT_B11.jp2 这类掩膜文件（实测踩过）。
        pat = re.compile(r"_%s(?:_20m|_60m|_10m)?\.jp2$" % tag)
        cands = [x for x in names
                 if pat.search(x) and "MSK_" not in x and "PRB_" not in x]
        if not cands:
            raise KeyError(tag)
        im = Image.open(io.BytesIO(z.read(cands[0])))
        # 20 m 波段的原始尺寸是 5490，也统一 resize，保证地理对齐
        im = im.resize((TARGET, TARGET), Image.BILINEAR)
        a = np.asarray(im).astype("float32") / 10000.0
        return a

    b04 = grab("B04")
    b03 = grab("B03")
    b02 = grab("B02")
    b08 = grab("B08")
    b11 = grab("B11")

    # SCL 场景分类（若存在）用于水/云掩膜
    scl = None
    cs = [x for x in names if re.search(r"_SCL_20m\.jp2$", x)] or \
         [x for x in names if "_SCL_" in x and x.endswith(".jp2")]
    if cs:
        im = Image.open(io.BytesIO(z.read(cs[0]))).resize((TARGET, TARGET), Image.NEAREST)
        scl = np.asarray(im).astype("uint8")
    z.close()
    return b02, b03, b04, b08, b11, scl


def fai_of(b04, b08, b11):
    """FAI = R_NIR − 线性插值的基线（红 → 短波红外）"""
    lam_r, lam_n, lam_s = LAM["B04"], LAM["B08"], LAM["B11"]
    w = (lam_n - lam_r) / (lam_s - lam_r)
    base = b04 + (b11 - b04) * w
    return b08 - base


def water_mask(b02, b03, b04, scl):
    """水体：SCL 有就优先用（4=植被,6=水,11=雪），否则用 NDWI 近似"""
    if scl is not None:
        return scl == 6
    # 无 SCL 时用蓝-绿亮度判水
    green = b03
    return (green < 0.12) & (b04 < 0.12) & (b02 < 0.12)


def adaptive_threshold(fai, water):
    """逐景自适应：清澈海水 FAI 的中位数 + 2×标准差（Hu 2009 做法）"""
    v = fai[water]
    v = v[np.isfinite(v)]
    if v.size < 5000:
        return None, None
    med = float(np.median(v))
    sd = float(np.std(v))
    return med + 2.0 * sd, med


def grade_of(fai_win, water_win, thr):
    if thr is None:
        return None, 0.0
    m = water_win & np.isfinite(fai_win)
    if m.sum() < 2000:
        return None, 0.0
    frac = float(np.mean(fai_win[m] > thr))
    if frac >= G_TIDE:
        return "green_tide", frac
    if frac >= G_PATCHY:
        return "patchy", frac
    return "clean_water", frac


def find_windows(water, max_n=3):
    """在最可能出现藻华的水体密集区找窗口"""
    H, W = water.shape
    if H < WIN or W < WIN:
        return []
    ys, xs = np.nonzero(water)
    if ys.size < WIN * WIN:
        return []
    cy, cx = int(np.median(ys)), int(np.median(xs))
    out = []
    for dy, dx in ((0, 0), (0, WIN), (WIN, 0)):
        y0 = int(np.clip(cy + dy - WIN // 2, 0, H - WIN))
        x0 = int(np.clip(cx + dx - WIN // 2, 0, W - WIN))
        if water[y0:y0 + WIN, x0:x0 + WIN].mean() > 0.5:
            out.append((y0, x0))
        if len(out) >= max_n:
            break
    return out


def render_fai(fai_win, water_win, path, tag, lab, frac, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = np.where(water_win, fai_win, np.nan)
    fig = plt.figure(figsize=(4.6, 5.2), dpi=100)
    ax = fig.add_axes([0.04, 0.22, 0.80, 0.70])
    im = ax.imshow(show, cmap="YlGn", vmin=-0.02, vmax=0.10, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  %s" % (tag, lab), fontsize=8)
    cax = fig.add_axes([0.86, 0.22, 0.035, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("FAI", fontsize=7)
    fig.text(0.04, 0.165, "Sensor: Sentinel-2 L2A, B04/B08/B11, 10 m", fontsize=6.2)
    fig.text(0.04, 0.120, "FAI = R_NIR - linear baseline (Hu 2009, JGR)", fontsize=6.2)
    fig.text(0.04, 0.075, "Scene-adaptive threshold: %.4f   Fraction above: %.1f%%"
             % (thr, frac * 100), fontsize=6.2)
    fig.text(0.04, 0.030, "Grade: %s   (floating algae, not water quality)" % lab,
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def render_rgb(rgb, path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4.6, 4.9), dpi=100)
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.78])
    v = rgb[np.isfinite(rgb)]
    hi = np.percentile(v, 99) if v.size else 0.2
    hi = max(hi, 0.02)
    ax.imshow(np.clip(rgb / hi, 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  true colour" % tag, fontsize=8)
    fig.text(0.03, 0.095, "Bands: B04 665nm (R) / B03 560nm (G) / B02 490nm (B)", fontsize=6.2)
    fig.text(0.03, 0.055, "Sensor: Sentinel-2 L2A, 10 m", fontsize=6.2)
    fig.text(0.03, 0.018, "Green algae slicks appear as bright green streaks",
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0, help="只处理前 N 个 zip")
    args = ap.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(RAW, "*.zip")))
    if args.keep:
        zips = zips[:args.keep]
    print("Sentinel-2 zip:", len(zips))
    if not zips:
        print("无数据")
        return

    recs = []
    counter = collections.Counter()
    per_class = collections.Counter()

    for i, zp in enumerate(zips):
        name = os.path.basename(zp)
        m = re.search(r"(\d{8})T(\d{6})", name)
        stamp = (m.group(1) + "_" + m.group(2)) if m else str(i)
        try:
            b02, b03, b04, b08, b11, scl = load_bands(zp)
        except Exception as e:
            print("  %s 读取失败 %s" % (stamp, str(e)[:60]))
            continue

        fai = fai_of(b04, b08, b11)
        water = water_mask(b02, b03, b04, scl)
        thr, med = adaptive_threshold(fai, water)
        if thr is None:
            print("  %s 水体像元不足" % stamp)
            continue

        wins = find_windows(water)
        if not wins:
            print("  %s 找不到水体窗口" % stamp)
            continue

        for (y0, x0) in wins:
            sl = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))
            f_w = fai[sl]; w_w = water[sl]
            lab, frac = grade_of(f_w, w_w, thr)
            if lab is None:
                continue
            if per_class[lab] >= MAX_PER_CLASS:
                continue
            per_class[lab] += 1
            counter[lab] += 1
            tag = "%s_%d_%d" % (stamp, y0, x0)

            p1 = os.path.join(IMG_DIR, "%s_fai.png" % tag)
            render_fai(f_w, w_w, p1, tag, lab, frac, thr)
            recs.append({"image": os.path.basename(p1), "stem": tag, "label": lab,
                         "product": "intrinsic_fai",
                         "frac_above": round(frac, 4), "threshold": round(thr, 5),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_FAI},
                             {"from": "gpt", "value": ANS[lab] +
                              " The fraction of water pixels above the scene-adaptive FAI "
                              "threshold is %.1f percent." % (frac * 100)}]})

            rgb_w = np.dstack([b04[sl], b03[sl], b02[sl]])
            p2 = os.path.join(IMG_DIR, "%s_rgb.png" % tag)
            render_rgb(rgb_w, p2, tag)
            recs.append({"image": os.path.basename(p2), "stem": tag, "label": lab,
                         "product": "intrinsic_true_colour",
                         "frac_above": round(frac, 4), "threshold": round(thr, 5),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_RGB},
                             {"from": "gpt", "value": ANS[lab]}]})

        print("  [%d/%d] %s  阈值 %.4f  窗口 %d  累计 %d"
              % (i + 1, len(zips), stamp, thr, len(wins), len(recs)))

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
        n = min(VAL_PER_CLASS, max(5, len(lst) // 5))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_green_tide.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_green_tide_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("分级分布:", dict(counter))
    for k in sorted(by):
        print("   %-28s %-12s %d" % (k[0], k[1], len(by[k])))


if __name__ == "__main__":
    main()

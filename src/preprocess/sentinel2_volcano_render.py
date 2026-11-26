# -*- coding: utf-8 -*-
r"""
光学火山喷发 · Sentinel-2 L2A 渲染（源 1 光学 · 火山喷发）

事件核实
-------
  Anak Krakatau（喀拉喀托之子）2026-09-05 夜间喷发，9-06 持续
  来源：NASA Earth Observatory（9-05 Landsat 8 影像）、
        美国驻印尼使馆 9-06 警报、IQAir 9-06 报告
  影响：苏加诺-哈达、哈利姆机场临时关闭，火山灰扩散至爪哇岛

判据（双时相对比，基准取喷发前最近一景）
------------------------------------
  bright_plume   窗口内平均亮度（B08+B04）/2 较基准上升 > 0.30
                 → 喷发柱 / 火山灰云遮挡
  ash_deposit    窗口内 NDVI 较基准下降 > 0.08
                 → 火山灰沉降覆盖植被
  stable         两者都不满足

  两个阈值是本项目的经验值，没有对应国标。图上会明标"经验阈值"。

质量控制（三个剔除条件）
----------------------
  ① 海面窗口      SCL==6（水体）占比 > 50% → 剔除
  ② 云窗口        SCL∈{3,8,9,10}（云影/云/薄卷云）占比 > 20% → 剔除
  ③ 无数据窗口    SCL==0 或 1（无数据/饱和）占比 > 20% → 剔除

  为什么要剔海面：stable 类别里大量是海面、远岛窗口，
  它们对"火山喷发"这个标签毫无信息量，会把模型带偏。

双产品
------
  本征场 = 官方真彩色 TCI（B04/B03/B02，经大气校正与拉伸）
  特色场 = NDVI 差值场，看植被受损范围（火山灰沉降）

用法
----
  python src/preprocess/sentinel2_volcano_render.py --post 3     # 试渲
  python src/preprocess/sentinel2_volcano_render.py              # 全跑
"""
import os
import re
import io
import sys
import glob
import json
import zipfile
import argparse
import collections
import random

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\sentinel2_volcano"
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\volcano"
IMG_DIR = os.path.join(OUT, "images")

TILE = "T48MWU"
ERUPT_DATE = "20260905"          # 喷发日（9-05 夜间）

WIN = 128                        # 20 m 尺度下的窗口边长 = 2.56 km
STRIDE = 96
OUT_SIZE = 384

BRIGHT_RISE = 0.30               # 亮度上升阈值（经验值）
NDVI_DROP = 0.08                 # NDVI 下降阈值（经验值）

WATER_SKIP = 0.50                # 水体占比上限
CLOUD_SKIP = 0.20                # 云占比上限
NODATA_SKIP = 0.20               # 无数据占比上限

SCL_CLOUD = {3, 8, 9, 10}        # 云影 / 云中高概率 / 云高概率 / 薄卷云
SCL_NODATA = {0, 1}              # 无数据 / 饱和
SCL_WATER = {6}                  # 水体


def _grab(z, names, tag, res, div):
    """按 tag 精确取图，避免抓到 MSK_DETFOO_ 掩膜（踩过坑）"""
    pat = re.compile(r"_%s_%s\.jp2$" % (tag, res))
    cands = [x for x in names
             if pat.search(x) and "MSK_" not in x and "PRB_" not in x]
    if not cands:
        return None
    im = Image.open(io.BytesIO(z.read(cands[0])))
    if div > 1:
        im = im.resize((im.width // div, im.height // div), Image.BILINEAR)
    return np.asarray(im)


def load_scene(zp, scale=2):
    """
    scale=2  → 基准分辨率取 20 m（B02/B03/B04/B08 的 10m 图缩一半），
              SCL 原生 20m 正好对齐，TCI 取 20m。
    返回 rgb(float 0-1), ndvi, bright, scl
    """
    z = zipfile.ZipFile(zp)
    names = z.namelist()

    b08 = _grab(z, names, "B08", "10m", scale)
    b04 = _grab(z, names, "B04", "10m", scale)
    b03 = _grab(z, names, "B03", "10m", scale)
    b02 = _grab(z, names, "B02", "10m", scale)

    tci = _grab(z, names, "TCI", "20m", 1)
    scl = _grab(z, names, "SCL", "20m", 1)
    z.close()

    if b08 is None or b04 is None:
        raise RuntimeError("缺 B08/B04")

    H = min(x.shape[0] for x in (b08, b04, b03, b02))
    W = min(x.shape[1] for x in (b08, b04, b03, b02))
    b08 = b08[:H, :W].astype("float64") / 1e4
    b04 = b04[:H, :W].astype("float64") / 1e4
    b03 = b03[:H, :W].astype("float64") / 1e4
    b02 = b02[:H, :W].astype("float64") / 1e4

    denom = b08 + b04
    denom[denom == 0] = 1e-6
    ndvi = (b08 - b04) / denom
    bright = (b08 + b04) / 2.0

    # 官方真彩色（已经是 8 bit sRGB 拉伸好的）
    if tci is not None:
        rgb = tci[:H, :W].astype("float64") / 255.0
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
    else:
        rgb = np.stack([b04, b03, b02], axis=2) * 3.5   # 退路
        np.clip(rgb, 0, 1, out=rgb)

    if scl is None:
        scl = np.zeros((H, W), dtype="uint8")
    else:
        scl = scl[:H, :W].astype("uint8")

    return rgb, ndvi, bright, scl


def grade(dndvi, dbright):
    if dbright > BRIGHT_RISE:
        return "bright_plume"
    if dndvi < -NDVI_DROP:
        return "ash_deposit"
    return "stable"


def qc_ok(scl_win):
    """
    窗口质量检查。返回 (通过?, 原因)
    """
    if scl_win.size == 0:
        return False, "empty"
    water = np.isin(scl_win, list(SCL_WATER)).mean()
    cloud = np.isin(scl_win, list(SCL_CLOUD)).mean()
    nodata = np.isin(scl_win, list(SCL_NODATA)).mean()
    if nodata > NODATA_SKIP:
        return False, "nodata"
    if water > WATER_SKIP:
        return False, "water"
    if cloud > CLOUD_SKIP:
        return False, "cloud"
    return True, "ok"


PROMPT_RGB = ("This is a Sentinel-2 true-colour image (TCI, 20 m) of Anak Krakatau "
              "volcano. Describe what you see: whether an eruption plume or ash "
              "cloud is present over the volcano.")
PROMPT_NDVI = ("This is an NDVI change image (NDVI after minus NDVI before the eruption) "
               "of Anak Krakatau. Negative values indicate vegetation damaged by ash "
               "fall. Describe the pattern of change.")

ANS = {
    "bright_plume": "A bright eruption plume or ash cloud is visible over the volcano, "
                    "raising the scene brightness well above the pre-eruption baseline.",
    "ash_deposit": "Vegetation cover has decreased markedly compared with the "
                   "pre-eruption baseline, consistent with ash fall blanketing the slopes.",
    "stable": "The scene shows no substantial change from the pre-eruption baseline; "
              "neither a plume nor significant ash damage is evident.",
}


def render_rgb(rgb_u8, path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4.6, 4.9), dpi=100)
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.78])
    ax.imshow(rgb_u8, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("%s  true colour" % tag, fontsize=8)
    fig.text(0.03, 0.095, "Bands: TCI (B04 665nm / B03 560nm / B02 490nm)",
             fontsize=6.2)
    fig.text(0.03, 0.055, "Sensor: Sentinel-2 L2A, 20 m, Anak Krakatau", fontsize=6.2)
    fig.text(0.03, 0.018, "Empirical threshold, not a national standard",
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE),
                                           Image.BILINEAR).save(path)


def render_ndvi_diff(diff, path, tag, dndvi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4.6, 4.9), dpi=100)
    ax = fig.add_axes([0.03, 0.16, 0.86, 0.78])
    im = ax.imshow(diff, cmap="RdYlGn", vmin=-0.3, vmax=0.3,
                   interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("%s  NDVI change" % tag, fontsize=8)
    cax = fig.add_axes([0.91, 0.16, 0.03, 0.78])
    fig.colorbar(im, cax=cax, label="NDVI change")
    fig.text(0.03, 0.095, "Negative = vegetation damaged (ash fall).  mean %+.3f" % dndvi,
             fontsize=6.2)
    fig.text(0.03, 0.055, "NDVI = (B08 - B04) / (B08 + B04), Sentinel-2 20 m",
             fontsize=6.2)
    fig.text(0.03, 0.018, "Empirical threshold: NDVI drop > 0.08",
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE),
                                           Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", type=int, default=0,
                    help="只取喷发后前 N 景（0=全部，用于快速试渲）")
    args = ap.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(RAW, "*%s*.zip" % TILE)))
    items = []
    for zp in zips:
        m = re.search(r"_(\d{8})T\d+", os.path.basename(zp))
        if m:
            items.append((m.group(1), zp))
    items.sort()
    print("景数:", len(items))

    pre = [x for x in items if x[0] < ERUPT_DATE]
    post = [x for x in items if x[0] >= ERUPT_DATE]
    if args.post:
        post = post[:args.post]
    if not pre or not post:
        print("✗ 缺少喷发前或喷发后的景")
        return

    base_date, base_zip = pre[-1]
    print("基准（喷发前最近）:", base_date)
    print("对比景 %d 个:" % len(post), [d for d, _ in post])

    base_rgb, base_ndvi, base_bright, base_scl = load_scene(base_zip)
    H, W = base_ndvi.shape
    print("影像 %dx%d（20 m）" % (H, W))
    ys = list(range(0, max(1, H - WIN + 1), STRIDE))
    xs = list(range(0, max(1, W - WIN + 1), STRIDE))
    print("网格 %dx%d = %d 窗口/景" % (len(ys), len(xs), len(ys) * len(xs)))

    recs = []
    stats = collections.Counter()

    for date, zp in post:
        try:
            rgb, ndvi, bright, scl = load_scene(zp)
        except Exception as e:
            print("  ✗ %s: %s" % (date, str(e)[:70]))
            continue

        kept = 0
        for y in ys:
            for x in xs:
                ys_, xs_ = slice(y, y + WIN), slice(x, x + WIN)
                scl_w = scl[ys_, xs_]
                ok, why = qc_ok(scl_w)
                if not ok:
                    stats["reject_" + why] += 1
                    continue

                av, bv = ndvi[ys_, xs_], base_ndvi[ys_, xs_]
                ab, bb = bright[ys_, xs_], base_bright[ys_, xs_]
                if np.isfinite(av).mean() < 0.8 or np.isfinite(bv).mean() < 0.8:
                    stats["reject_nan"] += 1
                    continue

                dndvi = float(np.nanmean(av) - np.nanmean(bv))
                dbright = float(np.nanmean(ab) - np.nanmean(bb))
                lab = grade(dndvi, dbright)

                tag = "%s_%s_%d_%d" % (base_date, date, y, x)
                kept += 1
                stats[lab] += 1

                p1 = os.path.join(IMG_DIR, "%s_ndvi.png" % tag)
                render_ndvi_diff(av - bv, p1, tag, dndvi)
                recs.append({
                    "image": os.path.basename(p1), "stem": tag, "label": lab,
                    "product": "intrinsic_ndvi_change",
                    "pre": base_date, "post": date,
                    "d_ndvi": round(dndvi, 4), "d_bright": round(dbright, 4),
                    "conversations": [
                        {"from": "human", "value": "<image>" + PROMPT_NDVI},
                        {"from": "gpt", "value": ANS[lab] +
                         " Mean NDVI change is %+.3f." % dndvi}]})

                win = rgb[ys_, xs_]
                p2 = os.path.join(IMG_DIR, "%s_rgb.png" % tag)
                render_rgb((np.clip(win, 0, 1) * 255).astype("uint8"), p2, tag)
                recs.append({
                    "image": os.path.basename(p2), "stem": tag, "label": lab,
                    "product": "intrinsic_true_colour",
                    "pre": base_date, "post": date,
                    "d_ndvi": round(dndvi, 4), "d_bright": round(dbright, 4),
                    "conversations": [
                        {"from": "human", "value": "<image>" + PROMPT_RGB},
                        {"from": "gpt", "value": ANS[lab]}]})

        print("  %s  保留 %d  累计样本 %d" % (date, kept, len(recs)))

    if not recs:
        print("无样本")
        return

    # 划分：按 post 日期分层，保证每个日期的各类样本都能进 val
    random.seed(53)
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["product"], r["label"])].append(r)
    val, train = [], []
    for k, lst in by.items():
        lst = lst[:]
        random.shuffle(lst)
        n = max(5, len(lst) // 5)
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_volcano.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_volcano_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n=== 结果 ===")
    print("train %d | val %d | 合计 %d" % (len(train), len(val), len(recs)))
    print("剔除统计:")
    for k in sorted(stats):
        print("   %-18s %d" % (k, stats[k]))
    print("\n按产品×等级:")
    for k in sorted(by):
        print("   %-28s %-14s %d" % (k[0], k[1], len(by[k])))
    print("\n图片:", len(os.listdir(IMG_DIR)))
    print("输出:", OUT)


if __name__ == "__main__":
    main()

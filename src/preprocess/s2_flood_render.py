# -*- coding: utf-8 -*-
r"""
Sentinel-2 L2A 洪水/涝灾样本渲染（源 1 光学）

关键修复（相对旧版）：
  1. **SCL 云掩膜**：SCL 值 3=云影 8=中概率云 9=高概率云 10=薄卷云 11=雪
     → 含云量过高的窗直接丢弃（旧版没做，云被当成地物）
  2. **双产品**：
       本征场 = 真彩色 TCI（该源的标准视感）
       特色场 = NDWI 水体指数（洪水特征）
  3. **判据**：窗口内水体占比
       >= 0.55  → flood（洪水）
       >= 0.15  → waterlogging（涝/积水）
       否则      → normal（无水）

依赖：PIL（自带 openjpeg 支持 jp2）

用法:
  python s2_flood_render.py [--grid 0.05] [--min-valid 0.5]
"""
import os
import sys
import glob
import json
import argparse
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\sentinel2"
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\flood"
IMG = os.path.join(OUT, "images")
OUT_SIZE = 384
VAL_PER_CLASS = 30
MAX_PER_FILE = 12

# SCL 分类码
SCL_CLOUD = {3, 8, 9, 10, 11}
CLOUD_MAX = 0.05        # 窗内云比例超过 5% 丢弃
NDWI_WATER = 0.15
FLOOD_FRAC = 0.55
LOGGING_FRAC = 0.15

PROMPT = ("This is a Sentinel-2 optical scene of a flood-prone area. "
          "Classify the water condition: normal, waterlogging, or flood.")


def find_band(granule, tag):
    p = glob.glob(os.path.join(granule, "IMG_DATA", "R20m", "*_%s_20m.jp2" % tag))
    if not p:
        p = glob.glob(os.path.join(granule, "IMG_DATA", "R10m", "*_%s_10m.jp2" % tag))
    if not p:
        p = glob.glob(os.path.join(granule, "IMG_DATA", "R60m", "*_%s_60m.jp2" % tag))
    return p[0] if p else None


def read_band(path, div):
    im = Image.open(path)
    a = np.array(im).astype("float32")
    if div > 1:
        a = a[::div, ::div]
    a[a == 0] = np.nan
    return a / 10000.0


def load_scene(safe_dir, div=4):
    """返回 (rgb, ndwi, cloud_mask)；都按 div 降采样"""
    gr = glob.glob(os.path.join(safe_dir, "GRANULE", "*"))
    if not gr:
        return None
    gr = gr[0]
    paths = {t: find_band(gr, t) for t in ("B02", "B03", "B04", "B08", "SCL")}
    if any(v is None for v in paths.values()):
        return None

    b02 = read_band(paths["B02"], div)
    b03 = read_band(paths["B03"], div)
    b04 = read_band(paths["B04"], div)
    b08 = read_band(paths["B08"], div)

    scl_im = Image.open(paths["SCL"])
    scl = np.array(scl_im)
    scl = scl[::div, ::div]

    # 对齐尺寸（jp2 各分辨率可能差 1 像素）
    h = min(b03.shape[0], scl.shape[0])
    w = min(b03.shape[1], scl.shape[1])
    b02, b03, b04, b08 = (a[:h, :w] for a in (b02, b03, b04, b08))
    scl = scl[:h, :w]

    cloud = np.isin(scl, list(SCL_CLOUD))
    valid = (scl == 4) | (scl == 5) | (scl == 6) | (scl == 7) | (scl == 2)

    denom = b03 + b08
    denom[denom == 0] = np.nan
    ndwi = (b03 - b08) / denom

    def bright(a):
        v = np.nanpercentile(a[~np.isnan(a)], 98) if np.any(~np.isnan(a)) else 1.0
        return np.clip(a / (v if v > 0 else 1.0), 0, 1)

    rgb = np.dstack([bright(b04), bright(b03), bright(b02)])
    rgb[~valid] = 1.0          # 无效区域刷白（不喂噪声）

    return rgb, ndwi, cloud, valid


def grade(frac, valid_frac):
    if valid_frac < 0.5:
        return None
    if frac >= FLOOD_FRAC:
        return "flood"
    if frac >= LOGGING_FRAC:
        return "waterlogging"
    return "normal"


def draw(rgb, ndwi, size, path, title, mode):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=96)
    if mode == "rgb":
        ax.imshow(rgb)
    else:
        im = ax.imshow(np.nan_to_num(ndwi, nan=-1), cmap="Blues", vmin=-0.4, vmax=0.6)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("NDWI", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--div", type=int, default=4)
    ap.add_argument("--keep", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    scenes = sorted(glob.glob(os.path.join(RAW, "*.SAFE")))
    if args.keep:
        scenes = scenes[:args.keep]
    print("场景 %d | 窗 %.3f° | 降采样 1/%d" % (len(scenes), args.grid, args.div))
    if not scenes:
        print("没有解压后的 SAFE 目录"); return

    recs, cnt = [], {}
    for si, sd in enumerate(scenes, 1):
        stamp = os.path.basename(sd).split("_")[2][:8]
        try:
            out = load_scene(sd, args.div)
        except Exception as e:
            print("  %s 读取失败 %s" % (stamp, str(e)[:60])); continue
        if out is None:
            print("  %s 缺波段，跳过" % stamp); continue
        rgb, ndwi, cloud, valid = out
        H, W = ndwi.shape
        print("  [%d/%d] %s  %dx%d" % (si, len(scenes), stamp, W, H), flush=True)

        step = max(64, int(args.grid / 0.0002 / args.div))
        by = {}
        for y in range(0, H - step, step):
            for x in range(0, W - step, step):
                nb = ndwi[y:y + step, x:x + step]
                nc = cloud[y:y + step, x:x + step]
                nv = valid[y:y + step, x:x + step]
                vf = float(nv.mean())
                cf = float(nc.mean())
                if vf < 0.5 or cf > CLOUD_MAX:
                    continue
                water = float(np.nanmean(nb > NDWI_WATER))
                lab = grade(water, vf)
                if lab is None:
                    continue
                by.setdefault(lab, []).append((abs(water - 0.35), x, y, water))

        picked = []
        for lab, lst in by.items():
            lst.sort()
            picked += [(lab, t) for t in lst[:MAX_PER_FILE]]

        for lab, (_, x, y, water) in picked:
            cnt[lab] = cnt.get(lab, 0) + 1
            for sfx, mode, prod in (("_rgb", "rgb", "intrinsic_truecolor"),
                                    ("_ndwi", "ndwi", "feature_ndwi")):
                png = os.path.join(IMG, "%s_%d_%d%s.png" % (stamp, x, y, sfx))
                draw(rgb[y:y + step, x:x + step],
                     ndwi[y:y + step, x:x + step], OUT_SIZE,
                     png, "%s %s" % (stamp, lab), mode)
                ans = {"normal": "No significant surface water (NDWI below 0.15).",
                       "waterlogging": "Partial standing water / waterlogging detected (NDWI water fraction 15-55%).",
                       "flood": "Extensive flooding detected (water fraction above 55%)."}[lab]
                recs.append({"conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": ans}],
                    "images": [png], "product": prod, "label": lab,
                    "water_frac": round(water, 3), "stamp": stamp})

    import random, collections
    random.seed(7)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 6))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_flood.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_flood_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("窗分布:", dict(cnt))
    print("val:", dict(collections.Counter(r["label"] for r in val)))


if __name__ == "__main__":
    main()

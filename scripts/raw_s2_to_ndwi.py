# -*- coding: utf-8 -*-
r"""
原始 Sentinel-2 L2A（SAFE/zip）→ NDWI 指数图（与训练同规格）。

用途：验证「原始卫星影像 → 渲染 → 判读」工作流，不依赖已建好的数据集。

渲染规格与训练数据一致：
  - 384x384 PNG
  - NDWI = (B03 - B08) / (B03 + B08)，读数范围 -0.4 ~ 0.6
  - 色标 Blues，带色标图例与日期标题
  - SCL 云掩膜（3/8/9/10/11），无效区刷白

用法：
  python scripts/raw_s2_to_ndwi.py --safe <xxx.SAFE 或 zip> --windows 6 --outdir outputs/raw_test
"""
import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SCL_CLOUD = {3, 8, 9, 10, 11}
OUT_SIZE = 384
NDWI_WATER = 0.15
FLOOD_FRAC = 0.55
LOGGING_FRAC = 0.15


def find_band(granule, tag):
    for res in ("R20m", "R10m", "R60m"):
        p = glob.glob(os.path.join(granule, "IMG_DATA", res, "*_%s_%s.jp2" % (tag, res[1:])))
        if p:
            return p[0]
    return None


def read_band(path, div):
    a = np.array(Image.open(path)).astype("float32")
    if div > 1:
        a = a[::div, ::div]
    a[a == 0] = np.nan
    return a / 10000.0


def grade(frac):
    if frac >= FLOOD_FRAC:
        return "flood"
    if frac >= LOGGING_FRAC:
        return "waterlogging"
    return "normal"


def draw(ndwi, step, path, title):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=96)
    im = ax.imshow(np.nan_to_num(ndwi, nan=-1), cmap="Blues", vmin=-0.4, vmax=0.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("NDWI", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=96)
    plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", required=True, help="已解压 SAFE 目录，或 .zip")
    ap.add_argument("--outdir", default=r"C:\vlmdisaster\outputs\raw_test")
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--div", type=int, default=4)
    ap.add_argument("--grid", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    safe = args.safe
    if safe.lower().endswith(".zip"):
        with zipfile.ZipFile(safe) as z:
            safe_dirs = {n.split("/")[0] for n in z.namelist() if n.endswith("MTD_MSIL2A.xml")}
            z.extractall(os.path.dirname(safe))
        safe = os.path.join(os.path.dirname(safe), sorted(safe_dirs)[0])
    gr = glob.glob(os.path.join(safe, "GRANULE", "*"))
    if not gr:
        raise SystemExit("找不到 GRANULE：%s" % safe)
    gr = gr[0]
    paths = {t: find_band(gr, t) for t in ("B02", "B03", "B04", "B08", "SCL")}
    missing = [k for k, v in paths.items() if not v]
    if missing:
        raise SystemExit("缺波段：%s" % missing)

    b02 = read_band(paths["B02"], args.div)
    b03 = read_band(paths["B03"], args.div)
    b04 = read_band(paths["B04"], args.div)
    b08 = read_band(paths["B08"], args.div)
    scl = np.array(Image.open(paths["SCL"]))[::args.div, ::args.div]

    h = min(b03.shape[0], scl.shape[0])
    w = min(b03.shape[1], scl.shape[1])
    b02, b03, b04, b08, scl = (a[:h, :w] for a in (b02, b03, b04, b08, scl))

    cloud = np.isin(scl, list(SCL_CLOUD))
    valid = (scl == 4) | (scl == 5) | (scl == 6) | (scl == 7) | (scl == 2)
    denom = b03 + b08
    denom[denom == 0] = np.nan
    ndwi = (b03 - b08) / denom

    stamp = os.path.basename(safe).split("_")[2][:8] if "_" in os.path.basename(safe) else "raw"
    step = max(64, int(args.grid / 0.0002 / args.div))
    print("场景 %s | 网格 %dx%d | 窗 %d | 降采样 1/%d" % (stamp, w, h, step, args.div))

    rows = []
    for y in range(0, max(1, h - step), step):
        for x in range(0, max(1, w - step), step):
            vf = float(valid[y:y + step, x:x + step].mean())
            cf = float(cloud[y:y + step, x:x + step].mean())
            if vf < 0.5 or cf > 0.05:
                continue
            nb = ndwi[y:y + step, x:x + step]
            water = float(np.nanmean(nb > NDWI_WATER))
            rows.append((x, y, round(water, 3), grade(water)))
    rows.sort(key=lambda r: -r[2])
    picked = rows[:args.windows] + [r for r in rows if r[3] == "normal"][:1]
    print("候选窗口 %d，取 %d 个" % (len(rows), len(picked)))

    out = []
    for x, y, water, lab in picked:
        png = os.path.join(args.outdir, "%s_%d_%d_ndwi.png" % (stamp, x, y))
        draw(ndwi[y:y + step, x:x + step], step, png, "%s %s" % (stamp, lab))
        out.append({"image": png, "x": x, "y": y, "water_frac": water, "label": lab,
                    "cloud_frac": round(float(cloud[y:y + step, x:x + step].mean()), 3)})
        print("  %-12s water=%.3f  %s" % (lab, water, os.path.basename(png)))

    json.dump(out, open(os.path.join(args.outdir, "windows.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("写出", args.outdir)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
火烧迹地 × 光学（Sentinel-2 L1C NBR）二分类（新元素：burn_scar_s2）

NBR = (B08 − B12) / (B08 + B12)；烧毁区 NBR 显著低（近红外降、短波红外升）。
标签：none / burned；窗口平均 NBR < 0.10 记 burned（样本取自 2025 年北方火场灾后影像）。
"""
import glob
import json
import os
import random
import re
import sys
import zipfile

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "01_optical_vnir", "sentinel2_burn")
OUT = os.path.join(ROOT, "datasets", "01_optical_vnir", "burn_scar_s2")
IMG = os.path.join(OUT, "images")

WIN = 256
STRIDE = 256
POS_PER_SCENE = 6
NEG_PER_SCENE = 6
SEED = 42
OUT_SIZE = 384
SCALE_DIV = 4      # 10980/4 = 2745

PROMPT = ("This is a Sentinel-2 Normalized Burn Ratio (NBR) map over a fire-affected area: "
          "low NBR means burned surface. Output a JSON object with fields: disaster_type, "
          "detected (true/false), source, physical_quantity, severity, criterion. severity "
          "must be one of these exact words: none / burned.")


def load_band(z, names, band, target):
    cand = [n for n in names if re.search(r"_%s_20m\.jp2$" % band, n)] or \
           [n for n in names if re.search(r"_%s_10m\.jp2$" % band, n)] or \
           [n for n in names if re.search(r"_%s\.jp2$" % band, n)]
    if not cand:
        return None
    with z.open(cand[0]) as f:
        im = Image.open(f).convert("I")
        if im.size[0] != target:
            im = im.resize((target, target), Image.BILINEAR)
        return np.asarray(im).astype("float32") / 10000.0


def nbr_of(bands):
    b8 = bands.get("B08")
    b12 = bands.get("B12")
    if b8 is None or b12 is None:
        return None
    return (b8 - b12) / (b8 + b12 + 1e-6)


def render(win, path, tag, frac):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="RdYlGn", vmin=-0.3, vmax=0.9, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  burned %.0f%%" % (tag, frac * 100), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("NBR", fontsize=7)
    fig.text(0.04, 0.075, "Sentinel-2 L1C, NBR = (B08-B12)/(B08+B12)", fontsize=6)
    fig.text(0.04, 0.035, "red = burned, green = healthy vegetation", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    zips = sorted(glob.glob(os.path.join(RAW, "*.zip")))
    print("场景 %d" % len(zips))
    records = []
    random.seed(SEED)
    for zp in zips:
        tag = os.path.basename(zp).split("_")[2][:8] if "_" in os.path.basename(zp) else "scene"
        try:
            z = zipfile.ZipFile(zp)
        except Exception:
            continue
        names = z.namelist()
        target = 10980 // SCALE_DIV
        bands = {}
        for b in ("B08", "B12"):
            v = load_band(z, names, b, target)
            if v is not None:
                bands[b] = v
        z.close()
        nd = nbr_of(bands)
        if nd is None:
            print("  skip", tag)
            continue
        h, w = nd.shape
        pos, neg = [], []
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                win = nd[r0:r0 + WIN, c0:c0 + WIN]
                if np.isfinite(win).mean() < 0.5:
                    continue
                m = float(np.nanmean(win))
                item = (r0, c0, m)
                if m < 0.10:
                    pos.append(item)
                elif m > 0.30:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        for det, (r0, c0, m) in [(True, x) for x in pos[:POS_PER_SCENE]] + \
                [(False, x) for x in neg[:NEG_PER_SCENE]]:
            sev = "burned" if det else "none"
            nm = "burn_%s_r%d_c%d.png" % (tag, r0, c0)
            pp = os.path.join(IMG, nm)
            render(nd[r0:r0 + WIN, c0:c0 + WIN], pp, tag, float((nd[r0:r0 + WIN, c0:c0 + WIN] < 0.1).mean()))
            ans = {
                "disaster_type": "burn_scar",
                "detected": bool(det),
                "source": "OPT",
                "physical_quantity": "nbr",
                "severity": sev,
                "criterion": "NBR < 0.10 indicates burned surface (Key & Benson 2006)",
            }
            records.append({
                "image": pp, "label": sev, "stamp": tag,
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        print("  %s pos %d neg %d 累计 %d" % (tag, len(pos), len(neg), len(records)))

    seen, uniq = set(), []
    for r in records:
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        uniq.append(r)
    random.seed(SEED)
    by = {}
    for r in uniq:
        by.setdefault(r["label"], []).append(r)
    val, train = [], []
    for k, lst in sorted(by.items()):
        lst = lst[:]
        random.shuffle(lst)
        n = min(max(2, int(round(len(lst) * 0.25))), 40, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])
    json.dump(train, open(os.path.join(OUT, "train_burn_scar_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_burn_scar_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()

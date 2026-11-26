# -*- coding: utf-8 -*-
r"""
闪电（GOES GLM 全球雷暴小时数）—— 源 10 第一个元素

数据：glmhth（0.05°，24 小时，thunder_hours）
双产品：
  本征场 = 日雷暴小时数场（0~24 h）
  特色场 = 雷暴分级（WMO 口径近似：日雷暴小时数）

判据：
  >= 6 h  → severe_thunderstorm（强雷暴日）
  >= 1 h  → thunderstorm（有雷暴）
  否则     → no_storm

输入：data\raw\10_lightning\glm\*.nc
输出：datasets\10_lightning\thunderstorm\images\*.png + json

用法: python lightning_render.py [--grid 20] [--keep N]
"""
import os, re, sys, glob, json, argparse
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from PIL import Image

RAW = r"C:\vlmdisaster\data\raw\10_lightning\glm"
OUT = r"C:\vlmdisaster\datasets\10_lightning\thunderstorm"
IMG = os.path.join(OUT, "images")
OUT_SIZE = 384
VAL_PER_CLASS = 30
MAX_PER_FILE = 10

SEVERE = 6.0
ANY = 1.0

PROMPT = ("This is a GOES GLM daily thunder-hour map (hours with lightning activity, 0-24 h). "
          "Classify the thunderstorm level: no_storm, thunderstorm, or severe_thunderstorm.")


def load(fp):
    ds = Dataset(fp)
    th = np.asarray(ds.variables["thunder_hours"][:]).astype("float32")  # (lon,lat,time)
    lat = np.asarray(ds.variables["lat"][:])
    lon = np.asarray(ds.variables["lon"][:])
    ds.close()
    daily = np.nansum(th, axis=2)          # 24 小时累加 → (lon, lat)
    daily = np.transpose(daily)            # → (lat, lon)
    return daily, lat, lon


def grade_of(mx):
    if mx >= SEVERE:
        return "severe_thunderstorm"
    if mx >= ANY:
        return "thunderstorm"
    return "no_storm"


def render(arr, lat, lon, la, lo, g, path, stamp, lab, with_grade):
    li = np.where((lat >= la) & (lat < la + g))[0]
    lj = np.where((lon >= lo) & (lon < lo + g))[0]
    sub = arr[np.ix_(li, lj)]
    slat = lat[li]; slon = lon[lj]
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=96)
    im = ax.imshow(sub, origin="lower",
                   extent=[slon[0], slon[-1], slat[0], slat[-1]],
                   cmap="YlOrRd", vmin=0, vmax=12, aspect="auto")
    if with_grade:
        try:
            cs = ax.contour(slon, slat, sub, levels=[ANY, SEVERE], colors="k",
                            linewidths=0.6, linestyles="--")
            ax.clabel(cs, fmt="%.0f", fontsize=5)
        except Exception:
            pass
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  %s" % (stamp, lab), fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("thunder hours / day", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=20.0)
    ap.add_argument("--keep", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    fs = sorted(glob.glob(os.path.join(RAW, "*.nc")))
    if args.keep:
        fs = fs[:args.keep]
    print("待渲染:", len(fs), "| 网格", args.grid)
    if not fs:
        print("无数据"); return

    recs, c = [], {}
    g = args.grid
    for i, f in enumerate(fs):
        m = re.search(r"(\d{8})", os.path.basename(f))
        stamp = m.group(1) if m else str(i)
        try:
            arr, lat, lon = load(f)
        except Exception as e:
            print("  %s 失败 %s" % (stamp, str(e)[:60])); continue

        cells = []
        for la in np.arange(-60, 60, g):
            for lo in np.arange(-180, 180, g):
                li = np.where((lat >= la) & (lat < la + g))[0]
                lj = np.where((lon >= lo) & (lon < lo + g))[0]
                if li.size == 0 or lj.size == 0:
                    continue
                sub = arr[np.ix_(li, lj)]
                if sub.size == 0:
                    continue
                cells.append((float(np.nanmax(sub)), la, lo))
        if not cells:
            continue
        by = {}
        for mx, la, lo in cells:
            by.setdefault(grade_of(mx), []).append((mx, la, lo))
        picked = []
        for lab, lst in by.items():
            lst.sort(key=lambda t: -t[0])
            picked += [(lab, t) for t in lst[:MAX_PER_FILE]]

        for lab, (mx, la, lo) in picked:
            c[lab] = c.get(lab, 0) + 1
            tag = "%s%d_%s%d" % ("N" if la >= 0 else "S", abs(int(la)), "E" if lo >= 0 else "W", abs(int(lo)))
            for sfx, wg, prod in (("_th", False, "intrinsic_thunder_hours"), ("_grade", True, "feature_grade")):
                png = os.path.join(IMG, "%s_%s%s.png" % (stamp, tag, sfx))
                render(arr, lat, lon, la, lo, g, png, stamp, lab, wg)
                ans = {"no_storm": "No significant thunderstorm activity (under 1 thunder-hour).",
                       "thunderstorm": "This region experienced thunderstorm activity (1-6 thunder-hours).",
                       "severe_thunderstorm": "This region experienced severe thunderstorm activity (6 or more thunder-hours)."}[lab]
                recs.append({"conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": ans}],
                    "images": [png], "product": prod, "label": lab, "stamp": stamp})
        print("  [%d/%d] %s" % (i + 1, len(fs), stamp))

    import random, collections
    random.seed(41)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 6))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_thunderstorm.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_thunderstorm_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("分布:", dict(c))
    print("val:", dict(collections.Counter(r["label"] for r in val)))


if __name__ == "__main__":
    main()

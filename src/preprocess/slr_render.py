# -*- coding: utf-8 -*-
r"""
测高（SEALEVEL SLA）——沿海异常高水位渲染 —— 源 8 第一个元素

⚠️ 命名说明：这是「海面高度异常场」，不是严格的「风暴潮」（风暴潮需潮汐基准+气压+风）。
   在 1 年数据下，「海面上升」是长期趋势做不了，故本次做「异常高水位/增水」。
   诚实口径。

双产品：
  本征场 = SLA 海面高度异常场（m）
  特色场 = 增水分级（本地 ±2σ 判定）

判据（本地统计）：
  窗内 max SLA >= +0.3m → high_water（异常高水位）
  >= +0.15m             → elevated（偏高）
  否则                   → normal

输入：data\raw\SEALEVEL_GLO_PHY_L4_MY_008_047\*.nc
输出：datasets\08_altimeter\coastal_high_water\images\*.png + json

用法: python slr_render.py [--grid 30] [--keep N]
"""
import os, re, sys, glob, json, argparse
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from PIL import Image

RAW = r"C:\vlmdisaster\data\raw\SEALEVEL_GLO_PHY_L4_MY_008_047"
OUT = r"C:\vlmdisaster\datasets\08_altimeter\coastal_high_water"
IMG = os.path.join(OUT, "images")
OUT_SIZE = 384
VAL_PER_CLASS = 30
MAX_PER_FILE = 10

HIGH = 0.30
ELEV = 0.15

PROMPT = ("This is an altimetry sea-level-anomaly map (meters) of the ocean. "
          "Classify the coastal water-level condition: normal, elevated, or high_water.")


def load(fp):
    ds = Dataset(fp)
    sla = np.asarray(ds.variables["sla"][0]).astype("float64")
    lat = np.asarray(ds.variables["latitude"][:])
    lon = np.asarray(ds.variables["longitude"][:])
    ds.close()
    sla[np.abs(sla) > 10] = np.nan
    return sla, lat, lon


def grade_of(mx):
    if mx >= HIGH:
        return "high_water"
    if mx >= ELEV:
        return "elevated"
    return "normal"


def render(sla, lat, lon, la, lo, g, path, stamp, lab, with_grade):
    li = np.where((lat >= la) & (lat < la + g))[0]
    lj = np.where((lon >= lo) & (lon < lo + g))[0]
    sub = sla[np.ix_(li, lj)]
    slat = lat[li]; slon = lon[lj]
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=96)
    im = ax.imshow(sub, origin="lower",
                   extent=[slon[0], slon[-1], slat[0], slat[-1]],
                   cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    if with_grade:
        try:
            cs = ax.contour(slon, slat, sub, levels=[ELEV, HIGH], colors="k",
                            linewidths=0.6, linestyles="--")
            ax.clabel(cs, fmt="%.2f", fontsize=5)
        except Exception:
            pass
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  %s" % (stamp, lab), fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("SLA (m)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=30.0)
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
            sla, lat, lon = load(f)
        except Exception as e:
            print("  %s 失败 %s" % (stamp, str(e)[:50])); continue

        cells = []
        for la in np.arange(-60, 60, g):
            for lo in np.arange(-180, 180, g):
                li = np.where((lat >= la) & (lat < la + g))[0]
                lj = np.where((lon >= lo) & (lon < lo + g))[0]
                if li.size == 0 or lj.size == 0:
                    continue
                sub = sla[np.ix_(li, lj)]
                v = sub[np.isfinite(sub)]
                if v.size < sub.size * 0.3:
                    continue
                cells.append((float(np.nanmax(v)), la, lo))
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
            for sfx, wg, prod in (("_sla", False, "intrinsic_sla"), ("_grade", True, "feature_grade")):
                png = os.path.join(IMG, "%s_%s%s.png" % (stamp, tag, sfx))
                render(sla, lat, lon, la, lo, g, png, stamp, lab, wg)
                ans = {"normal": "Sea level anomaly is normal in this region.",
                       "elevated": "Sea level is elevated (above +0.15 m) in this region.",
                       "high_water": "Sea level is anomalously high (above +0.30 m), indicating coastal high-water conditions."}[lab]
                recs.append({"conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": ans}],
                    "images": [png], "product": prod, "label": lab, "stamp": stamp})
        if i % 50 == 0 or i == len(fs) - 1:
            print("  [%d/%d] %s" % (i + 1, len(fs), stamp))

    import random, collections
    random.seed(37)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 8))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_coastal_high_water.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_coastal_high_water_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("分布:", dict(c))
    print("val:", dict(collections.Counter(r["label"] for r in val)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
海冰灾害渲染（源 4/8 · CMEMS SEAICE 海冰浓度）

双产品：
  本征场 = 海冰浓度场（%，EASE2 极地投影 → 经纬度绘制）
  特色场 = 冰情分级（WMO 标准：浓度 >= 15% = 航行危险临界）

判据（WMO / OSI SAF）：
  >= 70%  → heavy_ice（重冰区，航行受阻）
  >= 15%  → ice（有冰，航行危险）
  <  15%  → open_water（无冰危险）

输入：data\raw\SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009\*.nc
输出：datasets\04_passive_microwave\sea_ice\images\*.png + json

用法: python seaice_render.py [--keep N]
"""
import os, re, sys, glob, json, argparse
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from PIL import Image

RAW = r"C:\vlmdisaster\data\raw\SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009"
OUT = r"C:\vlmdisaster\datasets\04_passive_microwave\sea_ice"
IMG = os.path.join(OUT, "images")
OUT_SIZE = 384
VAL_PER_CLASS = 30

ICE = 15.0
HEAVY = 70.0

PROMPT = ("This is a sea-ice concentration map (percent) of the Northern Hemisphere. "
          "Classify the ice condition: open_water, ice, or heavy_ice.")


def load(fp):
    ds = Dataset(fp)
    ice = np.asarray(ds.variables["ice_conc"][0]).astype("float64")
    lat = np.asarray(ds.variables["lat"][:])
    lon = np.asarray(ds.variables["lon"][:])
    ds.close()
    ice[(ice < 0) | (ice > 100)] = np.nan
    lat[(lat < -90) | (lat > 90)] = np.nan
    return ice, lat, lon


def grade_of(ice):
    """按**冰覆盖面积占比**判冰情（而非取极值）。
    口径：在关注海区(55~90N)内，浓度>=15% 的像元占比。
      >=60% → heavy_ice（重冰封海）
      >=20% → ice（有冰，航行危险）
      否则    → open_water（以开阔水域为主）
    """
    v = ice[np.isfinite(ice)]
    if v.size == 0:
        return None, 0.0
    frac = float(np.mean(v >= ICE))
    if frac >= 0.60:
        return "heavy_ice", frac
    if frac >= 0.20:
        return "ice", frac
    return "open_water", frac


def render(ice, lat, lon, path, stamp, lab, with_grade):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=96)
    im = ax.pcolormesh(lon, lat, ice, cmap="Blues", vmin=0, vmax=100, shading="auto")
    if with_grade:
        try:
            cs = ax.contour(lon, lat, ice, levels=[ICE, HEAVY], colors="k",
                            linewidths=0.6, linestyles="--")
            ax.clabel(cs, fmt="%.0f", fontsize=5)
        except Exception:
            pass
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 360)
    ax.set_ylim(55, 90)          # 北半球高纬
    ax.set_title("%s  %s" % (stamp, lab), fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("sea-ice conc (%)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    fs = sorted(glob.glob(os.path.join(RAW, "*.nc")))
    if args.keep:
        fs = fs[:args.keep]
    print("待渲染:", len(fs))
    if not fs:
        print("无数据"); return

    recs = []
    c = {}
    for i, f in enumerate(fs):
        m = re.search(r"(\d{8})", os.path.basename(f))
        stamp = m.group(1) if m else str(i)
        try:
            ice, lat, lon = load(f)
        except Exception as e:
            print("  %s 失败 %s" % (stamp, str(e)[:50])); continue
        v = ice[np.isfinite(ice)]
        if v.size == 0:
            continue
        lab, frac = grade_of(ice)
        if lab is None:
            continue
        c[lab] = c.get(lab, 0) + 1
        for suffix, wg, prod in (("_conc", False, "intrinsic_conc"), ("_grade", True, "feature_grade")):
            png = os.path.join(IMG, "%s%s.png" % (stamp, suffix))
            render(ice, lat, lon, png, stamp, lab, wg)
            ans = {"open_water": "This region is mainly open water; no significant sea ice.",
                   "ice": "This region has sea ice (concentration above 15%%), a navigation hazard.",
                   "heavy_ice": "This region is covered by heavy sea ice (above 70%%), navigation blocked."}[lab]
            recs.append({"conversations": [
                {"from": "human", "value": "<image>" + PROMPT},
                {"from": "gpt", "value": ans}],
                "images": [png], "product": prod, "label": lab, "stamp": stamp})
        if i % 50 == 0 or i == len(fs) - 1:
            print("  [%d/%d] %s => %s (ice area %.0f%%)" % (i + 1, len(fs), stamp, lab, frac * 100))

    # val 切分
    import random, collections
    random.seed(23)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 6))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_sea_ice.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_sea_ice_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("分布:", dict(c))
    print("val:", dict(collections.Counter(r["label"] for r in val)))


if __name__ == "__main__":
    main()

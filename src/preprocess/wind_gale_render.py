# -*- coding: utf-8 -*-
r"""
海上强风 / 大风灾害 渲染（源7 散射计 · CMEMS 风场）—— 区域切窗版

双产品（本征场 + 特色场）：
  本征场  = 10m 风速场（m/s）+ 风向矢量
  特色场  = 大风分级场（蒲福风级：>=17.2 大风 / >=24.5 狂风）

判据（蒲福风级，国际通用）：
  >=24.5 m/s (10级) → storm
  >=17.2 m/s (8级)  → gale
  否则               → normal
每窗用"窗内最大风速"定档 → 同一时刻正负自然都有。

⚠️ 缺陷：散射计强风饱和，内核风速 ~28m/s 封顶，"台风级(>=32.7)"档做不出。

输入：data\raw\07_scatterometer\wind\*.nc（CMEMS 融合风场，0.125°）
输出：datasets\07_scatterometer\ocean_gale\images\*.png + json

用法: python wind_gale_render.py [--grid 30] [--keep N]
"""
import os, re, sys, glob, json, argparse
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from PIL import Image

RAW = r"C:\vlmdisaster\data\raw\07_scatterometer\wind"
OUT = r"C:\vlmdisaster\datasets\07_scatterometer\ocean_gale"
IMG = os.path.join(OUT, "images")
GALE = 17.2      # 8级大风
STORM = 24.5     # 10级狂风
OUT_SIZE = 384
VAL_PER_CLASS = 30
MAX_PER_FILE = 12

PROMPT = ("This is a satellite scatterometer 10-m ocean wind-speed map (m/s) with wind vectors. "
          "Classify the wind hazard level: normal, gale, or storm.")


def load(fp):
    ds = Dataset(fp)
    u = np.asarray(ds.variables["eastward_wind"][0]).astype("float64")
    v = np.asarray(ds.variables["northward_wind"][0]).astype("float64")
    lat = np.asarray(ds.variables["lat"][:]); lon = np.asarray(ds.variables["lon"][:])
    ds.close()
    u = np.where(np.abs(u) > 200, np.nan, u)
    v = np.where(np.abs(v) > 200, np.nan, v)
    return u, v, lat, lon


def grade_of(mx):
    if mx >= STORM:
        return "storm"
    if mx >= GALE:
        return "gale"
    return "normal"


def window_cells(spd, lat, lon, g):
    """按 g°×g° 切窗，每窗 (max_speed, la, lo)"""
    out = []
    for la in np.arange(-80, 80, g):
        for lo in np.arange(-180, 180, g):
            li = np.where((lat >= la) & (lat < la + g))[0]
            lj = np.where((lon >= lo) & (lon < lo + g))[0]
            if li.size == 0 or lj.size == 0:
                continue
            sub = spd[np.ix_(li, lj)]
            v = sub[np.isfinite(sub)]
            if v.size < sub.size * 0.2:
                continue
            # 过滤全 NaN 海域（纯陆地窗会全 NaN）
            out.append((float(np.nanmax(v)), la, lo))
    return out


def render_cell(u, v, lat, lon, la, lo, g, path, stamp, grade, with_grade):
    li = np.where((lat >= la) & (lat < la + g))[0]
    lj = np.where((lon >= lo) & (lon < lo + g))[0]
    su = u[np.ix_(li, lj)]; sv = v[np.ix_(li, lj)]
    spd = np.sqrt(su**2 + sv**2)
    slat = lat[li]; slon = lon[lj]
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=96)
    im = ax.imshow(spd, origin="lower",
                   extent=[slon[0], slon[-1], slat[0], slat[-1]],
                   cmap="turbo", vmin=0, vmax=30, aspect="auto")
    if with_grade:
        cs = ax.contour(slon, slat, spd, levels=[GALE, STORM],
                        colors="k", linewidths=0.7, linestyles="--")
        ax.clabel(cs, fmt="%.0f", fontsize=5)
        ax.set_title("%s  %s" % (stamp, grade), fontsize=8)
    else:
        step = max(1, spd.shape[0] // 12)
        ys = np.arange(0, spd.shape[0], step); xs = np.arange(0, spd.shape[1], step)
        X, Y = np.meshgrid(slon[xs], slat[ys])
        ax.quiver(X, Y, su[np.ix_(ys, xs)], sv[np.ix_(ys, xs)],
                  color="k", scale=400, width=0.0025)
        ax.set_title("%s  speed+vectors" % stamp, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("m/s", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0)
    ap.add_argument("--grid", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    # 清旧图
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(RAW, "*.nc")))
    if args.keep:
        files = files[:args.keep]
    print("待渲染:", len(files), "| 网格", args.grid, "deg")
    if not files:
        print("无数据。"); return

    recs = []
    c = {}
    for i, f in enumerate(files):
        m = re.search(r"(\d{10})", os.path.basename(f))
        stamp = m.group(1) if m else str(i)
        try:
            u, v, lat, lon = load(f)
        except Exception as e:
            print("  %s 失败 %s" % (stamp, str(e)[:50])); continue
        spd = np.sqrt(u**2 + v**2)
        cells = window_cells(spd, lat, lon, args.grid)
        # 每档最多取 MAX_PER_FILE 窗（正负均衡）
        by = {}
        for mx, la, lo in cells:
            by.setdefault(grade_of(mx), []).append((mx, la, lo))
        picked = []
        for lab, lst in by.items():
            lst.sort(key=lambda t: -t[0])
            picked += [(lab, t) for t in lst[:MAX_PER_FILE]]
        for lab, (mx, la, lo) in picked:
            c[lab] = c.get(lab, 0) + 1
            tag = "%s%d_%s%d" % ("N" if la >= 0 else "S", abs(la), "E" if lo >= 0 else "W", abs(lo))
            p1 = os.path.join(IMG, "%s_%s_speed.png" % (stamp, tag))
            p2 = os.path.join(IMG, "%s_%s_grade.png" % (stamp, tag))
            render_cell(u, v, lat, lon, la, lo, args.grid, p1, stamp, lab, False)
            render_cell(u, v, lat, lon, la, lo, args.grid, p2, stamp, lab, True)
            ans = {"normal": "No significant gale in this region; wind speeds are normal.",
                   "gale": "This region has gale-force winds (>=17.2 m/s).",
                   "storm": "This region has storm-force winds (>=24.5 m/s)."}[lab]
            for png, prod in ((p1, "intrinsic_speed"), (p2, "feature_grade")):
                recs.append({"conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": ans}],
                    "images": [png], "product": prod, "label": lab, "stamp": stamp})
        if i % 5 == 0 or i == len(files) - 1:
            print("  [%d/%d] %s" % (i + 1, len(files), stamp))

    json.dump(recs, open(os.path.join(OUT, "train_ocean_gale.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # ---- val 切分（每类 30 张）----
    import random, collections
    random.seed(19)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 5))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_ocean_gale.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_ocean_gale_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("分级分布:", dict(c))
    print("val 分布:", dict(collections.Counter(r["label"] for r in val)))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
重力（GRACE）地下水亏空渲染 —— 源 9 第一个元素

数据：GRACEDADM_CLSM025GL_7D（0.25°，7天，地下水储量百分位 gws_inst）
双产品：
  本征场 = 地下水储量百分位场（%）
  特色场 = 亏空分级（<10% 极度亏空 / 10~30% 偏低 / >30% 正常）

⚠️ 填充值 -999 需过滤。
⚠️ 定位：与 SAR 超采「同灾害、不同源」→ 服务导师的「多源 vs 单源」对比。

输入：data\raw\09_gravity\grace\*.nc4
输出：datasets\09_gravity\groundwater_depletion\images\*.png + json

用法: python grace_render.py [--keep N]
"""
import os, re, sys, glob, json, argparse
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from PIL import Image

RAW = r"C:\vlmdisaster\data\raw\09_gravity\grace"
OUT = r"C:\vlmdisaster\datasets\09_gravity\groundwater_depletion"
IMG = os.path.join(OUT, "images")
OUT_SIZE = 384
VAL_PER_CLASS = 30
MAX_PER_FILE = 6

SEVERE = 10.0    # <10% 极度亏空
LOW = 30.0       # 10~30% 偏低

PROMPT = ("This is a GRACE satellite groundwater-storage percentile map (percent) of the land surface. "
          "Classify the groundwater condition: normal, low, or severe_depletion.")


def load(fp):
    ds = Dataset(fp)
    g = np.asarray(ds.variables["gws_inst"][0]).astype("float64")
    lat = np.asarray(ds.variables["lat"][:])
    lon = np.asarray(ds.variables["lon"][:])
    ds.close()
    g[(g < -100) | (g > 100)] = np.nan     # 滤 -999 填充
    return g, lat, lon


def grade_of(med):
    if med < SEVERE:
        return "severe_depletion"
    if med < LOW:
        return "low"
    return "normal"


def render(g, lat, lon, la, lo, gsz, path, stamp, lab, with_grade):
    li = np.where((lat >= la) & (lat < la + gsz))[0]
    lj = np.where((lon >= lo) & (lon < lo + gsz))[0]
    sub = g[np.ix_(li, lj)]
    slat = lat[li]; slon = lon[lj]
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=96)
    im = ax.imshow(sub, origin="lower",
                   extent=[slon[0], slon[-1], slat[0], slat[-1]],
                   cmap="BrBG", vmin=0, vmax=100, aspect="auto")
    if with_grade:
        try:
            cs = ax.contour(slon, slat, sub, levels=[SEVERE, LOW], colors="k",
                            linewidths=0.6, linestyles="--")
            ax.clabel(cs, fmt="%.0f", fontsize=5)
        except Exception:
            pass
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  %s" % (stamp, lab), fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("groundwater percentile (%)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0)
    ap.add_argument("--grid", type=float, default=30.0)
    args = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    fs = sorted(glob.glob(os.path.join(RAW, "*.nc4")))
    if args.keep:
        fs = fs[:args.keep]
    print("待渲染:", len(fs), "| 网格", args.grid)
    if not fs:
        print("无数据"); return

    recs, c = [], {}
    gsz = args.grid
    for i, f in enumerate(fs):
        m = re.search(r"\.A(\d{8})\.", os.path.basename(f))
        stamp = m.group(1) if m else str(i)
        try:
            g, lat, lon = load(f)
        except Exception as e:
            print("  %s 失败 %s" % (stamp, str(e)[:50])); continue

        cells = []
        for la in np.arange(-60, 60, gsz):
            for lo in np.arange(-180, 180, gsz):
                li = np.where((lat >= la) & (lat < la + gsz))[0]
                lj = np.where((lon >= lo) & (lon < lo + gsz))[0]
                if li.size == 0 or lj.size == 0:
                    continue
                sub = g[np.ix_(li, lj)]
                v = sub[np.isfinite(sub)]
                if v.size < sub.size * 0.3:
                    continue      # 大部分是海/填充，跳过
                cells.append((float(np.median(v)), la, lo))
        if not cells:
            continue
        by = {}
        for med, la, lo in cells:
            by.setdefault(grade_of(med), []).append((med, la, lo))
        picked = []
        for lab, lst in by.items():
            lst.sort(key=lambda t: t[0])      # 亏空最重的优先
            picked += [(lab, t) for t in lst[:MAX_PER_FILE]]

        for lab, (med, la, lo) in picked:
            c[lab] = c.get(lab, 0) + 1
            tag = "%s%d_%s%d" % ("N" if la >= 0 else "S", abs(int(la)), "E" if lo >= 0 else "W", abs(int(lo)))
            for sfx, wg, prod in (("_pct", False, "intrinsic_percentile"), ("_grade", True, "feature_grade")):
                png = os.path.join(IMG, "%s_%s%s.png" % (stamp, tag, sfx))
                render(g, lat, lon, la, lo, gsz, png, stamp, lab, wg)
                ans = {"normal": "Groundwater storage is normal (percentile above 30%).",
                       "low": "Groundwater storage is low (percentile 10-30%).",
                       "severe_depletion": "Groundwater storage is severely depleted (percentile below 10%)."}[lab]
                recs.append({"conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": ans}],
                    "images": [png], "product": prod, "label": lab, "stamp": stamp})
        if i % 100 == 0 or i == len(fs) - 1:
            print("  [%d/%d] %s" % (i + 1, len(fs), stamp))

    import random, collections
    random.seed(31)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 8))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_groundwater_depletion.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_groundwater_depletion_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("分布:", dict(c))
    print("val:", dict(collections.Counter(r["label"] for r in val)))


if __name__ == "__main__":
    main()

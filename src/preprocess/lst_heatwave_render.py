# -*- coding: utf-8 -*-
r"""
高温/热浪（源 3 热红外 · 独立元素）

判据（MODIS LST：白天 P90 > 35°C 且高温像元占比 > 50%）
  heatwave : 白天 P90 > 35°C 且 >35°C 像元占比 >= 50%
  normal   : 其余

双产品：
  本征场 = 白天 LST 温度场（℃）
  特色场 = 高温区标注（>35°C 掩膜叠色）

数据：data\raw\03_thermal_infrared\lst\*.hdf（MOD11A2，2023-2024，8 tile）

用法: python lst_heatwave_render.py [--max N]
"""
import os
import re
import sys
import glob
import json
import argparse
import warnings
import numpy as np
from pyhdf.SD import SD, SDC
from PIL import Image
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = r"C:\vlmdisaster\data\raw\03_thermal_infrared\lst"
OUT = r"C:\vlmdisaster\datasets\03_thermal_infrared\heatwave"

IMG = os.path.join(OUT, "images")
WIN, STRIDE = 256, 128
OUT_SIZE = 384
SCALE = 0.02
VAL_PER_CLASS = 30
MAX_PER_FILE = 20
MIN_VALID = 0.30
HEAT_DAY = 35.0
FRAC_MIN = 0.50

PROMPT = ("This is a MODIS daytime land surface temperature (LST, degrees C) map. "
          "Classify the heat condition: normal or heatwave.")


def load(f):
    h = SD(f, SDC.READ)
    day = h.select("LST_Day_1km")[:].astype("float64")
    qc = h.select("QC_Day")[:]
    h.end()
    day[(day < 7500) | (day > 65535)] = np.nan
    day[qc > 2] = np.nan
    return day * SCALE - 273.15


def draw(day, hot, path, lab, with_mask):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=96)
    im = ax.imshow(day, cmap="RdYlBu_r", vmin=-10, vmax=45)
    if with_mask and hot is not None:
        overlay = np.ma.masked_where(~hot, hot.astype(float))
        ax.imshow(overlay, cmap="autumn_r", alpha=0.45)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(lab, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("LST (degC)", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)
    for f in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(f)

    fs = sorted(glob.glob(os.path.join(RAW, "*.hdf")))
    if args.max:
        fs = fs[:args.max]
    print("文件 %d" % len(fs))
    if not fs:
        print("无数据"); return

    recs, cnt = [], {}
    for i, f in enumerate(fs, 1):
        m = re.search(r"A(\d{7}).*(h\d{2}v\d{2})", os.path.basename(f))
        stamp = (m.group(1) + "_" + m.group(2)) if m else str(i)
        try:
            day = load(f)
        except Exception as e:
            print("  %s 失败 %s" % (stamp, str(e)[:50])); continue
        H, W = day.shape
        by = {}
        for y in range(0, H - WIN, STRIDE):
            for x in range(0, W - WIN, STRIDE):
                sub = day[y:y + WIN, x:x + WIN]
                v = sub[np.isfinite(sub)]
                if v.size < WIN * WIN * MIN_VALID:
                    continue
                p90 = float(np.percentile(v, 90))
                frac = float(np.mean(v > HEAT_DAY))
                lab = "heatwave" if (p90 > HEAT_DAY and frac >= FRAC_MIN) else "normal"
                by.setdefault(lab, []).append((abs(p90 - 35.0), x, y, p90, frac))
        picked = []
        for lab, lst in by.items():
            lst.sort()
            picked += [(lab, t) for t in lst[:MAX_PER_FILE]]

        for lab, (_, x, y, p90, frac) in picked:
            cnt[lab] = cnt.get(lab, 0) + 1
            sub = day[y:y + WIN, x:x + WIN]
            hot = sub > HEAT_DAY
            for sfx, wm, prod in (("_lst", False, "intrinsic_lst"),
                                  ("_hot", True, "feature_hot_mask")):
                png = os.path.join(IMG, "%s_%d_%d_%s%s.png" % (stamp, x, y, lab, sfx))
                draw(sub, hot, png, "%s p90=%.1fC" % (lab, p90), wm)
                ans = ("Heatwave: daytime LST 90th percentile above 35 degC, "
                       "with over 50%% of pixels exceeding 35 degC." if lab == "heatwave"
                       else "Normal thermal conditions: daytime LST below the 35 degC heatwave criterion.")
                recs.append({"conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": ans}],
                    "images": [png], "product": prod, "label": lab,
                    "lst_p90": round(p90, 2), "hot_frac": round(frac, 3), "stamp": stamp})
        if i % 50 == 0:
            print("  %d/%d %s" % (i, len(fs), dict(cnt)), flush=True)

    import random, collections
    random.seed(35)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 6))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_heatwave.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_heatwave_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n图 %d | train %d | val %d" % (len(recs), len(train), len(val)))
    print("分布:", dict(cnt))
    print("val:", dict(collections.Counter(r["label"] for r in val)))


if __name__ == "__main__":
    main()

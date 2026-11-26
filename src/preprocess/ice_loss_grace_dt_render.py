# -*- coding: utf-8 -*-
r"""
冰盖消融 × GRACE（JPL mascon 陆地水当量异常百分位）二分类（新元素：ice_loss_grace_dt）

区域：格陵兰（lat 60–84, lon -75…-10）
逐像元百分位（近 120 个月）：低百分位 = 质量异常偏低（冰/水质量亏损）。
"""
import json
import os
import random
import sys

import numpy as np
from netCDF4 import Dataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "13_grace_mascon", "mascon_global.nc")
OUT = os.path.join(ROOT, "datasets", "13_grace_mascon", "ice_loss_grace_dt")
IMG = os.path.join(OUT, "images")

WIN = 16
STRIDE = 8
MIN_VALID = 0.5
POS_PER_MONTH = 5
NEG_PER_MONTH = 5
SEED = 42
OUT_SIZE = 384
LATB = (60.0, 84.0)
LONB = (-75.0, -10.0)
N_MONTHS = 120

PROMPT = ("This is a GRACE-derived mass-anomaly percentile map (0-100) over Greenland: low values "
          "mean the ice/water mass is unusually low. Output a JSON object with fields: "
          "disaster_type, detected (true/false), source, physical_quantity, severity, criterion. "
          "severity must be one of these exact words: none / ice_loss.")


def main():
    os.makedirs(IMG, exist_ok=True)
    for old in os.listdir(IMG):
        if old.endswith(".png"):
            os.remove(os.path.join(IMG, old))

    ds = Dataset(RAW)
    lwe = np.array(ds.variables["lwe_thickness"][:], dtype="float32")
    lat = np.array(ds.variables["lat"][:], dtype="float64")
    lon = np.array(ds.variables["lon"][:], dtype="float64")
    import cftime
    t = ds.variables["time"]
    times = [str(cftime.num2date(x, t.units, t.calendar))[:7].replace("-", "")
             for x in t[:]]
    ds.close()
    lon2 = np.where(lon > 180, lon - 360, lon)
    rmask = (lat >= LATB[0]) & (lat <= LATB[1])
    cmask = (lon2 >= LONB[0]) & (lon2 <= LONB[1])
    sub = lwe[-N_MONTHS:][:, rmask][:, :, cmask]
    la = lat[rmask]
    lo = lon2[cmask]
    print("sub", sub.shape)

    # 去趋势：逐像元减线性拟合，再取残差的百分位
    D = sub.shape[0]
    tt = np.arange(D, dtype="float64")
    A = np.vstack([tt, np.ones(D)]).T
    flat = sub.reshape(D, -1)
    resid = np.full_like(flat, np.nan)
    for k in range(flat.shape[1]):
        y = flat[:, k]
        ok = np.isfinite(y)
        if ok.sum() < 20:
            continue
        coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
        resid[ok, k] = y[ok] - (A[ok] @ coef)
    resid = resid.reshape(sub.shape)
    pct = np.full_like(resid, np.nan)
    for i in range(D):
        v = resid[i]
        lower = np.nansum(resid < v, axis=0)
        valid = np.sum(np.isfinite(resid), axis=0)
        pct[i] = np.where(valid >= 10, 100.0 * lower / np.maximum(valid, 1), np.nan)

    h, w = pct.shape[1], pct.shape[2]
    records = []
    random.seed(SEED)
    for i in range(D):
        p = pct[i]
        stamp = times[-N_MONTHS:][i]
        pos, neg = [], []
        for r0 in range(0, h - WIN + 1, STRIDE):
            for c0 in range(0, w - WIN + 1, STRIDE):
                wp = p[r0:r0 + WIN, c0:c0 + WIN]
                if np.isfinite(wp).mean() < MIN_VALID:
                    continue
                mp = float(np.nanmean(wp))
                item = (r0, c0, float(la[r0 + WIN // 2]), float(lo[c0 + WIN // 2]), mp)
                if mp <= 10:
                    pos.append(item)
                elif mp >= 40:
                    neg.append(item)
        random.shuffle(pos)
        random.shuffle(neg)
        for det, (r0, c0, la0, lo0, mp) in [(True, x) for x in pos[:POS_PER_MONTH]] + \
                [(False, x) for x in neg[:NEG_PER_MONTH]]:
            sev = "ice_loss" if det else "none"
            nm = "ice_%s_r%d_c%d.png" % (stamp, r0, c0)
            pp = os.path.join(IMG, nm)
            fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
            ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
            im = ax.imshow(p[r0:r0 + WIN, c0:c0 + WIN], cmap="YlGnBu", vmin=0, vmax=100,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la0, lo0), fontsize=8)
            cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("mass percentile", fontsize=7)
            fig.text(0.04, 0.075, "GRACE mascon, 0.5 deg", fontsize=6)
            fig.text(0.04, 0.035, "low = mass deficit", fontsize=6, color="#666")
            fig.savefig(pp, dpi=100)
            plt.close(fig)
            PILImage.open(pp).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(pp)
            ans = {
                "disaster_type": "ice_loss",
                "detected": bool(det),
                "source": "GRACE",
                "physical_quantity": "mass_anomaly_percentile",
                "severity": sev,
                "criterion": "GRACE detrended mass-anomaly percentile <= 10 = unusual mass deficit",
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp,
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        if (i + 1) % 20 == 0:
            print("  [%d/%d] 累计 %d" % (i + 1, D, len(records)))

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
        n = min(max(2, int(round(len(lst) * 0.2))), 60, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])
    json.dump(train, open(os.path.join(OUT, "train_ice_loss_grace_dt_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_ice_loss_grace_dt_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()

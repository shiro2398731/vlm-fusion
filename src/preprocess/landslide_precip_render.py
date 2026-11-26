# -*- coding: utf-8 -*-
r"""
滑坡 × 降水（IMERG 3 天累计）二分类（新元素：landslide_precip）

事件来源：ICIMOD HKH 滑坡目录（公开 ArcGIS，带 ev_date 与坐标）
任务：severity none / landslide；detected 与 severity 一致。
窗口：3 天累计降水（事件日 ±1），32 像素（3.2°）。
"""
import datetime
import glob
import json
import os
import random
import re
import sys

import numpy as np
from netCDF4 import Dataset

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
RAW = os.path.join(ROOT, "data", "raw", "06_radar_precipitation", "imerg_landslide")
EVENTS = os.path.join(ROOT, "data", "raw", "landslide_icimod.json")
OUT = os.path.join(ROOT, "datasets", "06_radar_precipitation", "landslide_precip")
IMG = os.path.join(OUT, "images")

WIN = 32
STRIDE = 8
MIN_VALID = 0.4
POS_PER_EVENT = 5
NEG_PER_EVENT = 4
SEED = 42
OUT_SIZE = 384

PROMPT = ("This is a 3-day accumulated precipitation map (mm) from GPM IMERG around a landslide "
          "event. Heavy rain indicates landslide risk. Output a JSON object with fields: "
          "disaster_type, detected (true/false), source, physical_quantity, severity, criterion. "
          "severity must be one of these exact words: none / landslide.")


def load_precip(fp):
    ds = Dataset(fp)
    p = np.array(ds.variables["precipitation"][0], dtype="float32").T
    lat = np.array(ds.variables["lat"][:], dtype="float64")
    lon = np.array(ds.variables["lon"][:], dtype="float64")
    ds.close()
    p[(p < 0) | (p > 1000)] = np.nan
    return p, lat, lon


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="Blues", vmin=0, vmax=800, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("3-day precip (mm)", fontsize=7)
    fig.text(0.04, 0.075, "GPM IMERG daily, 0.1 deg", fontsize=6)
    fig.text(0.04, 0.035, "white = no data", fontsize=6, color="#666")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    files = {}
    for f in glob.glob(os.path.join(RAW, "*.nc4")):
        m = re.search(r"3IMERG\.(\d{8})-", os.path.basename(f))
        if m:
            files[m.group(1)] = f
    print("IMERG 日文件 %d" % len(files))

    ev = json.load(open(EVENTS, encoding="utf-8"))
    ev = [e for e in ev if e.get("trigger") in ("downpour", "continuous_rain") and e.get("ev_date")]
    ev.sort(key=lambda e: -(e.get("fatalities") or 0))
    print("滑坡事件 %d" % len(ev))

    cache = {}

    def cum3(d):
        arr = None
        for k in (-1, 0, 1):
            s = (d + datetime.timedelta(days=k)).strftime("%Y%m%d")
            if s not in files:
                return None
            if s not in cache:
                cache[s] = load_precip(files[s])
            p, lat, lon = cache[s]
            arr = p.copy() if arr is None else arr + p
        return arr, lat, lon

    records = []
    random.seed(SEED)
    used = 0
    for e in ev:
        d0 = datetime.datetime.utcfromtimestamp(e["ev_date"] / 1000).date()
        got = cum3(d0)
        if got is None:
            continue
        cum, lat, lon = got
        h, w = cum.shape
        elat, elon = float(e["lat"]), float(e["lon"])
        r0 = int(np.clip((90 - elat) / 0.1, 0, h - WIN))
        c0 = int(np.clip((elon + 180) / 0.1, 0, w - WIN))
        pos, neg = [], []
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                rr = int(np.clip(r0 + dr * STRIDE, 0, h - WIN))
                cc = int(np.clip(c0 + dc * STRIDE, 0, w - WIN))
                win = cum[rr:rr + WIN, cc:cc + WIN]
                if np.isfinite(win).mean() < MIN_VALID:
                    continue
                mx = float(np.nanmax(win))
                la = float(np.nanmean(lat[rr:rr + WIN]))
                lo = float(np.nanmean(lon[cc:cc + WIN]))
                if mx >= 50:
                    pos.append((rr, cc, la, lo, mx))
        for _ in range(200):
            if len(neg) >= NEG_PER_EVENT:
                break
            rr = random.randrange(0, h - WIN + 1, STRIDE * 4)
            cc = random.randrange(0, w - WIN + 1, STRIDE * 4)
            win = cum[rr:rr + WIN, cc:cc + WIN]
            if np.isfinite(win).mean() < MIN_VALID:
                continue
            mx = float(np.nanmax(win))
            if mx >= 20:
                continue
            la = float(np.nanmean(lat[rr:rr + WIN]))
            lo = float(np.nanmean(lon[cc:cc + WIN]))
            neg.append((rr, cc, la, lo, mx))
        pos = sorted(pos, key=lambda x: -x[4])[:POS_PER_EVENT]
        random.shuffle(neg)
        stamp = d0.strftime("%Y%m%d")
        for detection, (rr, cc, la, lo, mx) in [(True, x) for x in pos] + [(False, x) for x in neg[:NEG_PER_EVENT]]:
            sev = "landslide" if detection else "none"
            nm = "ls_%s_r%d_c%d.png" % (stamp, rr, cc)
            pp = os.path.join(IMG, nm)
            render(cum[rr:rr + WIN, cc:cc + WIN], pp, stamp, la, lo)
            ans = {
                "disaster_type": "landslide",
                "detected": bool(detection),
                "source": "PRECIP",
                "physical_quantity": "cum3_precipitation",
                "severity": sev,
                "criterion": ("3-day accumulated precipitation >= 50 mm near a landslide event "
                              "(ICIMOD HKH catalog); binary none/landslide"),
            }
            records.append({
                "image": pp, "label": sev, "stamp": stamp,
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        used += 1
        if used % 5 == 0:
            print("  事件 %d 累计 %d" % (used, len(records)), flush=True)

    seen = set()
    uniq = []
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
        n = min(max(2, int(round(len(lst) * 0.2))), 80, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])
    json.dump(train, open(os.path.join(OUT, "train_landslide_precip_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_landslide_precip_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()

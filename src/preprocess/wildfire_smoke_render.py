# -*- coding: utf-8 -*-
r"""
野火烟羽：FIRMS 火点 + OMI UVAI 切窗（新元素：wildfire_smoke）

物理链路
--------
  MODIS/VIIRS 热异常（FIRMS 火点）→ 烟羽 → OMI 紫外吸收性气溶胶指数 UVAI 升高
真值
----
  FIRMS 火点位置（中国 2024，confidence 非低）；正样本 = 火点附近 UVAI 窗
任务
----
  detected：窗口是否在火点集群附近（真值）
  数值字段：max_uvai（见附图，只能从图上读）
  severity：<1 none / 1-2 light / 2-4 moderate / >=4 heavy

用法
----
  .\venv\Scripts\python.exe src\preprocess\wildfire_smoke_render.py --limit-days 5
  .\venv\Scripts\python.exe src\preprocess\wildfire_smoke_render.py
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from PIL import Image as PILImage

ROOT = r"C:\vlmdisaster"
FIRMS = os.path.join(ROOT, "data", "raw", "03_thermal_infrared", "firms", "chn")
UVAI = os.path.join(ROOT, "data", "raw", "11_atmospheric", "omi_uvai")
OUT = os.path.join(ROOT, "datasets", "11_atmospheric", "wildfire_smoke")
IMG = os.path.join(OUT, "images")

WIN = 8
STRIDE = 8
MIN_VALID = 0.3
POS_PER_DAY = 4
NEG_PER_DAY = 4
UVAI_MIN = 1.0
SEED = 42
OUT_SIZE = 384

PROMPT = ("This is an OMI UV Aerosol Index (UVAI) map, dimensionless, over a region with active "
          "fires. UVAI > 1 indicates absorbing aerosol such as smoke. Output a JSON object "
          "with fields: disaster_type, detected (true/false), source, physical_quantity, "
          "severity, max_uvai, criterion. severity must be one of these exact words: "
          "none / light / moderate / heavy.")


def severity_of(v):
    if v < 1:
        return "none"
    if v < 2:
        return "light"
    if v < 4:
        return "moderate"
    return "heavy"


def load_uvai(fp):
    ds = Dataset(fp)
    g = ds.groups["HDFEOS"].groups["GRIDS"].groups["Aerosol NearUV Grid"]
    a = np.ma.filled(np.array(g.groups["Data Fields"].variables["UVAerosolIndex"][:],
                              dtype="float64"), np.nan)
    ds.close()
    a[(a < -100) | (a > 100)] = np.nan
    return a


def load_fires(fp):
    fires = []
    with open(fp, encoding="utf-8", errors="replace") as f:
        head = f.readline().strip().split(",")
        ix = {k: i for i, k in enumerate(head)}
        for line in f:
            p = line.strip().split(",")
            if len(p) < len(head):
                continue
            try:
                la = float(p[ix["latitude"]])
                lo = float(p[ix["longitude"]])
                conf = p[ix.get("confidence", -1)]
                frp = float(p[ix["frp"]]) if p[ix.get("frp", -1)] else 0.0
            except (ValueError, KeyError, IndexError):
                continue
            if conf in ("low", "l"):
                continue
            fires.append((la, lo, frp))
    return fires


def render(win, path, stamp, la, lo):
    fig = plt.figure(figsize=(4.2, 4.2), dpi=100)
    ax = fig.add_axes([0.04, 0.14, 0.80, 0.76])
    im = ax.imshow(win, cmap="YlOrRd", vmin=-0.5, vmax=3, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  (%.1fN %.1fE)" % (stamp, la, lo), fontsize=8)
    cax = fig.add_axes([0.86, 0.14, 0.035, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("UVAI (dimensionless)", fontsize=7)
    fig.text(0.04, 0.075, "OMI/Aura OMAERUVd, daily 1 deg", fontsize=6)
    fig.text(0.04, 0.035, "UVAI > 1 = absorbing aerosol (smoke/dust)", fontsize=6, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for old in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(old)

    uv = {}
    for f in glob.glob(os.path.join(UVAI, "*.he5")):
        m = re.search(r"(\d{4})m(\d{2})(\d{2})", os.path.basename(f))
        if m:
            uv[m.group(1) + m.group(2) + m.group(3)] = f

    fire_files = sorted(glob.glob(os.path.join(FIRMS, "fire_2024-*.csv")))
    days = []
    for fp in fire_files:
        d = re.search(r"fire_(\d{4})-(\d{2})-(\d{2})", os.path.basename(fp))
        stamp = d.group(1) + d.group(2) + d.group(3)
        if stamp in uv:
            days.append((stamp, fp))
    if a.limit_days:
        days = days[:a.limit_days]
    print("可用火点日 %d 个" % len(days))

    records = []
    random.seed(SEED)
    for i, (stamp, fp) in enumerate(days, 1):
        fires = load_fires(fp)
        if len(fires) < 3:
            continue
        arr = load_uvai(uv[stamp])
        h, w = arr.shape
        # 火点聚成 2° 网格
        cl = collections.Counter()
        for (la, lo, frp) in fires:
            cl[(int(la // 2) * 2, int(lo // 2) * 2)] += 1
        centers = [k for k, n in cl.most_common(8) if n >= 2]
        pos = []
        for (cla, clo) in centers:
            r0 = int(np.clip(round(cla + 90 - WIN // 2), 0, h - WIN))
            c0 = int(np.clip(round(clo + 180 - WIN // 2), 0, w - WIN))
            win = arr[r0:r0 + WIN, c0:c0 + WIN]
            if np.isfinite(win).mean() < MIN_VALID:
                continue
            mx = float(np.nanmax(win))
            if mx < UVAI_MIN:
                continue
            pos.append((r0, c0, cla + 1, clo + 1, mx))
        neg = []
        for _ in range(300):
            if len(neg) >= NEG_PER_DAY:
                break
            rr = random.randrange(0, h - WIN + 1, STRIDE)
            cc = random.randrange(0, w - WIN + 1, STRIDE)
            la = -89.5 + (rr + WIN / 2) * 1.0
            lo = -179.5 + (cc + WIN / 2) * 1.0
            near = any(abs(la - cla) < 15 and abs(lo - clo) < 20 for (cla, clo) in cl)
            if near:
                continue
            ww = arr[rr:rr + WIN, cc:cc + WIN]
            if np.isfinite(ww).mean() < MIN_VALID:
                continue
            mx = float(np.nanmax(ww))
            if mx >= 0.5:
                continue
            neg.append((rr, cc, la, lo, mx))
        random.shuffle(pos)
        for det, (rr, cc, la, lo, mx) in [(True, p) for p in pos[:POS_PER_DAY]] + \
                [(False, p) for p in neg]:
            sev = severity_of(mx)
            nm = "smoke_%s_r%d_c%d.png" % (stamp, rr, cc)
            p = os.path.join(IMG, nm)
            render(arr[rr:rr + WIN, cc:cc + WIN], p, stamp, la, lo)
            ans = {
                "disaster_type": "wildfire_smoke",
                "detected": bool(det),
                "source": "ATM",
                "physical_quantity": "uv_aerosol_index",
                "severity": sev,
                "max_uvai": round(mx, 2),
                "criterion": ("fire detections from FIRMS (confidence not low); "
                              "UVAI >= 1 indicates absorbing smoke aerosol"),
            }
            records.append({
                "image": p, "label": sev, "stamp": stamp, "max_uvai": round(mx, 2),
                "conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": json.dumps(ans, ensure_ascii=False)},
                ],
            })
        if i % 20 == 0 or i == len(days):
            print("  [%d/%d] %s 累计 %d" % (i, len(days), stamp, len(records)))

    seen = set()
    uniq = []
    for r in records:
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        uniq.append(r)
    random.seed(SEED)
    by = collections.defaultdict(list)
    for r in uniq:
        by[r["label"]].append(r)
    val, train = [], []
    for k in sorted(by):
        lst = by[k][:]
        random.shuffle(lst)
        n = min(max(2, int(round(len(lst) * 0.2))), 30, len(lst))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_wildfire_smoke_json.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_wildfire_smoke_json_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 train %d | val %d" % (len(train), len(val)))
    print("标签分布:", {k: len(v) for k, v in sorted(by.items())})
    print("输出:", OUT)


if __name__ == "__main__":
    main()

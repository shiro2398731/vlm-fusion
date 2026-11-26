# -*- coding: utf-8 -*-
r"""
地磁暴（源 12 空间天气）

判据（WDC Kyoto Dst 指数，国际标准，NOAA G1-G5 对应）：
  Dst <= -350 → extreme_storm  （G5 超大磁暴）
  Dst <= -200 → severe_storm   （G4 强磁暴）
  Dst <= -100 → moderate_storm （G3 中等磁暴）
  Dst <=  -50 → weak_storm     （G2/G1 弱磁暴）
  Dst >   -50 → quiet

数据：WDC Kyoto（dst_realtime 当年 / dst_provisional 近两年 / dst_final 更早）
格式：DSTYYMM*DD[PPX120] + 24 小时值（nT）

双产品：
  本征场 = Dst 时序曲线（nT）
  特色场 = Dst 分级填色（阈值线 + 色带）

用法:
  python geomagnetic_render.py --start 2024-01-01 --end 2024-12-31
"""
import os
import re
import sys
import json
import argparse
import warnings
import numpy as np
from datetime import date, datetime, timedelta
import requests
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RAW = r"C:\vlmdisaster\data\raw\12_space_weather\dst"
OUT = r"C:\vlmdisaster\datasets\12_space_weather\geomagnetic_storm"
IMG = os.path.join(OUT, "images")
OUT_SIZE = 384
VAL_PER_CLASS = 30

S = requests.Session(); S.trust_env = False
S.headers["User-Agent"] = "Mozilla/5.0"

TH = [(-350, "extreme_storm"), (-200, "severe_storm"),
      (-100, "moderate_storm"), (-50, "weak_storm")]
PROMPT = ("This is a Dst index time series (geomagnetic disturbance, nT) over 24 hours. "
          "Classify the geomagnetic storm level: quiet, weak_storm, moderate_storm, "
          "severe_storm, or extreme_storm.")


def fetch_month(year, month):
    """下载某月 Dst 文本，缓存本地"""
    os.makedirs(RAW, exist_ok=True)
    local = os.path.join(RAW, "dst_%d%02d.txt" % (year, month))
    if os.path.exists(local) and os.path.getsize(local) > 500:
        return open(local, encoding="utf-8", errors="replace").read()
    ym = "%d%02d" % (year, month)
    for base in ("dst_realtime", "dst_provisional", "dst_final"):
        u = "https://wdc.kugi.kyoto-u.ac.jp/%s/%s/dst%s.for.request" % (base, ym, ym[2:])
        try:
            r = S.get(u, timeout=90)
            if r.status_code == 200 and len(r.content) > 500:
                open(local, "w", encoding="utf-8").write(r.text)
                return r.text
        except Exception:
            pass
    return None


def parse(txt):
    """返回 {date: [24 个值，缺测 None]}"""
    out = {}
    for line in txt.split("\n"):
        m = re.match(r"DST(\d{2})(\d{2})\*(\d{2})", line)
        if not m:
            continue
        yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = 2000 + yy if yy < 50 else 1900 + yy
        vals = []
        # 固定宽度格式：每个值 4 字符，从第 20 列开始
        body = line[20:]
        for i in range(24):
            chunk = body[i * 4:(i + 1) * 4].strip()
            if not chunk:
                vals.append(None)
                continue
            try:
                v = int(chunk)
                vals.append(None if abs(v) == 999 else v)
            except ValueError:
                vals.append(None)
        try:
            d = date(year, mm, dd)
        except ValueError:
            continue
        out[d] = vals[:24]
    return out


def grade(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    mn = min(v)
    for th, lab in TH:
        if mn <= th:
            return lab, mn
    return "quiet", mn


def draw(vals, path, lab, filled):
    hrs = np.arange(24)
    y = np.array([np.nan if v is None else v for v in vals], dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=96)
    if filled:
        ax.fill_between(hrs, y, 0, where=(y < 0), color="#c0392b", alpha=0.55, interpolate=True)
        for th, _ in TH:
            ax.axhline(th, color="#333", lw=0.6, ls="--")
            ax.text(0.2, th + 8, "%d" % th, fontsize=5, color="#333")
    ax.plot(hrs, y, marker="o", ms=2.2, lw=1.1, color="#1f3a93")
    ax.set_xlim(0, 23)
    ax.set_ylim(min(-450, float(np.nanmin(y)) - 40) if np.isfinite(y).any() else -450, 60)
    ax.set_xlabel("Hour (UT)", fontsize=7)
    ax.set_ylabel("Dst (nT)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(lab, fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR).save(path)


ANS = {"quiet": "Quiet geomagnetic conditions: Dst above -50 nT all day.",
       "weak_storm": "Weak geomagnetic storm (G1-G2): minimum Dst between -50 and -100 nT.",
       "moderate_storm": "Moderate geomagnetic storm (G3): minimum Dst between -100 and -200 nT.",
       "severe_storm": "Severe geomagnetic storm (G4): minimum Dst between -200 and -350 nT.",
       "extreme_storm": "Extreme geomagnetic storm (G5): minimum Dst below -350 nT."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--per-class", type=int, default=120)
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for f in os.listdir(IMG):
        if f.endswith(".png"):
            os.remove(os.path.join(IMG, f))

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)
    months = sorted({(d0.year, d0.month), (d1.year, d1.month)})
    cur = date(d0.year, d0.month, 1)
    months = []
    while cur <= d1:
        months.append((cur.year, cur.month))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

    data = {}
    for (y, m) in months:
        txt = fetch_month(y, m)
        if not txt:
            print("  %d-%02d 无数据" % (y, m)); continue
        data.update(parse(txt))
        print("  %d-%02d 解析 %d 天" % (y, m, len(parse(txt))), flush=True)

    days = sorted(d for d in data if d0 <= d <= d1)
    print("\n区间内 %d 天" % len(days))

    by = {}
    for d in days:
        lab, mn = grade(data[d])
        if lab:
            by.setdefault(lab, []).append((d, data[d], mn))

    print("分布:", {k: len(v) for k, v in by.items()})

    import random
    random.seed(11)
    recs, cnt = [], {}
    for lab, lst in by.items():
        random.shuffle(lst)
        for d, vals, mn in lst[:args.per_class]:
            cnt[lab] = cnt.get(lab, 0) + 1
            for sfx, filled, prod in (("_curve", False, "intrinsic_dst_curve"),
                                      ("_grade", True, "feature_dst_level")):
                png = os.path.join(IMG, "%s_%s%s.png" % (d.isoformat(), lab, sfx))
                draw(vals, png, "%s  min=%.0f nT" % (d.isoformat(), mn), filled)
                recs.append({"conversations": [
                    {"from": "human", "value": "<image>" + PROMPT},
                    {"from": "gpt", "value": ANS[lab]}],
                    "images": [png], "product": prod, "label": lab,
                    "min_dst": int(mn), "date": d.isoformat()})

    import collections
    by2 = collections.defaultdict(list)
    for r in recs:
        by2[r["label"]].append(r)
    val, train = [], []
    # 稀有类（extreme/severe）优先保证进 train：至少留一半给训练
    for lab, lst in by2.items():
        random.shuffle(lst)
        n = len(lst)
        if n <= 8:                      # 稀有类：留至少 n-2 个进 train
            k = max(1, min(2, n // 3))
        else:
            k = min(VAL_PER_CLASS, max(3, n // 6))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_geomagnetic.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_geomagnetic_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n图 %d | train %d | val %d" % (len(recs), len(train), len(val)))
    print("分布:", dict(cnt))
    print("val:", dict(collections.Counter(r["label"] for r in val)))


if __name__ == "__main__":
    main()

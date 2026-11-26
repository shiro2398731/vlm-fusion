# -*- coding: utf-8 -*-
r"""
太阳耀斑（源 12 空间天气）—— SDO AIA 193Å 图像 + HEK 耀斑等级

判据：HEK 的 fl_goescls（GOES 软 X 射线等级，国际标准）
  X 级  → x_flare
  M 级  → m_flare
  C 级  → c_flare
  B 级及以下 → quiet

流程：
  1) 读 HEK 耀斑表（已存 json）
  2) 按耀斑峰值时刻，找 SDO AIA 193Å 最近的整 10 分钟帧（±20 分钟窗口）
  3) 下载图像 → 渲染（本征场 = 193Å 原始光球/日冕图像；特色场 = 耀斑区标注）

用法:
  python flare_download.py [--max X] [--frames 3]
"""
import os
import re
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
import requests
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
S = requests.Session(); S.trust_env = False

RAW = r"C:\vlmdisaster\data\raw\12_space_weather\sdo"
OUT = r"C:\vlmdisaster\datasets\12_space_weather\solar_flare"
IMG = os.path.join(OUT, "images")
HEK = r"C:\vlmdisaster\data\raw\12_space_weather\hek_flares_2024.json"
BASE = "https://sdo.gsfc.nasa.gov/assets/img/browse/%s/"
RES = "1024"
BAND = "0193"
SIZE = 384


def grade(cls):
    c = (cls or "").strip().upper()
    if not c:
        return None
    return {"X": "x_flare", "M": "m_flare", "C": "c_flare"}.get(c[0], "quiet")


def list_day(day):
    """返回当天 193Å 可用时刻列表 [(hhmmss, url)]"""
    u = BASE % day.strftime("%Y/%m/%d")
    try:
        r = S.get(u, timeout=60)
    except Exception:
        return []
    pat = re.compile(r'href="(\d{8})_(\d{6})_%s_%s\.jpg"' % (RES, BAND))
    out = []
    for m in pat.finditer(r.text):
        out.append((m.group(2), u + "%s_%s_%s_%s.jpg" % (m.group(1), m.group(2), RES, BAND)))
    return out


def nearest(frames, when):
    """找最接近 when 的帧"""
    best, bd = None, None
    for hhmmss, url in frames:
        t = datetime(when.year, when.month, when.day,
                     int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:6]))
        d = abs((t - when).total_seconds())
        if bd is None or d < bd:
            best, bd = url, d
    return best, bd


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 20000:
        return True
    for a in range(3):
        try:
            r = S.get(url, timeout=60)
            if r.status_code != 200:
                time.sleep(2); continue
            with open(path, "wb") as f:
                f.write(r.content)
            return True
        except Exception:
            time.sleep(2)
    return False


def render(src, dst, title, mark=False, peak=None):
    im = Image.open(src).convert("RGB")
    im = im.resize((SIZE, SIZE), Image.BILINEAR)
    if mark and peak:
        # hpc 坐标（角秒，中心 0,0），SDO 1024 图视场约 ±1000"
        d = ImageDraw.Draw(im)
        cx, cy = peak
        x = SIZE / 2 + cx / 1000.0 * (SIZE / 2)
        y = SIZE / 2 - cy / 1000.0 * (SIZE / 2)
        rr = 18
        d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=(255, 80, 40), width=2)
    im.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="最多处理多少个耀斑（0=全部）")
    ap.add_argument("--frames", type=int, default=1, help="每个耀斑取几帧（前后扩展）")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    for f in os.listdir(IMG):
        if f.endswith(".png"):
            os.remove(os.path.join(IMG, f))
    os.makedirs(RAW, exist_ok=True)

    rows = json.load(open(HEK, encoding="utf-8"))
    # 去重（HEK 有重复条目）
    seen, ev = set(), []
    for r in rows:
        k = (r.get("event_peaktime"), r.get("fl_goescls"))
        if k in seen:
            continue
        seen.add(k)
        g = grade(r.get("fl_goescls"))
        if g:
            ev.append((r, g))
    print("去重后耀斑:", len(ev))
    if args.max:
        ev = ev[:args.max]

    cache = {}
    recs = []
    cnt = {}
    for i, (r, lab) in enumerate(ev, 1):
        try:
            peak = datetime.fromisoformat(r["event_peaktime"])
        except Exception:
            continue
        day = peak.date()
        if day not in cache:
            cache[day] = list_day(day)
        frames = cache[day]
        if not frames:
            print("  %s 无图像" % day); continue

        url, delta = nearest(frames, peak)
        if url is None or delta > 1800:
            continue

        stamp = peak.strftime("%Y%m%d_%H%M%S")
        raw = os.path.join(RAW, "%s.jpg" % stamp)
        if not download(url, raw):
            continue

        hpc = (r.get("hpc_x") or 0, r.get("hpc_y") or 0)
        for sfx, mark, prod in (("", False, "intrinsic_aia193"),
                                ("_loc", True, "feature_flare_location")):
            png = os.path.join(IMG, "%s_%s%s.png" % (stamp, lab, sfx))
            render(raw, png, stamp, mark, hpc)
            ans = {"x_flare": "X-class solar flare (major, GOES class X).",
                   "m_flare": "M-class solar flare (moderate, GOES class M).",
                   "c_flare": "C-class solar flare (minor, GOES class C).",
                   "quiet": "No significant flare activity (B class or below)."}[lab]
            recs.append({"conversations": [
                {"from": "human", "value": "<image>This is an SDO AIA 193Å EUV image of the Sun. "
                                            "Classify the solar activity: quiet, c_flare, m_flare, or x_flare."},
                {"from": "gpt", "value": ans}],
                "images": [png], "product": prod, "label": lab,
                "goes_class": r.get("fl_goescls"), "peak": peak.isoformat()})
        cnt[lab] = cnt.get(lab, 0) + 1
        if i % 20 == 0:
            print("  %d/%d  %s" % (i, len(ev), dict(cnt)), flush=True)

    import random, collections
    random.seed(13)
    by = collections.defaultdict(list)
    for x in recs:
        by[x["label"]].append(x)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(30, max(2, len(lst) // 6))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(OUT, "train_flare.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_flare_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n事件 %d | train %d | val %d" % (len(recs) // 2, len(train), len(val)))
    print("分布:", dict(cnt))
    print("val:", dict(collections.Counter(x["label"] for x in val)))


if __name__ == "__main__":
    main()

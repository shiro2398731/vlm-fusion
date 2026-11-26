# -*- coding: utf-8 -*-
r"""
光学绿潮 · VIIRS VJ102MOD 流水线（源 1 光学 · 赤潮/绿潮）

为什么用 FAI
-----------
  FAI（Floating Algae Index，漂浮藻类指数）：
      FAI = R_NIR − [ R_red + (R_SWIR − R_red)·(λ_NIR − λ_red)/(λ_SWIR − λ_red) ]
  近红外反射率减去"红—短波红外"基线在近红外处的线性插值。
  漂浮大型藻类（浒苔）在海面成厚层，近红外反射率显著高于该基线 → FAI > 0。

  出处：Hu C., 2009, JGR-Oceans（黄海浒苔绿潮 FAI 原始文献）
        Hu C. et al., 2023（绿潮遥感综述与验证，90 次引用）

  Hu 2009 的公式对传感器是通用的，换 VIIRS 只需换波长：
      MODIS  红 645 / NIR 859 / SWIR 2130
      VIIRS  红 640 (M05) / NIR 865 (M07) / SWIR 2250 (M11)
  插值权重按实际波长计算，不需要改算法。

为什么不用 NDCI
--------------
  NDCI = (B05−B04)/(B05+B04) 在 Sentinel-2 上是"实验性指in数"（Digital Earth
  Africa 官方说法），阈值需另行标定。FAI 的物理机制更直接：测的是海面漂浮物
  在近红外的强反射，不依赖叶绿素浓度的经验关系。

区域与时段
---------
  黄海浒苔：源头江苏盐城外海（5 月下旬起），随海流向北漂至山东半岛（6~7 月）
  框1  120.5~122.5E, 33.0~35.0N   盐城外海（源头）
  框2  120.0~123.0E, 35.0~37.0N   山东半岛南部（漂移区）
  时段 2023-05-20 ~ 2023-07-31

分级（逐景自适应阈值，Hu 2009 做法）
----------------------------------
  阈值 = 该景水体 FAI 的中位数 + 2×标准差
  这是 Hu 2009 的原始做法：FAI 绝对值受大气订正残差与太阳角度影响，
  绝对阈值不可跨场景移植，必须逐景标定。

  占比分档（窗口内超阈水体像元占比）
    >= 0.15   green_tide    明显绿潮覆盖
    0.03~0.15 patchy        零星漂浮藻斑
    < 0.03    clean_water   无绿潮

双产品
------
  本征场 = FAI 场
  特色场 = 真彩色（M05/M04/M03）

用法
----
  python src/preprocess/viirs_green_tide_pipeline.py --days 1 --keep
  python src/preprocess/viirs_green_tide_pipeline.py --days 1
  python src/preprocess/viirs_green_tide_pipeline.py
"""
import os
import re
import sys
import json
import time
import glob
import argparse
import datetime as dt
import collections
import random

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import h5py
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\viirs_green_tide"
L1B = os.path.join(RAW, "l1b")
GEO = os.path.join(RAW, "geo")
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\green_tide"
IMG = os.path.join(OUT, "images")

URS = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
CID_L1B = "C2103811746-LAADS"      # VJ102MOD
CID_GEO = "C2105084593-LAADS"      # VJ103MOD
USER = os.environ.get("EARTHDATA_USER", "")
PASS = os.environ.get("EARTHDATA_PASS", "")
HDF5_MAGIC = b"\x89HDF"
MIN_BYTES = 20_000_000

BOXES = [
    (120.5, 122.5, 33.0, 35.0),    # 盐城外海（源头）
    (120.0, 123.0, 35.0, 37.0),    # 山东半岛南部（漂移区）
]
START = "2023-05-20"
END = "2023-07-31"
LIMIT_PER_DAY = 3

WIN = 1024
MAXI = WIN
OUT_SIZE = 384
VAL_PER_CLASS = 30
MAX_PER_CLASS = 120

# VIIRS M 波段中心波长（nm）
LAM = {"M05": 640.0, "M07": 865.0, "M11": 2250.0}

G_TIDE = 0.15
G_PATCHY = 0.03

S = requests.Session()
S.trust_env = False
_TOK = {"v": None, "t": 0.0}


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def token(force=False):
    now = time.time()
    if not force and _TOK["v"] and now - _TOK["t"] < 1800:
        return _TOK["v"]
    for _ in range(4):
        try:
            r = S.post(URS, auth=(USER, PASS), timeout=60)
            if r.status_code == 200:
                t = r.json().get("access_token")
                if t:
                    _TOK["v"] = t
                    _TOK["t"] = now
                    return t
        except Exception:
            pass
        time.sleep(5)
    return None


def search(day, box, cid, limit):
    params = {
        "collection_concept_id": cid,
        "temporal": "%sT00:00:00Z,%sT23:59:59Z" % (day.isoformat(), day.isoformat()),
        "bounding_box": "%.2f,%.2f,%.2f,%.2f" % (box[0], box[2], box[1], box[3]),
        "page_size": 50,
    }
    for _ in range(3):
        try:
            r = S.get(CMR, params=params, timeout=90)
            if r.status_code != 200:
                time.sleep(3)
                continue
            out = []
            for e in r.json().get("feed", {}).get("entry", []):
                url = None
                for l in e.get("links", []):
                    h = l.get("href", "")
                    if "prod-lads" in h and h.endswith(".nc"):
                        url = h
                        break
                if url:
                    out.append({"filename": os.path.basename(url),
                                "size_mb": float(e.get("granule_size") or 0),
                                "url": url})
            out.sort(key=lambda x: x["filename"])
            return out[:limit]
        except Exception:
            time.sleep(3)
    return []


def fetch(url, dst, tries=3):
    part = dst + ".part"
    for _ in range(tries):
        tok = token()
        if not tok:
            return False
        h = {"Authorization": "Bearer " + tok}
        u = url
        for _ in range(8):
            try:
                r = S.get(u, headers=h, allow_redirects=False,
                          timeout=(60, 600), stream=True)
            except Exception:
                break
            if r.status_code in (301, 302, 303, 307, 308):
                u = r.headers.get("Location", "")
                r.close()
                if not u:
                    break
                continue
            if r.status_code != 200:
                r.close()
                break
            n = 0
            try:
                with open(part, "wb") as f:
                    for ch in r.iter_content(4 * 1024 * 1024):
                        f.write(ch)
                        n += len(ch)
            except Exception:
                r.close()
                break
            r.close()
            if n < MIN_BYTES:
                if os.path.exists(part):
                    os.remove(part)
                break
            with open(part, "rb") as f:
                if not f.read(8).startswith(HDF5_MAGIC):
                    os.remove(part)
                    return False
            os.replace(part, dst)
            return True
        time.sleep(3)
    return False


def find_geo(name):
    parts = name.split(".")
    stamp = parts[1] + "." + parts[2]
    if not os.path.isdir(GEO):
        return None
    cands = [f for f in os.listdir(GEO) if f.startswith("VJ103MOD.%s" % stamp)]
    return os.path.join(GEO, cands[0]) if cands else None


def load_scene(l1b_path, geo_path):
    with h5py.File(l1b_path, "r") as f:
        od = f["observation_data"]
        def band(k):
            ds = od[k]
            raw = np.array(ds[:], dtype="uint16")
            sf = float(np.atleast_1d(ds.attrs.get("scale_factor", 1.0))[0])
            off = float(np.atleast_1d(ds.attrs.get("add_offset", 0.0))[0])
            fv = int(np.atleast_1d(ds.attrs.get("_FillValue", 65535))[0])
            v = raw.astype("float32") * sf + off
            v[raw == fv] = np.nan
            return v
        m03 = band("M03")
        m04 = band("M04")
        m05 = band("M05")
        m07 = band("M07")
        m11 = band("M11")

    with h5py.File(geo_path, "r") as f:
        g = f["geolocation_data"]
        def rg(k):
            ds = g[k]
            s = float(np.atleast_1d(ds.attrs.get("scale_factor", 1.0))[0])
            o = float(np.atleast_1d(ds.attrs.get("add_offset", 0.0))[0])
            return np.array(ds[:], dtype="float32") * s + o
        lat = rg("latitude")
        lon = rg("longitude")
        sza = rg("solar_zenith")
        lwm = np.array(g["land_water_mask"][:], dtype="uint8") \
            if "land_water_mask" in g else None

    h = min(m05.shape[0], lat.shape[0])
    w = min(m05.shape[1], lat.shape[1])
    m03, m04, m05 = m03[:h, :w], m04[:h, :w], m05[:h, :w]
    m07, m11 = m07[:h, :w], m11[:h, :w]
    lat, lon, sza = lat[:h, :w], lon[:h, :w], sza[:h, :w]
    if lwm is not None:
        lwm = lwm[:h, :w]

    valid = (np.isfinite(m05) & np.isfinite(m07) & np.isfinite(m11)
             & np.isfinite(lat) & (sza < 80))
    return m03, m04, m05, m07, m11, valid, lat, lon, sza, lwm


def to_reflectance(v, sza):
    """反射率 = 观测值 / cos(太阳天顶角)，用于 FAI 计算"""
    cosz = np.maximum(np.cos(np.deg2rad(np.clip(sza, 0, 80))), 0.15)
    return v / cosz


def fai_of(r05, r07, r11):
    lam_r, lam_n, lam_s = LAM["M05"], LAM["M07"], LAM["M11"]
    w = (lam_n - lam_r) / (lam_s - lam_r)
    base = r05 + (r11 - r05) * w
    return r07 - base


def water_mask(r04, r05, lwm):
    """水体：用 land_water_mask（VJ103MOD 实际编码）

    实测 flag_values / flag_meanings：
        0 Shallow_Ocean / 1 Land / 2 Coastline / 3 Shallow_Inland
        4 Ephemeral / 5 ? / 6 Deep_Inland / 7 Continental_or_Deep_Ocean

    注意：**1 是陆地**。早期版本写成 isin((1,3,4))，把陆地当成了水，
    在黄海那景上算出“水体 86%”（实际是陆地 85%），反射率 0.37 的异常由此而来。
    """
    if lwm is not None:
        return np.isin(lwm, (0, 3, 4, 6, 7))
    return (r05 < 0.10) & (r04 < 0.12)


def adaptive_threshold(fai, water):
    v = fai[water]
    v = v[np.isfinite(v)]
    if v.size < 5000:
        return None, None
    return float(np.median(v) + 2.0 * np.std(v)), float(np.median(v))


def grade_of(fai_win, water_win, thr):
    if thr is None:
        return None, 0.0
    m = water_win & np.isfinite(fai_win)
    if m.sum() < 2000:
        return None, 0.0
    frac = float(np.mean(fai_win[m] > thr))
    if frac >= G_TIDE:
        return "green_tide", frac
    if frac >= G_PATCHY:
        return "patchy", frac
    return "clean_water", frac


def pick_windows(water, fai, thr, max_n=3):
    """按 FAI 高值定位窗口

    为什么不能按水体中位位置选：绿潮是稀疏斑块，水体中位位置往往在开阔水面，
    窗口会全部落在 clean_water（2023-05-20 实测：全图超阈 6.99%，窗口却是 clean）。

    这里优先找超阈像元最密集的窗口；若全图无超阈像元，再退回水体中位位置，
    用于采集负样本。判级仍按窗口内超阈占比算，不因定位方式而改变。
    """
    H, W = water.shape
    if H < WIN or W < WIN:
        return []
    out = []
    ok = water & np.isfinite(fai)

    hot = ok & (fai > thr) if thr is not None else np.zeros_like(ok)
    if hot.sum() >= 500:
        step = MAXI // 4
        best = []
        for y in range(0, H - WIN + 1, step):
            for x in range(0, W - WIN + 1, step):
                c = int(hot[y:y + WIN, x:x + WIN].sum())
                if c > 0:
                    best.append((c, y, x))
        best.sort(reverse=True)
        picked = []
        for (c, y, x) in best:
            if all(abs(y - yy) >= WIN // 2 or abs(x - xx) >= WIN // 2
                   for (yy, xx) in picked):
                picked.append((y, x))
            if len(picked) >= max_n:
                break
        out = picked

    if len(out) < max_n:
        ys, xs = np.nonzero(ok)
        if ys.size >= WIN * WIN * 0.3:
            cy, cx = int(np.median(ys)), int(np.median(xs))
            for dy, dx in ((0, 0), (0, WIN), (WIN, 0)):
                y0 = int(np.clip(cy + dy - WIN // 2, 0, max(0, H - WIN)))
                x0 = int(np.clip(cx + dx - WIN // 2, 0, max(0, W - WIN)))
                if any(abs(y0 - yy) < WIN // 2 and abs(x0 - xx) < WIN // 2
                       for (yy, xx) in out):
                    continue
                if ok[y0:y0 + WIN, x0:x0 + WIN].mean() > 0.3:
                    out.append((y0, x0))
                if len(out) >= max_n:
                    break
    return out


PROMPT_FAI = ("This is a Floating Algae Index (FAI) field derived from VIIRS over the "
              "Yellow Sea. Positive values mark floating macroalgae (Ulva) slicks on the "
              "sea surface. Describe where the floating algae is and how it is distributed.")
PROMPT_RGB = ("This is a VIIRS true-colour image (M05 red, M04 green, M03 blue, 750 m) "
              "of the Yellow Sea. Describe what you see: whether green floating algae "
              "slicks are present and how they are distributed.")

ANS = {
    "green_tide": "Dense green floating algae slicks cover a large part of the water "
                  "surface, forming elongated bands typical of Ulva prolifera green tides.",
    "patchy": "Scattered patches of floating algae are present on the water surface, "
              "but they do not form continuous slicks.",
    "clean_water": "The water surface is clear with no substantial floating algae; "
                   "only isolated specks may be present.",
}


def render_fai(fai_win, water_win, path, tag, lab, frac, thr):
    show = np.where(water_win, fai_win, np.nan)
    fig = plt.figure(figsize=(4.6, 5.2), dpi=100)
    ax = fig.add_axes([0.04, 0.22, 0.80, 0.70])
    im = ax.imshow(show, cmap="YlGn", vmin=-0.03, vmax=0.08, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  %s" % (tag, lab), fontsize=8)
    cax = fig.add_axes([0.86, 0.22, 0.035, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("FAI", fontsize=7)
    fig.text(0.04, 0.165, "Sensor: VIIRS/JPSS-1 M05 0.64 / M07 0.87 / M11 2.25 um", fontsize=6.2)
    fig.text(0.04, 0.120, "FAI: near-IR minus linear baseline (Hu 2009, JGR)", fontsize=6.2)
    fig.text(0.04, 0.075, "Scene-adaptive threshold %.4f   Fraction above %.1f%%"
             % (thr, frac * 100), fontsize=6.2)
    fig.text(0.04, 0.030, "Grade: %s   (floating algae, not water quality)" % lab,
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def render_rgb(rgb, path, tag):
    v = rgb[np.isfinite(rgb)]
    hi = np.percentile(v, 99) if v.size else 0.3
    hi = max(hi, 0.05)
    fig = plt.figure(figsize=(4.6, 4.9), dpi=100)
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.78])
    ax.imshow(np.clip(np.nan_to_num(rgb) / hi, 0, 1), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  true colour" % tag, fontsize=8)
    fig.text(0.03, 0.095, "Bands: M05 0.64 (R) / M04 0.55 (G) / M03 0.48 um (B)",
             fontsize=6.2)
    fig.text(0.03, 0.055, "Sensor: VIIRS/JPSS-1 VJ102MOD, 750 m", fontsize=6.2)
    fig.text(0.03, 0.018, "Green algae slicks appear as bright green streaks",
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def process_day(day, keep_raw, recs, counter, per_class):
    got_l1b, got_geo = [], []
    for box in BOXES:
        for g in search(day, box, CID_L1B, LIMIT_PER_DAY):
            dst = os.path.join(L1B, g["filename"])
            if os.path.exists(dst) and os.path.getsize(dst) >= MIN_BYTES:
                if dst not in got_l1b:
                    got_l1b.append(dst)
                continue
            if fetch(g["url"], dst):
                got_l1b.append(dst)
        for g in search(day, box, CID_GEO, LIMIT_PER_DAY):
            dst = os.path.join(GEO, g["filename"])
            if os.path.exists(dst) and os.path.getsize(dst) >= MIN_BYTES:
                if dst not in got_geo:
                    got_geo.append(dst)
                continue
            if fetch(g["url"], dst):
                got_geo.append(dst)

    if not got_l1b:
        log("    %s 无覆盖" % day)
        return

    for lp in got_l1b:
        name = os.path.basename(lp)
        stamp = ".".join(name.split(".")[1:3])
        gp = find_geo(name)
        if not gp:
            log("    %s 无地理定位" % stamp)
            continue
        try:
            m03, m04, m05, m07, m11, valid, lat, lon, sza, lwm = load_scene(lp, gp)
        except Exception as e:
            log("    %s 读取失败 %s" % (stamp, str(e)[:60]))
            continue

        r04 = to_reflectance(m04, sza)
        r05 = to_reflectance(m05, sza)
        r07 = to_reflectance(m07, sza)
        r11 = to_reflectance(m11, sza)
        fai = fai_of(r05, r07, r11)
        fai[~valid] = np.nan
        water = water_mask(r04, r05, lwm) & valid

        if water.mean() < 0.005:
            log("    %s 水体太少 %.2f%%" % (stamp, water.mean() * 100))
            continue
        thr, med = adaptive_threshold(fai, water)
        if thr is None:
            log("    %s 水体像元不足" % stamp)
            continue

        wins = pick_windows(water, fai, thr)
        if not wins:
            log("    %s 无合适水体窗口" % stamp)
            continue

        for (y0, x0) in wins:
            sl = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))
            f_w, w_w = fai[sl], water[sl]
            lab, frac = grade_of(f_w, w_w, thr)
            if lab is None:
                continue
            if per_class[lab] >= MAX_PER_CLASS:
                continue
            per_class[lab] += 1
            counter[lab] += 1
            tag = "%s_%d_%d" % (stamp, y0, x0)

            p1 = os.path.join(IMG, "%s_fai.png" % tag)
            render_fai(f_w, w_w, p1, tag, lab, frac, thr)
            recs.append({"image": os.path.basename(p1), "stem": tag, "label": lab,
                         "product": "intrinsic_fai",
                         "frac_above": round(frac, 4), "threshold": round(thr, 5),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_FAI},
                             {"from": "gpt", "value": ANS[lab] +
                              " The fraction of water pixels above the scene-adaptive FAI "
                              "threshold is %.1f percent." % (frac * 100)}]})

            rgb_w = np.dstack([r05[sl], r04[sl], m03[sl]])
            p2 = os.path.join(IMG, "%s_rgb.png" % tag)
            render_rgb(rgb_w, p2, tag)
            recs.append({"image": os.path.basename(p2), "stem": tag, "label": lab,
                         "product": "intrinsic_true_colour",
                         "frac_above": round(frac, 4), "threshold": round(thr, 5),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_RGB},
                             {"from": "gpt", "value": ANS[lab]}]})

        log("    %s  水体%.1f%% 阈值%.4f 窗口%d 累计%d"
            % (stamp, water.mean() * 100, thr, len(wins), len(recs)))

    if not keep_raw:
        for p in got_l1b + got_geo:
            try:
                os.remove(p)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="只处理前 N 天（0=全部）")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    for d in (IMG, L1B, GEO):
        os.makedirs(d, exist_ok=True)

    d0 = dt.date.fromisoformat(START)
    d1 = dt.date.fromisoformat(END)
    alldays = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    days = alldays if not args.days else alldays[:args.days]

    log("=" * 66)
    log("绿潮 FAI 流水线   %s ~ %s   %d 天   保留原始: %s"
        % (START, END, len(days), args.keep))
    log("=" * 66)

    recs = []
    counter = collections.Counter()
    per_class = collections.Counter()

    for i, d in enumerate(days):
        log("[%d/%d] %s" % (i + 1, len(days), d))
        try:
            process_day(d, args.keep, recs, counter, per_class)
        except Exception as e:
            log("   异常 %s" % str(e)[:80])

    if not recs:
        log("无样本")
        return

    random.seed(53)
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["product"], r["label"])].append(r)
    val, train = [], []
    for k, lst in by.items():
        lst = lst[:]
        random.shuffle(lst)
        n = min(VAL_PER_CLASS, max(5, len(lst) // 5))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_green_tide.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_green_tide_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    log("=" * 66)
    log("样本 train %d | val %d" % (len(train), len(val)))
    log("分级分布: %s" % dict(counter))
    for k in sorted(by):
        log("   %-28s %-12s %d" % (k[0], k[1], len(by[k])))
    log("输出: %s" % IMG)


if __name__ == "__main__":
    main()

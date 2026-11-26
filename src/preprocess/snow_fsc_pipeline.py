# -*- coding: utf-8 -*-
r"""
光学积雪覆盖度 · 下载 → 渲染 → 删原始 一条龙（源 1 光学 · 暴雪/雪灾）

为什么做成流水线
---------------
  全量 56 GB，盘不够。改为按天上处理：
      下载某天的轨道 → 算 NDSI/FSC → 渲染 → 删原始 nc → 下一天
  盘上同时最多占 2 GB。中途断了只丢当前一批。

判据（国际业务标准，不改）
------------------------
  NDSI = (ρ0.55 − ρ1.61) / (ρ0.55 + ρ1.61)
  二值雪  NDSI > 0.4                        Hall, Riggs & Salomonson 1995
  FSC     0.5·tanh(2.65·NDSI − 1.42) + 0.5   Salomonson & Appel 2004
  两者为 NASA MODIS(MOD10A1) / VIIRS(VJ110) 积雪产品正式算法。

与国标对应
---------
  GB/T 20482-2017《牧区雪灾等级》三要素：积雪掩埋牧草程度、积雪面积比、持续日数。
  光学只能给"积雪面积比" = 窗口内 FSC 空间平均。**本元素不是雪灾等级。**

双产品
------
  本征场 = FSC 覆盖度场（由 NDSI 换算）
  特色场 = NDSI 伪彩（突出雪/非雪边界）

分级（按窗口 FSC 空间平均）
------------------------
  < 0.02        no_snow      无雪
  0.02 ~ 0.20   light        零星积雪
  0.20 ~ 0.60   moderate     部分覆盖
  >= 0.60       heavy        大范围积雪

用法
----
  python src/preprocess/snow_fsc_pipeline.py --days 1            # 先试一天
  python src/preprocess/snow_fsc_pipeline.py --days 1 --keep      # 保留原始文件
  python src/preprocess/snow_fsc_pipeline.py                     # 全量
"""
import os
import re
import sys
import json
import time
import shutil
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

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\viirs_snow"
L1B = os.path.join(RAW, "l1b")
GEO = os.path.join(RAW, "geo")
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\snow_fsc"
IMG = os.path.join(OUT, "images")

URS = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
CID_L1B = "C2103811746-LAADS"
CID_GEO = "C2105084593-LAADS"
USER = os.environ.get("EARTHDATA_USER", "")
PASS = os.environ.get("EARTHDATA_PASS", "")
HDF5_MAGIC = b"\x89HDF"
MIN_BYTES = 20_000_000

REGIONS = {
    "dongbei": {"name": "Northeast China", "box": (120.0, 135.0, 42.0, 54.0)},
    "neimenggu": {"name": "Inner Mongolia", "box": (105.0, 125.0, 40.0, 50.0)},
    "qingzang": {"name": "East Tibetan Plateau", "box": (95.0, 105.0, 30.0, 38.0)},
}
START = "2023-12-01"
END = "2024-02-29"
LIMIT_PER_DAY = 2

WIN = 1024
OUT_SIZE = 384
VAL_PER_GROUP = 30
NDSI_SNOW = 0.4

G_NOSNOW = 0.02
G_LIGHT = 0.20
G_MODERATE = 0.60

S = requests.Session()
S.trust_env = False
_TOK = {"v": None, "t": 0.0}


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


# ---------------- 下载 ----------------

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


# ---------------- 渲染 ----------------

def find_geo(name):
    parts = name.split(".")
    stamp = parts[1] + "." + parts[2]
    cands = [f for f in os.listdir(GEO) if f.startswith("VJ103MOD.%s" % stamp)] \
        if os.path.isdir(GEO) else []
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
        m04 = band("M04")     # 0.55 μm
        m10 = band("M10")     # 1.61 μm
        # 云掩膜用 M15 亮温
        m15 = band("M15")
        lut15 = np.array(od["M15_brightness_temperature_lut"][:], dtype="float32")
        dn15 = np.array(od["M15"][:], dtype="uint16")
        bt15 = lut15[np.clip(dn15, 0, len(lut15) - 1)].astype("float32")
        bt15[bt15 < 100] = np.nan

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

    h = min(m04.shape[0], lat.shape[0])
    w = min(m04.shape[1], lat.shape[1])
    m04, m10, bt15 = m04[:h, :w], m10[:h, :w], bt15[:h, :w]
    lat, lon, sza = lat[:h, :w], lon[:h, :w], sza[:h, :w]

    valid = np.isfinite(m04) & np.isfinite(m10) & np.isfinite(lat) & (sza < 80)
    # 云掩膜：不能用绝对亮温阈值。
    # 冬季东北地表只有 250~270 K，用 BT>275 K 会把地表全部滤成云（实测 6 条轨道
    # 全部“晴空像元不足”）。
    # 改用相对判断：云顶比该景地表冷，取该景 BT15 的 60 分位作为参照，
    # 高于参照值 - 8 K 的像元视为可能的晴空。
    ref = np.nanpercentile(bt15[np.isfinite(bt15)], 60) if np.isfinite(bt15).any() else 280.0
    clear = valid & (bt15 > ref - 8.0)
    return m04, m10, valid, clear, lat, lon, sza


def ndsi_of(m04, m10):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (m04 - m10) / (m04 + m10)


def fsc_of(ndsi):
    """Salomonson & Appel 2004"""
    return 0.5 * np.tanh(2.65 * ndsi - 1.42) + 0.5


def grade_of(fsc, clear):
    v = fsc[clear]
    v = v[np.isfinite(v)]
    if v.size < 2000:
        return None, 0.0
    mean = float(np.mean(np.clip(v, 0, 1)))
    if mean < G_NOSNOW:
        return "no_snow", mean
    if mean < G_LIGHT:
        return "light", mean
    if mean < G_MODERATE:
        return "moderate", mean
    return "heavy", mean


def pick_window(clear, lat, lon, box):
    lon0, lon1, lat0, lat1 = box
    inb = clear & (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    if inb.sum() < WIN * WIN * 0.05:
        return None
    ys, xs = np.nonzero(inb)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    H, W = clear.shape
    y0 = int(np.clip(cy - WIN // 2, 0, max(0, H - WIN)))
    x0 = int(np.clip(cx - WIN // 2, 0, max(0, W - WIN)))
    return y0, x0


PROMPT_FSC = ("This is a fractional snow cover field derived from VIIRS (NDSI method, "
              "750 m) over northern China in winter. Values run from 0 (no snow) to 1 "
              "(complete snow cover). Describe the snow cover pattern.")
PROMPT_NDSI = ("This is a VIIRS NDSI (normalized difference snow index) image of a "
               "winter landscape. Bright values indicate snow. Describe the snow cover "
               "distribution and how it relates to terrain.")

ANS = {
    "no_snow": "The scene shows almost no snow cover; the mean fractional snow cover in "
               "this window is below 2 percent.",
    "light": "Sparse snow patches are present; the mean fractional snow cover is in the "
             "low range, with snow mostly on higher or shaded ground.",
    "moderate": "Snow covers a substantial part of the window but not continuously; the "
                "mean fractional snow cover is moderate.",
    "heavy": "Snow cover is extensive and nearly continuous across the window, with a high "
             "mean fractional snow cover.",
}
ANS_CN = {
    "no_snow": "窗口内几乎无积雪",
    "light": "零星积雪",
    "moderate": "部分覆盖",
    "heavy": "大范围积雪",
}


def render_fsc(fsc_win, path, tag, lab, mean, rg_name):
    fig = plt.figure(figsize=(4.6, 5.2), dpi=100)
    ax = fig.add_axes([0.04, 0.22, 0.80, 0.70])
    im = ax.imshow(fsc_win, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  %s" % (tag, lab), fontsize=8)
    cax = fig.add_axes([0.86, 0.22, 0.035, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("fractional snow cover", fontsize=7)
    fig.text(0.04, 0.165, "Sensor: VIIRS/JPSS-1 M04 0.55um / M10 1.61um, 750 m", fontsize=6.2)
    fig.text(0.04, 0.120, "NDSI>0.4 (Hall 1995); FSC=0.5tanh(2.65NDSI-1.42)+0.5", fontsize=6.2)
    fig.text(0.04, 0.075, "Mean FSC in window: %.3f   Region: %s" % (mean, rg_name), fontsize=6.2)
    fig.text(0.04, 0.030, "Snow cover index only, not a snow-disaster grade (GB/T 20482)",
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def render_ndsi(ndsi_win, path, tag):
    fig = plt.figure(figsize=(4.6, 4.9), dpi=100)
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.78])
    ax.imshow(ndsi_win, cmap="RdYlBu_r", vmin=-0.2, vmax=0.9, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  NDSI" % tag, fontsize=8)
    fig.text(0.03, 0.095, "NDSI = (0.55um - 1.61um) / (0.55um + 1.61um)", fontsize=6.2)
    fig.text(0.03, 0.055, "Sensor: VIIRS/JPSS-1 VJ102MOD, 750 m", fontsize=6.2)
    fig.text(0.03, 0.018, "Snow threshold NDSI > 0.4 (Hall, Riggs & Salomonson 1995)",
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


# ---------------- 主流程 ----------------

def process_region(day, rk, rv, keep_raw, recs, counter):
    box = rv["box"]
    l1b_list = search(day, box, CID_L1B, LIMIT_PER_DAY)
    if not l1b_list:
        return
    geo_list = search(day, box, CID_GEO, LIMIT_PER_DAY)
    got_l1b, got_geo = [], []

    for g in l1b_list:
        dst = os.path.join(L1B, g["filename"])
        if os.path.exists(dst) and os.path.getsize(dst) >= MIN_BYTES:
            got_l1b.append(dst)
            continue
        if fetch(g["url"], dst):
            got_l1b.append(dst)
    for g in geo_list:
        dst = os.path.join(GEO, g["filename"])
        if os.path.exists(dst) and os.path.getsize(dst) >= MIN_BYTES:
            got_geo.append(dst)
            continue
        if fetch(g["url"], dst):
            got_geo.append(dst)

    for lp in got_l1b:
        name = os.path.basename(lp)
        stamp = ".".join(name.split(".")[1:3])
        gp = find_geo(name)
        if not gp:
            continue
        try:
            m04, m10, valid, clear, lat, lon, sza = load_scene(lp, gp)
        except Exception as e:
            log("    %s 读取失败 %s" % (stamp, str(e)[:60]))
            continue

        w = pick_window(clear, lat, lon, box)
        if not w:
            log("    %s %s 有效晴空像元不足" % (stamp, rv["name"]))
            continue
        y0, x0 = w
        sl = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))

        ndsi = ndsi_of(m04[sl], m10[sl])
        fsc = fsc_of(ndsi)
        c_w = clear[sl]
        lab, mean = grade_of(fsc, c_w)
        if lab is None:
            log("    %s %s 有效像元过少" % (stamp, rv["name"]))
            continue

        counter[lab] += 1
        tag = "%s_%s_%d_%d" % (stamp, rk, y0, x0)

        fsc_show = np.where(c_w, np.clip(fsc, 0, 1), np.nan)
        p1 = os.path.join(IMG, "%s_fsc.png" % tag)
        render_fsc(fsc_show, p1, tag, lab, mean, rv["name"])
        recs.append({
            "image": os.path.basename(p1), "stem": tag, "label": lab,
            "product": "intrinsic_fractional_snow_cover",
            "region": rk, "mean_fsc": round(mean, 4),
            "conversations": [
                {"from": "human", "value": "<image>" + PROMPT_FSC},
                {"from": "gpt", "value": ANS[lab] + " Mean FSC is %.3f." % mean}]})

        ndsi_show = np.where(c_w, ndsi, np.nan)
        p2 = os.path.join(IMG, "%s_ndsi.png" % tag)
        render_ndsi(ndsi_show, p2, tag)
        recs.append({
            "image": os.path.basename(p2), "stem": tag, "label": lab,
            "product": "intrinsic_ndsi",
            "region": rk, "mean_fsc": round(mean, 4),
            "conversations": [
                {"from": "human", "value": "<image>" + PROMPT_NDSI},
                {"from": "gpt", "value": ANS[lab]}]})

        log("    %s %-8s 窗口(%d,%d) 晴空%.0f%% 平均FSC %.3f → %s"
            % (stamp, rv["name"], y0, x0, c_w.mean() * 100, mean, lab))

    if not keep_raw:
        for p in got_l1b + got_geo:
            try:
                os.remove(p)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="只处理前 N 天（0=全部）")
    ap.add_argument("--keep", action="store_true", help="保留原始 nc")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    os.makedirs(L1B, exist_ok=True)
    os.makedirs(GEO, exist_ok=True)

    d0 = dt.date.fromisoformat(START)
    d1 = dt.date.fromisoformat(END)
    alldays = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    days = [d for d in alldays if d.weekday() in (1, 3, 5)]
    if args.days:
        days = days[:args.days]

    log("=" * 66)
    log("积雪覆盖度流水线   %d 天 × %d 区域   保留原始: %s"
        % (len(days), len(REGIONS), args.keep))
    log("=" * 66)

    recs = []
    counter = collections.Counter()
    for i, d in enumerate(days):
        log("[%d/%d] %s" % (i + 1, len(days), d))
        for rk, rv in REGIONS.items():
            try:
                process_region(d, rk, rv, args.keep, recs, counter)
            except Exception as e:
                log("   %s 异常 %s" % (rv["name"], str(e)[:80]))

    if not recs:
        log("无样本，检查下载是否成功")
        return

    random.seed(53)
    by = collections.defaultdict(list)
    for r in recs:
        by[(r["product"], r["label"])].append(r)
    val, train = [], []
    for k, lst in by.items():
        lst = lst[:]
        random.shuffle(lst)
        n = min(VAL_PER_GROUP, max(5, len(lst) // 5))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(os.path.join(OUT, "train_snow_fsc.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_snow_fsc_val.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    log("=" * 66)
    log("样本 train %d | val %d" % (len(train), len(val)))
    log("分级分布: %s" % dict(counter))
    for k in sorted(by):
        log("   %-36s %-10s %d" % (k[0], k[1], len(by[k])))
    log("输出: %s" % IMG)


if __name__ == "__main__":
    main()

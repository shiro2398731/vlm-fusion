# -*- coding: utf-8 -*-
r"""
统一入口：原始数据 → 渲染 → 判读（按元素）

当前已接：
  wildfire         FIRMS 每日火点 CSV（点 → 5° 窗口栅格 → 出图 → 判读）
  volcano_thermal  MODVOLC 火山热异常 txt（点 → 时间窗 → 出图 → 判读）

用法：
  python scripts/raw_workflow.py --elements wildfire,volcano_thermal \
      --firms-dir data/raw/03_thermal_infrared/firms/chn2025 \
      --modvolc-dir data/raw/03_thermal_infrared/modvolc \
      --outdir outputs/raw_pair1 --api http://127.0.0.1:8078
"""
import argparse
import collections
import csv
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_SIZE = 384


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def predict(api, element, png):
    with open(png, "rb") as f:
        data = f.read()
    r = requests.post(api + "/api/predict", data={"element": element},
                      files={"file": (os.path.basename(png), data, "image/png")}, timeout=300)
    d = r.json()
    return d.get("prediction") or {}, d.get("error")


# ---------------- wildfire ----------------
def run_wildfire(mod, firms_dir, outdir, api, grid=5.0, max_days=6):
    import csv as _csv
    files = sorted(glob.glob(os.path.join(firms_dir, "fire_*.csv")))

    def load_pts(fp):
        out = []
        with open(fp, encoding="utf-8", errors="replace") as f:
            for row in _csv.DictReader(f):
                try:
                    if int(row.get("type", "0") or 0) == 2:
                        continue
                    if (row.get("confidence") or "").strip().lower() == "l":
                        continue
                    out.append((float(row["latitude"]), float(row["longitude"]),
                                float(row["bright_ti4"]), float(row["frp"] or 0)))
                except Exception:
                    continue
        return out

    days = []
    for f in files[-max_days * 3:]:
        pts = load_pts(f)
        if pts:
            days.append((os.path.basename(f)[5:15], pts))
        if len(days) >= max_days:
            break

    rows = []
    for stamp, pts in days:
        cells = collections.Counter()
        for lat, lon, t, frp in pts:
            if 18 <= lat < 54 and 73 <= lon < 135:
                cells[(int((lat - 18) // grid), int((lon - 73) // grid))] += 1
        if not cells:
            continue
        # 取火点最多的格子 + 一个空白格
        (iy, ix), n = cells.most_common(1)[0]
        picks = [(iy, ix, True)]
        empty = [(a, b) for a in range(0, 7) for b in range(0, 12) if (a, b) not in cells]
        if empty:
            picks.append((empty[0][0], empty[0][1], False))
        for (a, b, has) in picks:
            la, lo = 18 + a * grid, 73 + b * grid
            field, frp, cnt = mod.pts_in(pts, la, lo, grid)
            label = "active_fire" if cnt.sum() > 0 else "no_active_fire"
            sev = mod.band_of(int(cnt.sum()))
            png = os.path.join(outdir, "wf_%s_%d_%d_frp.png" % (stamp, a, b))
            mod.draw(frp, cnt, la, lo, grid, png,
                     "%s  %s" % (stamp, label), "frp")
            pred, err = predict(api, "wildfire", png)
            rows.append({
                "element": "wildfire", "image": os.path.basename(png),
                "truth_label": label, "truth_severity": sev,
                "truth_count": int(cnt.sum()),
                "model_detected": pred.get("detected"),
                "model_severity": pred.get("severity"),
                "model_value": pred.get("value"),
                "error": err, "match": (pred.get("detected") == (label == "active_fire")),
            })
            print("  wildfire %-28s 真值 %-14s 模型 %s/%s %s" % (
                rows[-1]["image"], label, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
    return rows




# ---------------- snow_swe（AMSR2） ----------------
def run_snow_swe(mod, swe_dir, outdir, api, n_days=2, per_day=2):
    import glob as _g
    import re as _re
    files = sorted(_g.glob(os.path.join(swe_dir, "*.he5")))
    if not files:
        print("  no swe files")
        return []
    train_stamps = set()
    tp = os.path.join(ROOT, "LLaMA-Factory", "data", "train_snow_swe_json.json")
    if os.path.exists(tp):
        for r in json.load(open(tp, encoding="utf-8")):
            if r.get("stamp"):
                train_stamps.add(str(r["stamp"]))
    rows = []
    picked_days = 0
    for fp in reversed(files):          # 新日期优先
        m = _re.search(r"(\d{8})", os.path.basename(fp))
        stamp = m.group(1) if m else "unknown"
        if stamp in train_stamps:
            continue
        swe, lat, lon = mod.load_swe(fp)
        h, w = swe.shape
        cand = []
        for r0 in range(0, h - mod.WIN + 1, mod.STRIDE):
            for c0 in range(0, w - mod.WIN + 1, mod.STRIDE):
                ws = swe[r0:r0 + mod.WIN, c0:c0 + mod.WIN]
                if np.isfinite(ws).mean() < mod.MIN_VALID:
                    continue
                la = float(np.nanmean(lat[r0:r0 + mod.WIN, c0:c0 + mod.WIN]))
                lo = float(np.nanmean(lon[r0:r0 + mod.WIN, c0:c0 + mod.WIN]))
                if not (mod.LATB[0] < la < mod.LATB[1] and mod.LONB[0] < lo < mod.LONB[1]):
                    continue
                ms = float(np.nanmean(ws))
                cand.append((ms, r0, c0, la, lo))
        if not cand:
            continue
        cand.sort(reverse=True)
        picks = cand[:per_day] + [x for x in cand if x[0] < 3][:1]
        for ms, r0, c0, la, lo in picks:
            png = os.path.join(outdir, "swe_%s_r%d_c%d.png" % (stamp, r0, c0))
            mod.render(swe[r0:r0 + mod.WIN, c0:c0 + mod.WIN], png, stamp, la, lo)
            sev = mod.severity_of(ms)
            pred, err = predict(api, "snow_swe", png)
            rows.append({
                "element": "snow_swe", "image": os.path.basename(png), "stamp": stamp,
                "truth_mean_swe": round(ms, 1), "truth_label": sev,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "model_mean_swe": pred.get("mean_swe_mm"), "error": err,
                "match": (pred.get("severity") == sev),
            })
            print("  snow %-30s 真值 %-9s(%5.1f) 模型 %s/%s %s" % (
                rows[-1]["image"], sev, ms, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
        picked_days += 1
        if picked_days >= n_days:
            break
    return rows



# ---------------- waterlogging_pct（SMAP 百分位） ----------------
def run_waterlogging(mod, smap_dir, outdir, api, n_dates=2, per_date=2):
    import glob as _g
    import re as _re
    files = sorted(_g.glob(os.path.join(smap_dir, "SMAP_L3_SM_P_*.h5")))
    if not files:
        print("  no smap files")
        return []
    sm_list, lat, lon, stamps = [], None, None, []
    for fp in files:
        sm, la, lo = mod.load_sm(fp)
        sm_list.append(sm)
        if lat is None:
            lat, lon = la, lo
        m = _re.search(r"SMAP_L3_SM_P_(\d{8})", os.path.basename(fp))
        stamps.append(m.group(1) if m else "unknown")
    stack = np.stack(sm_list)
    pct = np.full_like(stack, np.nan)
    D = stack.shape[0]
    for i in range(D):
        v = stack[i]
        lower = np.nansum(stack < v, axis=0)
        valid = np.sum(np.isfinite(stack), axis=0)
        pct[i] = np.where(valid >= 5, 100.0 * lower / np.maximum(valid, 1), np.nan)

    rows = []
    for i in range(D - 1, max(-1, D - 1 - n_dates), -1):
        stamp, p = stamps[i], pct[i]
        h, w = p.shape
        cand = []
        for r0 in range(0, h - mod.WIN + 1, mod.STRIDE):
            for c0 in range(0, w - mod.WIN + 1, mod.STRIDE):
                wp = p[r0:r0 + mod.WIN, c0:c0 + mod.WIN]
                if np.isfinite(wp).mean() < mod.MIN_VALID:
                    continue
                la = float(np.nanmean(lat[r0:r0 + mod.WIN, c0:c0 + mod.WIN]))
                lo = float(np.nanmean(lon[r0:r0 + mod.WIN, c0:c0 + mod.WIN]))
                if not (mod.LATB[0] < la < mod.LATB[1] and mod.LONB[0] < lo < mod.LONB[1]):
                    continue
                cand.append((float(np.nanmax(wp)), r0, c0, la, lo))
        if not cand:
            continue
        cand.sort(reverse=True)
        picks = cand[:per_date] + [x for x in cand if x[0] < 50][:1]
        for mp, r0, c0, la, lo in picks:
            png = os.path.join(outdir, "wlp_%s_r%d_c%d.png" % (stamp, r0, c0))
            mod.render(p[r0:r0 + mod.WIN, c0:c0 + mod.WIN], png, stamp, la, lo)
            sev = mod.severity_of(mp)
            pred, err = predict(api, "waterlogging_pct", png)
            rows.append({
                "element": "waterlogging_pct", "image": os.path.basename(png), "stamp": stamp,
                "truth_pct": round(mp, 1), "truth_label": sev,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "model_pct": pred.get("mean_sm_percentile"), "error": err,
                "match": (pred.get("severity") == sev),
            })
            print("  wlp %-30s 真值 %-9s(%5.1f) 模型 %s/%s %s" % (
                rows[-1]["image"], sev, mp, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- GRACE（drought_grace / groundwater_grace） ----------------
def run_grace(mod, grace_dir, outdir, api, element, n_dates=2, per_date=2):
    import glob as _g
    import re as _re
    files = sorted(_g.glob(os.path.join(grace_dir, "*.nc4")))
    if not files:
        print("  no grace files")
        return []
    stack, lat0, lon0, stamps = [], None, None, []
    for fp in files:
        g, lat, lon = mod.load_gws(fp)
        stack.append(g)
        if lat0 is None:
            lat0, lon0 = lat, lon
        m = _re.search(r"\.A(\d{8})\.", os.path.basename(fp))
        stamps.append(m.group(1) if m else "unknown")
    stack = np.stack(stack)
    pct = np.full_like(stack, np.nan)
    D = stack.shape[0]
    for i in range(D):
        v = stack[i]
        lower = np.nansum(stack < v, axis=0)
        valid = np.sum(np.isfinite(stack), axis=0)
        pct[i] = np.where(valid >= 10, 100.0 * lower / np.maximum(valid, 1), np.nan)

    rows = []
    for i in range(D - 1, max(-1, D - 1 - n_dates), -1):
        stamp, p = stamps[i], pct[i]
        h, w = p.shape
        pos, neg = [], []
        for r0 in range(0, h - mod.WIN + 1, mod.STRIDE):
            for c0 in range(0, w - mod.WIN + 1, mod.STRIDE):
                wp = p[r0:r0 + mod.WIN, c0:c0 + mod.WIN]
                if np.isfinite(wp).mean() < mod.MIN_VALID:
                    continue
                la = float(np.nanmean(lat0[r0:r0 + mod.WIN]))
                lo = float(np.nanmean(lon0[c0:c0 + mod.WIN]))
                if not (mod.LATB[0] < la < mod.LATB[1] and mod.LONB[0] < lo < mod.LONB[1]):
                    continue
                mp = float(np.nanmean(wp))
                if mp <= 10:
                    pos.append((mp, r0, c0, la, lo))
                elif mp >= 50:
                    neg.append((mp, r0, c0, la, lo))
        pos.sort()
        neg.sort(reverse=True)
        picks = [(True, x) for x in pos[:per_date]] + [(False, x) for x in neg[:1]]
        for det, (mp, r0, c0, la, lo) in picks:
            png = os.path.join(outdir, "grace_%s_%s_r%d_c%d.png" % (element, stamp, r0, c0))
            mod.render(p[r0:r0 + mod.WIN, c0:c0 + mod.WIN], png, stamp, la, lo)
            sev = "drought" if (element == "drought_grace" and det) else (
                "depleted" if det else "none")
            pred, err = predict(api, element, png)
            rows.append({
                "element": element, "image": os.path.basename(png), "stamp": stamp,
                "truth_pct": round(mp, 1), "truth_label": sev,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "error": err, "match": (pred.get("detected") == det),
            })
            print("  %-16s %-38s 真值 %-8s(%5.1f) 模型 %s/%s %s" % (
                element, rows[-1]["image"], sev, mp, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- wind_gale（CMEMS/散射计风场） ----------------
def run_wind_gale(mod, wind_dir, outdir, api, grid=30, max_files=2):
    import glob as _g
    import re as _re
    files = sorted(_g.glob(os.path.join(wind_dir, "*.nc")))
    if not files:
        print("  no wind files")
        return []
    train_stamps = set()
    tp = os.path.join(ROOT, "LLaMA-Factory", "data", "train_wind_gale_json.json")
    if os.path.exists(tp):
        for r in json.load(open(tp, encoding="utf-8")):
            if r.get("stamp"):
                train_stamps.add(str(r["stamp"]))
    rows = []
    used = 0
    for fp in reversed(files):
        m = _re.search(r"(\d{10})", os.path.basename(fp))
        stamp = m.group(1) if m else "unknown"
        if stamp in train_stamps:
            continue
        u, v, lat, lon = mod.load(fp)
        spd = np.sqrt(u ** 2 + v ** 2)
        cells = mod.window_cells(spd, lat, lon, grid)
        if not cells:
            continue
        cells.sort(reverse=True)
        picks = [cells[0], cells[-1]]
        for mx, la, lo in picks:
            grade = mod.grade_of(mx)
            png = os.path.join(outdir, "wind_%s_%d_%d.png" % (stamp, int(la), int(lo)))
            mod.render_cell(u, v, lat, lon, la, lo, grid, png, stamp, grade, True)
            pred, err = predict(api, "wind_gale", png)
            rows.append({
                "element": "wind_gale", "image": os.path.basename(png), "stamp": stamp,
                "truth_max_ms": round(mx, 2), "truth_label": grade,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "model_max_ms": pred.get("max_wind_ms"), "error": err,
                "match": (pred.get("severity") == grade),
            })
            print("  wind %-32s 真值 %-7s(%5.1f) 模型 %s/%s %s" % (
                rows[-1]["image"], grade, mx, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
        used += 1
        if used >= max_files:
            break
    return rows



# ---------------- hurricane_rain（GPM IMERG 日累积） ----------------
def run_hurricane_rain(mod, gpm_dir, outdir, api, n_days=2):
    import glob as _g
    import re as _re
    files = sorted(_g.glob(os.path.join(gpm_dir, "*.HDF5")))
    dates = sorted({_re.search(r"3IMERG\.(\d{8})-", f).group(1) for f in files
                    if _re.search(r"3IMERG\.(\d{8})-", f)})
    if not dates:
        print("  no gpm dates")
        return []
    train_dates = set()
    tp = os.path.join(ROOT, "LLaMA-Factory", "data", "train_hurricane_rain_json.json")
    if os.path.exists(tp):
        for r in json.load(open(tp, encoding="utf-8")):
            img = (r.get("image") or (r.get("images") or [""])[0])
            m = _re.search(r"(\d{8})", str(img))
            if m:
                train_dates.add(m.group(1))
    rows = []
    used = 0
    unseen = [d for d in reversed(dates) if d not in train_dates]
    order = unseen if unseen else list(reversed(dates))
    for date in order:
        total, lat, lon = mod.load_day(date)
        if total is None:
            continue
        H, W = total.shape
        win = mod.WIN
        cand = []
        for ly in range(0, H - win, mod.STRIDE):
            for lx in range(0, W - win, mod.STRIDE):
                sub = total[ly:ly + win, lx:lx + win]
                if not np.isfinite(sub).any():
                    continue
                cand.append((float(np.nanmax(sub)), ly, lx))
        if not cand:
            continue
        cand.sort(reverse=True)
        picks = [cand[0]] + [c for c in cand if c[0] < 1.0][:1]
        for mx, ly, lx in picks:
            sub = total[ly:ly + win, lx:lx + win]
            grade = mod.grade_of(mx)
            png = os.path.join(outdir, "hur_%s_%d_%d.png" % (date, ly, lx))
            mod.render(sub, png, date, mx, "precip")
            pred, err = predict(api, "hurricane_rain", png)
            rows.append({
                "element": "hurricane_rain", "image": os.path.basename(png), "date": date,
                "truth_max_mm": round(mx, 1), "truth_label": grade,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "model_max_mm": pred.get("max_mm_per_day") or pred.get("value"),
                "error": err, "match": (pred.get("severity") == grade),
            })
            print("  hur %-28s %s 真值 %-16s(%6.1f) 模型 %s/%s %s" % (
                rows[-1]["image"], date, grade, mx, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
        used += 1
        if used >= n_days:
            break
    return rows



# ---------------- bloom_chla（ERDDAP 叶绿素日产品） ----------------
def run_bloom(mod, chla_dir, outdir, api, n_days=2):
    import glob as _g
    import re as _re
    import datetime as _dt
    files = sorted(_g.glob(os.path.join(chla_dir, "chla_*.nc")))
    if not files:
        print("  no chla files")
        return []
    train_stamps = set()
    tp = os.path.join(ROOT, "LLaMA-Factory", "data", "train_bloom_chla_json.json")
    if os.path.exists(tp):
        for r in json.load(open(tp, encoding="utf-8")):
            if r.get("stamp"):
                train_stamps.add(str(r["stamp"]))
    rows = []
    done = 0
    for f in reversed(files):
        try:
            ds = mod.Dataset(f)
        except Exception:
            continue
        chl = np.ma.filled(np.array(ds.variables["chlor_a"][:], dtype="float32"), -999.0)
        lat = np.array(ds.variables["latitude"][:], dtype="float64")
        lon = np.array(ds.variables["longitude"][:], dtype="float64")
        times = ds.variables["time"][:]
        for i in range(chl.shape[0]):
            stamp = str(_dt.datetime.utcfromtimestamp(float(times[i])))[:10].replace("-", "")
            if stamp in train_stamps:
                continue
            arr = chl[i, 0]
            arr = np.where((arr < 0) | (arr > 200), np.nan, arr)
            h, w = arr.shape
            cand = []
            for r0 in range(0, h - mod.WIN + 1, mod.STRIDE):
                for c0 in range(0, w - mod.WIN + 1, mod.STRIDE):
                    ww = arr[r0:r0 + mod.WIN, c0:c0 + mod.WIN]
                    v = ww[np.isfinite(ww)]
                    if v.size < ww.size * mod.MIN_VALID:
                        continue
                    cand.append((mod.p90(v), r0, c0))
            if not cand:
                continue
            cand.sort(reverse=True)
            picks = [cand[0]] + [c for c in cand if c[0] < 1.0][:1]
            for pv, r0, c0 in picks:
                win = arr[r0:r0 + mod.WIN, c0:c0 + mod.WIN]
                la = float(lat[min(r0 + mod.WIN // 2, len(lat) - 1)])
                lo = float(lon[min(c0 + mod.WIN // 2, len(lon) - 1)])
                png = os.path.join(outdir, "chl_%s_r%d_c%d.png" % (stamp, r0, c0))
                mod.render(win, png, stamp, la, lo)
                sev = mod.severity_of(pv)
                pred, err = predict(api, "bloom_chla", png)
                rows.append({
                    "element": "bloom_chla", "image": os.path.basename(png), "stamp": stamp,
                    "truth_p90": round(pv, 2), "truth_label": sev,
                    "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                    "model_p90": pred.get("p90_chla"), "error": err,
                    "match": (pred.get("severity") == sev),
                })
                print("  chl %-28s 真值 %-8s(%6.2f) 模型 %s/%s %s" % (
                    rows[-1]["image"], sev, pv, pred.get("detected"), pred.get("severity"),
                    "OK" if rows[-1]["match"] else "MISS"))
            done += 1
            if done >= n_days:
                break
        ds.close()
        if done >= n_days:
            break
    return rows



# ---------------- sea_ice_amsr2（NSIDC CDR 海冰浓度） ----------------
def run_sea_ice(mod, ice_dir, outdir, api, n_dates=2):
    import netCDF4 as nc
    import datetime as _dt
    conc_f = os.path.join(ice_dir, "nh_ice_conc_2020-12_2021-03.nc")
    grid_f = os.path.join(ice_dir, "nh_grid_latlon.nc")
    if not (os.path.exists(conc_f) and os.path.exists(grid_f)):
        print("  no sea ice files")
        return []
    ds = nc.Dataset(conc_f)
    conc = ds.variables["cdr_seaice_conc"][:]          # (T, 448, 304), 0-1
    tsec = ds.variables["time"][:]
    ds.close()
    g = nc.Dataset(grid_f)
    lat = g.variables["latitude"][:]
    lon = g.variables["longitude"][:]
    g.close()
    conc = np.ma.filled(conc, np.nan).astype("float64")
    if np.nanmax(conc) <= 1.5:
        conc = conc * 100.0
    conc[(conc < 0) | (conc > 100)] = np.nan

    rows = []
    T = conc.shape[0]
    for ti in range(T - 1, max(-1, T - 1 - n_dates), -1):
        stamp = _dt.datetime.utcfromtimestamp(float(tsec[ti])).strftime("%Y%m%d")
        ice = conc[ti]
        h, w = ice.shape
        cand = []
        for r0 in range(0, h - 128 + 1, 128):
            for c0 in range(0, w - 128 + 1, 128):
                wi = ice[r0:r0 + 128, c0:c0 + 128]
                if np.isfinite(wi).mean() < 0.5:
                    continue
                frac = float(np.nanmean(wi >= 15.0)) * 100.0
                cand.append((frac, r0, c0, float(np.nanmean(wi))))
        if not cand:
            continue
        cand.sort(reverse=True)
        picks = [cand[0]] + [c for c in cand if c[0] < 15][:1]
        for frac, r0, c0, mc in picks:
            win = ice[r0:r0 + 128, c0:c0 + 128]
            latc = float(np.nanmean(lat[r0:r0 + 128, c0:c0 + 128]))
            lonc = float(np.nanmean(lon[r0:r0 + 128, c0:c0 + 128]))
            png = os.path.join(outdir, "ice_%s_r%d_c%d.png" % (stamp, r0, c0))
            mod.render(win, png, stamp, r0, c0, latc, lonc)
            sev = mod.severity_of(frac)
            pred, err = predict(api, "sea_ice_amsr2", png)
            rows.append({
                "element": "sea_ice_amsr2", "image": os.path.basename(png), "stamp": stamp,
                "truth_ice_frac": round(frac, 1), "truth_label": sev,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "model_ice_frac": pred.get("ice_frac_pct"), "error": err,
                "match": (pred.get("severity") == sev),
            })
            print("  ice %-30s 真值 %-8s(%5.1f%%) 模型 %s/%s %s" % (
                rows[-1]["image"], sev, frac, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- gpm_nt（GPM 日累积，区域切窗） ----------------
def run_gpm_nt(mod, gpm_dir, outdir, api, n_days=2):
    import glob as _g
    import re as _re
    files = sorted(_g.glob(os.path.join(gpm_dir, "*.HDF5")))
    dates = sorted({_re.search(r"3IMERG\.(\d{8})-", f).group(1) for f in files
                    if _re.search(r"3IMERG\.(\d{8})-", f)})
    if not dates:
        print("  no gpm dates")
        return []
    train_dates = set()
    tp = os.path.join(ROOT, "LLaMA-Factory", "data", "train_gpm_nt_json.json")
    if os.path.exists(tp):
        for r in json.load(open(tp, encoding="utf-8")):
            m = _re.search(r"(\d{8})", str(r.get("image", "")))
            if m:
                train_dates.add(m.group(1))
    unseen = [d for d in reversed(dates) if d not in train_dates] or list(reversed(dates))
    rows = []
    for date in unseen[:n_days]:
        total, lat, lon = mod.load_day(date)
        if total is None:
            continue
        H, W = total.shape
        half = mod.HALF
        stride = mod.STRIDE_DEG
        cands, dry = [], []
        for ly in range(0, H - 2 * half, stride):
            for lx in range(0, W - 2 * half, stride):
                sub = total[ly:ly + 2 * half, lx:lx + 2 * half]
                mm = float(sub.mean())
                (cands if mm >= mod.RAIN_MEAN_MIN else dry).append((mm, ly, lx, sub))
        if not cands:
            continue
        cands.sort(reverse=True)
        picks = [cands[0]]
        if dry:
            dry.sort()
            picks.append(dry[0])
        for mm, ly, lx, sub in picks:
            grade = mod.grade_of(mm)
            png = os.path.join(outdir, "gpmnt_%s_%d_%d.png" % (date, ly, lx))
            mod.render(sub, png, date, mm)
            pred, err = predict(api, "gpm_nt", png)
            rows.append({
                "element": "gpm_nt", "image": os.path.basename(png), "date": date,
                "truth_mean_mm": round(mm, 2), "truth_label": grade,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "model_mean_mm": pred.get("mean_mm"), "error": err,
                "match": (pred.get("severity") == grade),
            })
            print("  gpmnt %-28s %s 真值 %-14s(%6.2f) 模型 %s/%s %s" % (
                rows[-1]["image"], date, grade, mm, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- wave（Sentinel-3 SRAL 沿轨有效波高） ----------------
def run_wave(mod, alt_dir, outdir, api, n_files=3):
    import glob as _g
    zips = sorted(_g.glob(os.path.join(alt_dir, "*.zip")))[-n_files:]
    rows = []
    for zp in zips:
        try:
            track = mod.load_track(zp)
        except Exception as e:
            print("  load fail", os.path.basename(zp)[:40], str(e)[:60])
            continue
        if track is None:
            continue
        lat, lon, swh, _t = track
        if lat is None or len(lat) < 200:
            continue
        dist = mod.path_distance(lat, lon)
        meta = {"short": os.path.basename(zp)[20:34]}
        m = np.isfinite(swh)
        if m.sum() < 50:
            continue
        mx = float(np.nanmax(swh[m]))
        med = float(np.nanmedian(swh[m]))
        band = mod.band_of(mx)
        name = os.path.basename(zp)[20:34]
        png = os.path.join(outdir, "%s_hs.png" % name)
        mod.render(dist, swh, lat, lon, meta, png, "hs")
        pred, err = predict(api, "wave", png)
        rows.append({
            "element": "wave", "image": os.path.basename(png), "file": os.path.basename(zp),
            "truth_max_hs": round(mx, 2), "truth_median_hs": round(med, 2), "truth_label": band,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "model_hs_max": pred.get("hs_max_m"), "error": err,
            "match": (pred.get("severity") == band),
        })
        print("  wave %-28s 真值 %-7s(%5.2f m) 模型 %s/%s %s" % (
            rows[-1]["image"], band, mx, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- flood_precip_global（GPM 3 日累积 + GDACS 事件） ----------------
def run_flood_precip(mod, gpm_dir, gdacs_path, outdir, api, n_events=2):
    import glob as _g
    import re as _re
    import datetime as _dt
    import h5py
    files = sorted(_g.glob(os.path.join(gpm_dir, "*.HDF5")))
    dates = sorted({_re.search(r"3IMERG\.(\d{8})-", f).group(1) for f in files
                    if _re.search(r"3IMERG\.(\d{8})-", f)})
    dset = set(dates)
    if not os.path.exists(gdacs_path):
        print("  no gdacs")
        return []

    def day_arr(date):
        fs = sorted(_g.glob(os.path.join(gpm_dir, "*3IMERG.%s-*.HDF5" % date)))
        if not fs:
            return None
        total = None
        lat = lon = None
        for f in fs:
            try:
                with h5py.File(f, "r") as h:
                    if lat is None:
                        lat = h["Grid"]["lat"][:]
                        lon = h["Grid"]["lon"][:]
                    a = h["Grid"]["precipitation"][0].astype("float64")
            except Exception:
                continue
            a[a < -1000] = 0.0
            a = a * 0.5
            total = a if total is None else total + a
        if total is None:
            return None
        return total.T, lat, lon          # (lat, lon)

    gj = json.load(open(gdacs_path, encoding="utf-8"))
    events = []
    for f in gj.get("features", []):
        p_ = f.get("properties", {})
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        lon_, lat_ = g["coordinates"][:2]
        dt = (p_.get("fromdate") or "")[:10]
        if dt:
            events.append((dt, float(lat_), float(lon_)))

    rows = []
    used = 0
    for dstr, elat, elon in reversed(events):
        d0 = _dt.date.fromisoformat(dstr)
        keys = [(d0 + _dt.timedelta(days=k)).strftime("%Y%m%d") for k in (-1, 0, 1)]
        if not all(k in dset for k in keys):
            continue
        arrs = [day_arr(k) for k in keys]
        if any(a is None for a in arrs):
            continue
        cum = arrs[0][0] + arrs[1][0] + arrs[2][0]
        lat = arrs[1][1]
        lon = arrs[1][2]
        h, w = cum.shape
        r0 = int(np.clip((90 - elat) / 0.1, 0, h - mod.WIN))
        c0 = int(np.clip((elon + 180) / 0.1, 0, w - mod.WIN))
        pos = None
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                rr = int(np.clip(r0 + dr * mod.STRIDE, 0, h - mod.WIN))
                cc = int(np.clip(c0 + dc * mod.STRIDE, 0, w - mod.WIN))
                win = cum[rr:rr + mod.WIN, cc:cc + mod.WIN]
                if np.isfinite(win).mean() < mod.MIN_VALID:
                    continue
                mx = float(np.nanmax(win))
                if pos is None or mx > pos[4]:
                    pos = (rr, cc, float(np.nanmean(lat[rr:rr + mod.WIN])),
                           float(np.nanmean(lon[cc:cc + mod.WIN])), mx)
        if pos is None:
            continue
        rr, cc, la, lo, mx = pos
        sev = "flood" if mx >= 100 else "none"
        png = os.path.join(outdir, "flp_%s_%d_%d.png" % (keys[1], rr, cc))
        mod.render(cum[rr:rr + mod.WIN, cc:cc + mod.WIN], png, keys[1], la, lo)
        pred, err = predict(api, "flood_precip_global", png)
        rows.append({
            "element": "flood_precip_global", "image": os.path.basename(png),
            "event_date": keys[1], "event_latlon": [round(elat, 2), round(elon, 2)],
            "truth_cum3_max": round(mx, 1), "truth_label": sev,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "error": err, "match": (pred.get("severity") == sev),
        })
        print("  flp %-28s 事件 %s 真值 %-8s(%6.1f mm) 模型 %s/%s %s" % (
            rows[-1]["image"], keys[1], sev, mx, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
        used += 1
        if used >= n_events:
            break

    if used == 0:
        # 无 GDACS 事件重叠：直接用连续 3 天找最大累积窗口（非事件驱动，标注清楚）
        print("  GDACS 无重叠，回退为最大 3 日累积窗口")
        trips = [(dates[i], dates[i + 1], dates[i + 2]) for i in range(len(dates) - 2)]
        for keys in reversed(trips[:n_events]):
            arrs = [day_arr(k) for k in keys]
            if any(a is None for a in arrs):
                continue
            cum = arrs[0][0] + arrs[1][0] + arrs[2][0]
            lat = arrs[1][1]
            lon = arrs[1][2]
            h, w = cum.shape
            best = None
            for rr in range(0, h - mod.WIN, mod.STRIDE * 4):
                for cc in range(0, w - mod.WIN, mod.STRIDE * 4):
                    win = cum[rr:rr + mod.WIN, cc:cc + mod.WIN]
                    if np.isfinite(win).mean() < mod.MIN_VALID:
                        continue
                    mx = float(np.nanmax(win))
                    if best is None or mx > best[4]:
                        best = (rr, cc, float(np.nanmean(lat[rr:rr + mod.WIN])),
                                float(np.nanmean(lon[cc:cc + mod.WIN])), mx)
            if best is None:
                continue
            rr, cc, la, lo, mx = best
            sev = "flood" if mx >= 100 else "none"
            png = os.path.join(outdir, "flp_nonevent_%s_%d_%d.png" % (keys[1], rr, cc))
            mod.render(cum[rr:rr + mod.WIN, cc:cc + mod.WIN], png, keys[1], la, lo)
            pred, err = predict(api, "flood_precip_global", png)
            rows.append({
                "element": "flood_precip_global", "image": os.path.basename(png),
                "event_date": keys[1], "event_latlon": [round(la, 2), round(lo, 2)],
                "truth_cum3_max": round(mx, 1), "truth_label": sev,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "error": err, "match": (pred.get("severity") == sev),
                "note": "非事件驱动，取最大 3 日累积窗口",
            })
            print("  flp %-28s 真值 %-8s(%6.1f mm) 模型 %s/%s %s" % (
                rows[-1]["image"], sev, mx, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
            used += 1
            if used >= n_events:
                break
    return rows



# ---------------- drought_ndvi_clim（MODIS MOD13C2 NDVI 百分位） ----------------
def run_ndvi(mod, ndvi_dir, outdir, api, n_months=2):
    import glob as _g
    import re as _re
    import datetime as _dt
    files = {}
    for f in _g.glob(os.path.join(ndvi_dir, "*.hdf")):
        m = _re.search(r"\.A(\d{4})(\d{3})\.", os.path.basename(f))
        if m:
            files[(int(m.group(1)), int(m.group(2)))] = f
    keys = sorted(files)
    if not keys:
        print("  no ndvi files")
        return []
    stack, stamps = [], []
    for (y, doy) in keys:
        stack.append(mod.load_china(files[(y, doy)]))
        stamps.append("A%d%03d" % (y, doy))
    stack = np.stack(stack)
    months = [(_dt.date(2001, 1, 1) + _dt.timedelta(days=int(st[1:]) - 1)).month for st in stamps]
    pct = np.full_like(stack, np.nan)
    D = stack.shape[0]
    for mo in range(1, 13):
        idx = [i for i in range(D) if months[i] == mo]
        if len(idx) < 2:
            continue
        grp = stack[idx]
        for j, i in enumerate(idx):
            v = grp[j]
            lower = np.nansum(grp < v, axis=0)
            valid = np.sum(np.isfinite(grp), axis=0)
            pct[i] = np.where(valid >= 2, 100.0 * lower / np.maximum(valid, 1), np.nan)

    h, w = pct.shape[1], pct.shape[2]
    lats = mod.LATB[1] - np.arange(h) * mod.RES
    lons = mod.LONB[0] + np.arange(w) * mod.RES
    rows = []
    for i in range(D - 1, max(-1, D - 1 - n_months), -1):
        stamp, p = stamps[i], pct[i]
        pos, neg = [], []
        for r0 in range(0, h - mod.WIN + 1, mod.STRIDE):
            for c0 in range(0, w - mod.WIN + 1, mod.STRIDE):
                wp = p[r0:r0 + mod.WIN, c0:c0 + mod.WIN]
                if np.isfinite(wp).mean() < mod.MIN_VALID:
                    continue
                mp = float(np.nanmean(wp))
                if mp <= 10:
                    pos.append((mp, r0, c0))
                elif mp >= 40:
                    neg.append((mp, r0, c0))
        pos.sort()
        neg.sort(reverse=True)
        picks = [(True, x) for x in pos[:2]] + ([(False, neg[0])] if neg else [])
        for det, (mp, r0, c0) in picks:
            la = float(lats[r0 + mod.WIN // 2])
            lo = float(lons[c0 + mod.WIN // 2])
            png = os.path.join(outdir, "ndp_%s_r%d_c%d.png" % (stamp, r0, c0))
            mod.render(p[r0:r0 + mod.WIN, c0:c0 + mod.WIN], png, stamp, la, lo)
            sev = "drought" if det else "none"
            pred, err = predict(api, "drought_ndvi_clim", png)
            rows.append({
                "element": "drought_ndvi_clim", "image": os.path.basename(png), "stamp": stamp,
                "truth_pct": round(mp, 1), "truth_label": sev,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "error": err, "match": (pred.get("detected") == det),
            })
            print("  ndvi %-30s 真值 %-8s(%5.1f) 模型 %s/%s %s" % (
                rows[-1]["image"], sev, mp, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- geomagnetic（Dst 指数） ----------------
def run_geomagnetic(mod, dst_dir, outdir, api, months=("2026-05", "2026-06")):
    import glob as _g
    import datetime as _dt
    import requests as _rq
    os.makedirs(dst_dir, exist_ok=True)
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    rows = []
    for ym in months:
        local = os.path.join(dst_dir, "dst_%s.txt" % ym.replace("-", ""))
        if not (os.path.exists(local) and os.path.getsize(local) > 500):
            y, m = ym.split("-")
            url = "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/%s%s/dst%s.for.request" % (y, m, ym[2:].replace("-", ""))
            try:
                r = _rq.get(url, proxies=proxy, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200 and len(r.content) > 500:
                    open(local, "w", encoding="utf-8").write(r.text)
                    print("  downloaded", os.path.basename(local), len(r.content), "bytes")
            except Exception as e:
                print("  fetch fail", ym, str(e)[:60])
        if not os.path.exists(local):
            continue
        days = mod.parse(open(local, encoding="utf-8", errors="replace").read())
        if not days:
            continue
        graded = []
        for d, vals in days.items():
            lab, mn = mod.grade(vals)
            if lab:
                graded.append((mn, d, vals, lab))
        if not graded:
            continue
        graded.sort()
        picks = [graded[0], graded[-1]]
        for mn, d, vals, lab in picks:
            png = os.path.join(outdir, "dst_%s_%s.png" % (d.strftime("%Y%m%d"), lab))
            mod.draw(vals, png, lab, True)
            pred, err = predict(api, "geomagnetic", png)
            rows.append({
                "element": "geomagnetic", "image": os.path.basename(png), "date": str(d),
                "truth_min_dst": mn, "truth_label": lab,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "model_min_dst": pred.get("min_dst_nt"), "error": err,
                "match": (pred.get("severity") == lab),
            })
            print("  dst %-28s 真值 %-14s(%5d nT) 模型 %s/%s %s" % (
                rows[-1]["image"], lab, mn, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- haze（MODIS MOD08 AOD，OPeNDAP 取变量） ----------------
def run_haze(mod, outdir, api, date_="2026-06-16"):
    import re as _re
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from credentials_local import EARTHDATA_TOKEN as TOK
    except ImportError:
        from config_local import EARTHDATA_TOKEN as TOK
    # 1) CMR 查当日 granule
    y = date_[:4]
    doy = (__import__("datetime").date.fromisoformat(date_) - __import__("datetime").date(int(y), 1, 1)).days + 1
    q = ("https://cmr.earthdata.nasa.gov/search/granules.json?short_name=MOD08_D3"
         "&temporal=%sT00:00:00Z,%sT23:59:59Z&page_size=1" % (date_, date_))
    r = _rq.get(q, proxies=proxy, timeout=60)
    ent = r.json().get("feed", {}).get("entry", [])
    if not ent:
        print("  no MOD08 granule for", date_)
        return []
    name = None
    for l in ent[0].get("links", []):
        href = l.get("href", "")
        if href.endswith(".hdf") and "MOD08_D3.A" in href:
            name = os.path.basename(href)
            break
    if not name:
        print("  cannot find hdf filename")
        return []
    mm = _re.search(r"A(\d{4})(\d{3})\.", name)
    yy2, dd2 = (mm.group(1), int(mm.group(2))) if mm else (y, doy)
    url = ("https://ladsweb.modaps.eosdis.nasa.gov/opendap/RemoteResources/laads/allData/61/"
           "MOD08_D3/%s/%03d/%s.ascii?%s[0:1:179][0:1:359]" % (yy2, dd2, name, mod.VAR))
    r = _rq.get(url, headers={"Authorization": "Bearer " + TOK}, proxies=proxy, timeout=180)
    if r.status_code != 200:
        print("  opendap fail", r.status_code, r.text[:80])
        return []
    grid = np.full((180, 360), np.nan)
    for line in r.text.splitlines():
        if not line.startswith(mod.VAR + "["):
            continue
        try:
            head, rest = line.split("],", 1)
            iy = int(head.split("[")[-1])
            vals = [float(x) for x in rest.split(",")]
        except Exception:
            continue
        if 0 <= iy < 180:
            grid[iy, :min(360, len(vals))] = vals[:360]
    grid = grid * mod.SCALE
    grid[(grid <= 0) | (grid > 5)] = np.nan
    print("  AOD grid valid frac %.2f" % float(np.isfinite(grid).mean()))

    h, w = grid.shape
    cand = []
    for yy in range(0, h - mod.WIN + 1, mod.STRIDE):
        for xx in range(0, w - mod.WIN + 1, mod.STRIDE):
            sub = grid[yy:yy + mod.WIN, xx:xx + mod.WIN]
            v = sub[np.isfinite(sub)]
            if v.size < mod.WIN * mod.WIN * 0.4:
                continue
            cand.append((float(v.mean()), float(v.max()), yy, xx))
    if not cand:
        print("  no valid windows")
        return []
    cand.sort(key=lambda c: -c[0])
    rows = []
    picks = [cand[0], cand[-1]]
    for mean_aod, mx, yy, xx in picks:
        sub = grid[yy:yy + mod.WIN, xx:xx + mod.WIN]
        lab = mod.grade(mean_aod, mx)
        png = os.path.join(outdir, "haze_%s_%d_%d.png" % (date_.replace("-", ""), yy, xx))
        mod.render(sub, png)
        pred, err = predict(api, "haze", png)
        rows.append({
            "element": "haze", "image": os.path.basename(png), "date": date_,
            "truth_mean_aod": round(mean_aod, 3), "truth_max_aod": round(mx, 3),
            "truth_label": lab,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "model_mean_aod": pred.get("mean_aod"), "error": err,
            "match": (pred.get("severity") == lab),
        })
        print("  haze %-28s 真值 %-12s(mean %.2f) 模型 %s/%s %s" % (
            rows[-1]["image"], lab, mean_aod, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- sanddust（MOD08 Deep Blue AOD 陆地） ----------------
def run_sanddust(mod, outdir, api, date_="2026-06-16"):
    import re as _re
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from credentials_local import EARTHDATA_TOKEN as TOK
    except ImportError:
        from config_local import EARTHDATA_TOKEN as TOK
    q = ("https://cmr.earthdata.nasa.gov/search/granules.json?short_name=MOD08_D3"
         "&temporal=%sT00:00:00Z,%sT23:59:59Z&page_size=1" % (date_, date_))
    ent = _rq.get(q, proxies=proxy, timeout=60).json().get("feed", {}).get("entry", [])
    if not ent:
        print("  no MOD08 granule")
        return []
    name = None
    for l in ent[0].get("links", []):
        href = l.get("href", "")
        if href.endswith(".hdf") and "MOD08_D3.A" in href:
            name = os.path.basename(href)
            break
    if not name:
        print("  no filename")
        return []
    mm = _re.search(r"A(\d{4})(\d{3})\.", name)
    yy, dd = mm.group(1), int(mm.group(2))
    url = ("https://ladsweb.modaps.eosdis.nasa.gov/opendap/RemoteResources/laads/allData/61/"
           "MOD08_D3/%s/%03d/%s.ascii?%s[0:1:179][0:1:359]" % (yy, dd, name, mod.VAR))
    r = _rq.get(url, headers={"Authorization": "Bearer " + TOK}, proxies=proxy, timeout=180)
    if r.status_code != 200:
        print("  opendap fail", r.status_code)
        return []
    grid = np.full((180, 360), np.nan)
    for line in r.text.splitlines():
        if not line.startswith(mod.VAR + "["):
            continue
        try:
            head, rest = line.split("],", 1)
            iy = int(head.split("[")[-1])
            vals = [float(x) for x in rest.split(",")]
        except Exception:
            continue
        if 0 <= iy < 180:
            grid[iy, :min(360, len(vals))] = vals[:360]
    grid = grid * mod.SCALE
    grid[(grid <= 0) | (grid > 5)] = np.nan

    h, w = grid.shape
    cand = []
    for yy_ in range(0, h - mod.WIN + 1, mod.STRIDE):
        for xx_ in range(0, w - mod.WIN + 1, mod.STRIDE):
            sub = grid[yy_:yy_ + mod.WIN, xx_:xx_ + mod.WIN]
            v = sub[np.isfinite(sub)]
            if v.size < mod.WIN * mod.WIN * 0.3:
                continue
            cand.append((float(v.mean()), float(v.max()), yy_, xx_))
    if not cand:
        print("  no windows")
        return []
    cand.sort(key=lambda c: -c[1])
    rows = []
    for mean_aod, mx, yy_, xx_ in [cand[0], cand[-1]]:
        sub = grid[yy_:yy_ + mod.WIN, xx_:xx_ + mod.WIN]
        lab = mod.grade(mean_aod, mx)
        png = os.path.join(outdir, "dust_%s_%d_%d.png" % (date_.replace("-", ""), yy_, xx_))
        mod.render(sub, png)
        pred, err = predict(api, "sanddust", png)
        rows.append({
            "element": "sanddust", "image": os.path.basename(png), "date": date_,
            "truth_mean_aod": round(mean_aod, 3), "truth_max_aod": round(mx, 3),
            "truth_label": lab,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "model_mean_aod": pred.get("mean_aod"), "error": err,
            "match": (pred.get("severity") == lab),
        })
        print("  dust %-28s 真值 %-11s(mean %.2f max %.2f) 模型 %s/%s %s" % (
            rows[-1]["image"], lab, mean_aod, mx, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- uvai_dust（OMI OMAERUVd 日产品） ----------------
def run_uvai(mod, raw_dir, outdir, api, date_="2026-06-16"):
    import re as _re
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from credentials_local import EARTHDATA_TOKEN as TOK
    except ImportError:
        from config_local import EARTHDATA_TOKEN as TOK
    os.makedirs(raw_dir, exist_ok=True)
    q = ("https://cmr.earthdata.nasa.gov/search/granules.json?short_name=OMAERUVd"
         "&temporal=%sT00:00:00Z,%sT23:59:59Z&page_size=1" % (date_, date_))
    ent = _rq.get(q, proxies=proxy, timeout=60).json().get("feed", {}).get("entry", [])
    if not ent:
        print("  no OMAERUVd granule")
        return []
    href = None
    for l in ent[0].get("links", []):
        h = l.get("href", "")
        if h.endswith(".he5") and h.startswith("https://data.gesdisc"):
            href = h
            break
    if not href:
        print("  no data link")
        return []
    local = os.path.join(raw_dir, os.path.basename(href))
    if not (os.path.exists(local) and os.path.getsize(local) > 200_000):
        r = _rq.get(href, headers={"Authorization": "Bearer " + TOK}, proxies=proxy, timeout=300)
        if r.status_code != 200:
            print("  download fail", r.status_code)
            return []
        open(local, "wb").write(r.content)
        print("  downloaded %s  %.0f KB" % (os.path.basename(local), len(r.content) / 1024))

    a = mod.load(local)
    h, w = a.shape
    cand = []
    for yy in range(0, h - mod.WIN + 1, mod.STRIDE):
        for xx in range(0, w - mod.WIN + 1, mod.STRIDE):
            sub = a[yy:yy + mod.WIN, xx:xx + mod.WIN]
            v = sub[np.isfinite(sub)]
            if v.size < mod.WIN * mod.WIN * mod.MIN_VALID:
                continue
            cand.append((float(v.max()), yy, xx))
    if not cand:
        print("  no windows")
        return []
    cand.sort(reverse=True)
    rows = []
    # 经纬度：OMAERUVd 1°×1.25°，lat 从 -90 到 +90，lon -180~180
    lats = -90 + np.arange(h) * 1.0
    lons = -180 + np.arange(w) * 1.25
    for mx, yy, xx in [cand[0], cand[-1]]:
        sub = a[yy:yy + mod.WIN, xx:xx + mod.WIN]
        sev = mod.severity_of(mx)
        paste = os.path.join(outdir, "uvai_%s_%d_%d.png" % (date_.replace("-", ""), yy, xx))
        mod.render(sub, paste, date_.replace("-", ""), float(lats[yy]), float(lons[xx]))
        pred, err = predict(api, "uvai_dust", paste)
        rows.append({
            "element": "uvai_dust", "image": os.path.basename(paste), "date": date_,
            "truth_max_uvai": round(mx, 2), "truth_label": sev,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "model_max_uvai": pred.get("max_uvai"), "error": err,
            "match": (pred.get("severity") == sev),
        })
        print("  uvai %-28s 真值 %-8s(%5.2f) 模型 %s/%s %s" % (
            rows[-1]["image"], sev, mx, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- wildfire_smoke（FIRMS 火点 + OMI UVAI） ----------------
def run_smoke(mod, firms_dir, uvai_dir, outdir, api, date_="2025-06-15"):
    import re as _re
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from credentials_local import EARTHDATA_TOKEN as TOK
    except ImportError:
        from config_local import EARTHDATA_TOKEN as TOK
    os.makedirs(uvai_dir, exist_ok=True)
    q = ("https://cmr.earthdata.nasa.gov/search/granules.json?short_name=OMAERUVd"
         "&temporal=%sT00:00:00Z,%sT23:59:59Z&page_size=1" % (date_, date_))
    ent = _rq.get(q, proxies=proxy, timeout=60).json().get("feed", {}).get("entry", [])
    if not ent:
        print("  no OMAERUVd")
        return []
    href = None
    for l in ent[0].get("links", []):
        h = l.get("href", "")
        if h.endswith(".he5") and h.startswith("https://data.gesdisc"):
            href = h
            break
    if not href:
        print("  no data link")
        return []
    local = os.path.join(uvai_dir, os.path.basename(href))
    if not (os.path.exists(local) and os.path.getsize(local) > 200_000):
        r = _rq.get(href, headers={"Authorization": "Bearer " + TOK}, proxies=proxy, timeout=300)
        if r.status_code != 200:
            print("  download fail", r.status_code)
            return []
        open(local, "wb").write(r.content)
        print("  downloaded UVAI %.0f KB" % (len(r.content) / 1024))

    uv = mod.load_uvai(local)
    fires = mod.load_fires(os.path.join(firms_dir, "fire_%s.csv" % date_))
    if not fires:
        # 换最近一天有火点的
        import glob as _g
        cands = sorted(_g.glob(os.path.join(firms_dir, "fire_2025-*.csv")))
        for cf in reversed(cands):
            fires = mod.load_fires(cf)
            if fires:
                date_ = os.path.basename(cf)[5:15]
                print("  use fires from", date_)
                break
    h, w = uv.shape
    lat = -90 + np.arange(h) * 1.0
    lon = -180 + np.arange(w) * 1.25
    rows = []
    picks = []
    for _f in (fires[:40] if fires else []):
        flat, flon = float(_f[0]), float(_f[1])
        iy = int(round((flat + 90) / 1.0))
        ix = int(round((flon + 180) / 1.25))
        for dy in (-2, 2):
            for dx in (-2, 2):
                yy, xx = iy + dy, ix + dx
                if 0 <= yy < h - mod.WIN and 0 <= xx < w - mod.WIN:
                    sub = uv[yy:yy + mod.WIN, xx:xx + mod.WIN]
                    if np.isfinite(sub).mean() >= mod.MIN_VALID:
                        picks.append((float(np.nanmax(sub)), yy, xx))
    if not picks:
        print("  no valid windows near fires")
        return []
    picks.sort(reverse=True)
    seen = set()
    for mx, yy, xx in picks:
        if (yy, xx) in seen:
            continue
        seen.add((yy, xx))
        sub = uv[yy:yy + mod.WIN, xx:xx + mod.WIN]
        sev = mod.severity_of(mx)
        png = os.path.join(outdir, "smoke_%s_%d_%d.png" % (date_.replace("-", ""), yy, xx))
        mod.render(sub, png, date_.replace("-", ""), float(lat[min(yy + 4, len(lat) - 1)]),
                   float(lon[min(xx + 4, len(lon) - 1)]))
        pred, err = predict(api, "wildfire_smoke", png)
        rows.append({
            "element": "wildfire_smoke", "image": os.path.basename(png), "date": date_,
            "truth_max_uvai": round(mx, 2), "truth_label": sev,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "model_max_uvai": pred.get("max_uvai"), "error": err,
            "match": (pred.get("severity") == sev),
        })
        print("  smoke %-30s 真值 %-8s(%5.2f) 模型 %s/%s %s" % (
            rows[-1]["image"], sev, mx, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
        if len(rows) >= 4:
            break
    return rows



# ---------------- coral_dhw（NOAA CRW DHW，ERDDAP 截取） ----------------
def run_coral(mod, outdir, api, time_="2026-11-20T12:00:00Z",
              lat_rng=(-24, -14), lon_rng=(144, 154)):
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    q = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.csv?"
         "CRW_DHW[(%s)][(%d):1:(%d)][(%d):1:(%d)]" %
         (time_, lat_rng[0], lat_rng[1], lon_rng[0], lon_rng[1]))
    r = _rq.get(q, proxies=proxy, timeout=180)
    if r.status_code != 200:
        print("  erddap fail", r.status_code, r.text[:80])
        return []
    lats, lons, vals = [], [], []
    for line in r.text.splitlines()[2:]:
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            la, lo, v = float(parts[1]), float(parts[2]), parts[3]
            v = float(v) if v not in ("NaN", "", "null") else np.nan
        except Exception:
            continue
        lats.append(la); lons.append(lo); vals.append(v)
    if not vals:
        print("  no data rows")
        return []
    lat_u = sorted(set(lats), reverse=True)
    lon_u = sorted(set(lons))
    grid = np.full((len(lat_u), len(lon_u)), np.nan)
    li = {v: i for i, v in enumerate(lat_u)}
    lj = {v: j for j, v in enumerate(lon_u)}
    for la, lo, v in zip(lats, lons, vals):
        grid[li[la], lj[lo]] = v
    print("  网格 %dx%d，有效 %.0f%%" % (grid.shape[0], grid.shape[1],
                                       100 * float(np.isfinite(grid).mean())))

    h, w = grid.shape
    cand = []
    stride = getattr(mod, "STRIDE", 16)
    for yy in range(0, h - mod.WIN + 1, stride):
        for xx in range(0, w - mod.WIN + 1, stride):
            sub = grid[yy:yy + mod.WIN, xx:xx + mod.WIN]
            v = sub[np.isfinite(sub)]
            if v.size < mod.WIN * mod.WIN * 0.3:
                continue
            cand.append((float(v.max()), yy, xx))
    if not cand:
        print("  no windows")
        return []
    cand.sort(reverse=True)
    stamp = time_[:10].replace("-", "")
    rows = []
    for mx, yy, xx in [cand[0], cand[-1]]:
        sub = grid[yy:yy + mod.WIN, xx:xx + mod.WIN]
        sev = mod.severity_of(mx)
        paste = os.path.join(outdir, "coral_%s_%d_%d.png" % (stamp, yy, xx))
        mod.render(sub, paste, stamp,
                   float(lat_u[yy + mod.WIN // 2]), float(lon_u[xx + mod.WIN // 2]))
        pred, err = predict(api, "coral_dhw", paste)
        rows.append({
            "element": "coral_dhw", "image": os.path.basename(paste), "date": time_[:10],
            "truth_max_dhw": round(mx, 2), "truth_label": sev,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "model_max_dhw": pred.get("max_dhw"), "error": err,
            "match": (pred.get("severity") == sev),
        })
        print("  coral %-30s 真值 %-8s(%5.2f) 模型 %s/%s %s" % (
            rows[-1]["image"], sev, mx, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- volcano_so2（OMI OMSO2e 单日） ----------------
def run_so2(mod, raw_dir, outdir, api, date_="2025-06-15"):
    import re as _re
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from credentials_local import EARTHDATA_TOKEN as TOK
    except ImportError:
        from config_local import EARTHDATA_TOKEN as TOK
    os.makedirs(raw_dir, exist_ok=True)
    q = ("https://cmr.earthdata.nasa.gov/search/granules.json?short_name=OMSO2e"
         "&temporal=%sT00:00:00Z,%sT23:59:59Z&page_size=1" % (date_, date_))
    ent = _rq.get(q, proxies=proxy, timeout=60).json().get("feed", {}).get("entry", [])
    if not ent:
        print("  no OMSO2e granule")
        return []
    href = None
    for l in ent[0].get("links", []):
        h = l.get("href", "")
        if h.endswith(".he5") and h.startswith("https://data.gesdisc"):
            href = h
            break
    if not href:
        print("  no data link")
        return []
    local = os.path.join(raw_dir, os.path.basename(href))
    if not (os.path.exists(local) and os.path.getsize(local) > 1_000_000):
        r = _rq.get(href, headers={"Authorization": "Bearer " + TOK}, proxies=proxy, timeout=300)
        if r.status_code != 200:
            print("  download fail", r.status_code)
            return []
        open(local, "wb").write(r.content)
        print("  downloaded SO2 %.1f MB" % (len(r.content) / 1e6))

    so2, lat, lon = mod.load(local)
    h, w = so2.shape
    cand = []
    for yy in range(0, h - mod.WIN + 1, mod.STRIDE):
        for xx in range(0, w - mod.WIN + 1, mod.STRIDE):
            sub = so2[yy:yy + mod.WIN, xx:xx + mod.WIN]
            v = sub[np.isfinite(sub)]
            if v.size < mod.WIN * mod.WIN * mod.MIN_VALID:
                continue
            cand.append((float(v.max()), yy, xx))
    if not cand:
        print("  no windows")
        return []
    cand.sort(reverse=True)
    stamp = date_.replace("-", "")
    rows = []
    for mx, yy, xx in [cand[0], cand[-1]]:
        sub = so2[yy:yy + mod.WIN, xx:xx + mod.WIN]
        sev = mod.severity_of(mx)
        png = os.path.join(outdir, "so2_%s_%d_%d.png" % (stamp, yy, xx))
        la_v = float(lat[min(yy + 30, h - 1), min(xx + 30, w - 1)])
        lo_v = float(lon[min(yy + 30, h - 1), min(xx + 30, w - 1)])
        mod.render(sub, png, stamp, la_v, lo_v, mx)
        pred, err = predict(api, "volcano_so2", png)
        rows.append({
            "element": "volcano_so2", "image": os.path.basename(png), "date": date_,
            "truth_max_so2": round(mx, 2), "truth_label": sev,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "model_max_so2": pred.get("max_so2_du"), "error": err,
            "match": (pred.get("severity") == sev),
            "note": "无事件库，窗口按最大 SO2 选取",
        })
        print("  so2 %-30s 真值 %-8s(%5.2f) 模型 %s/%s %s" % (
            rows[-1]["image"], sev, mx, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- enso（OISST 距平，ERDDAP 截取） ----------------
def run_enso(mod, outdir, api, time_="2026-11-01T12:00:00Z"):
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    q = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.csv?"
         "anom[(%s)][(0.0)][(-30):1:(30)][(180):1:(280)]" % time_)
    r = _rq.get(q, proxies=proxy, timeout=180)
    if r.status_code != 200:
        print("  erddap fail", r.status_code, r.text[:80])
        return []
    lats, lons, vals = [], [], {}
    for line in r.text.splitlines()[2:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            la, lo, v = float(parts[2]), float(parts[3]), parts[4]
            v = float(v) if v not in ("NaN", "", "null") else np.nan
        except Exception:
            continue
        lats.append(la); lons.append(lo); vals[(la, lo)] = v
    lat_u = sorted(set(lats)); lon_u = sorted(set(lons))
    a = np.full((len(lat_u), len(lon_u)), np.nan)
    li = {v: i for i, v in enumerate(lat_u)}; lj = {v: i for i, v in enumerate(lon_u)}
    for (la, lo), v in vals.items():
        a[li[la], lj[lo]] = v
    lat = np.array(lat_u); lon = np.array(lon_u)
    # 0-360 经度
    lon360 = np.where(lon < 0, lon + 360, lon)
    order = np.argsort(lon360)
    a = a[:, order]; lon360 = lon360[order]
    val = mod.nino34_mean(a, lat, lon360)
    phase, v2 = mod.phase_of(val)
    ymd = time_[:10].replace("-", "")
    png = os.path.join(outdir, "enso_%s_anom.png" % ymd)
    mod.render_anom(a, lat, lon360, png, ymd, phase, val)
    pred, err = predict(api, "enso", png)
    lab = pred.get("severity")
    rows = [{
        "element": "enso", "image": os.path.basename(png), "date": time_[:10],
        "truth_nino34_anom": round(float(val), 2) if val is not None else None,
        "truth_label": phase, "model_detected": pred.get("detected"),
        "model_severity": lab, "error": err,
        "match": (str(lab).lower().replace("_", " ") == phase.lower()),
        "note": "用当日距平代替 3 个月 ONI",
    }]
    print("  enso Nino3.4=%.2f 真值 %s 模型 %s %s" % (
        val if val is not None else float("nan"), phase, lab,
        "OK" if rows[0]["match"] else "MISS"))
    return rows



# ---------------- iod（OISST 距平，印度洋） ----------------
def run_iod(mod, outdir, api, time_="2026-11-01T12:00:00Z"):
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    q = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.csv?"
         "anom[(%s)][(0.0)][(-20):1:(20)][(40):1:(120)]" % time_)
    r = _rq.get(q, proxies=proxy, timeout=180)
    if r.status_code != 200:
        print("  erddap fail", r.status_code, r.text[:80])
        return []
    lats, lons, vals = [], [], {}
    for line in r.text.splitlines()[2:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            la, lo, v = float(parts[2]), float(parts[3]), parts[4]
            v = float(v) if v not in ("NaN", "", "null") else np.nan
        except Exception:
            continue
        lats.append(la); lons.append(lo); vals[(la, lo)] = v
    lat = np.array(sorted(set(lats)))
    lon = np.array(sorted(set(lons)))
    a = np.full((len(lat), len(lon)), np.nan)
    li = {v: i for i, v in enumerate(lat)}; lj = {v: i for i, v in enumerate(lon)}
    for (la, lo), v in vals.items():
        a[li[la], lj[lo]] = v
    dmi = mod.dmi_from_field(a, lat, lon)
    phase = mod.iod_phase(dmi)
    ymd = time_[:10].replace("-", "")
    png = os.path.join(outdir, "iod_%s_anom.png" % ymd)
    mod.render_mean(a, (lat, lon), png, title="%s SST Anomaly (IOD: %s)" % (ymd, phase))
    pred, err = predict(api, "iod", png)
    truth = {"positive IOD": "positive_iod", "negative IOD": "negative_iod",
             "neutral": "neutral_iod"}.get(str(phase), str(phase))
    lab = pred.get("severity")
    rows = [{
        "element": "iod", "image": os.path.basename(png), "date": time_[:10],
        "truth_dmi": round(float(dmi), 2) if dmi is not None else None,
        "truth_label": truth, "model_detected": pred.get("detected"),
        "model_severity": lab, "error": err, "match": (lab == truth),
    }]
    print("  iod DMI=%.2f 真值 %s 模型 %s %s" % (
        dmi if dmi is not None else float("nan"), truth, lab,
        "OK" if rows[0]["match"] else "MISS"))
    return rows



# ---------------- marine_heatwave（OISST 距平，30°窗） ----------------
def run_mhw(mod, outdir, api, time_="2024-08-01T12:00:00Z",
            lat_rng=(30, 60), lon_rng=(190, 220)):
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    q = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.csv?"
         "anom[(%s)][(0.0)][(%d):1:(%d)][(%d):1:(%d)]" %
         (time_, lat_rng[0], lat_rng[1], lon_rng[0], lon_rng[1]))
    r = _rq.get(q, proxies=proxy, timeout=180)
    if r.status_code != 200:
        print("  erddap fail", r.status_code)
        return []
    lats, lons, vals = [], [], {}
    for line in r.text.splitlines()[2:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            la, lo, v = float(parts[2]), float(parts[3]), parts[4]
            v = float(v) if v not in ("NaN", "", "null") else np.nan
        except Exception:
            continue
        lats.append(la); lons.append(lo); vals[(la, lo)] = v
    lat = np.array(sorted(set(lats))); lon = np.array(sorted(set(lons)))
    a = np.full((len(lat), len(lon)), np.nan)
    li = {v: i for i, v in enumerate(lat)}; lj = {v: i for i, v in enumerate(lon)}
    for (la, lo), v in vals.items():
        a[li[la], lj[lo]] = v
    frac = float(np.nanmean(a > 1.0)) * 100.0
    lab = "marine_heatwave" if frac >= 20.0 else "normal"
    ymd = time_[:10].replace("-", "")
    png = os.path.join(outdir, "mhw_%s_anom.png" % ymd)
    mod._render(a, float(lat[0]), float(lat[-1]), float(lon[0]), float(lon[-1]),
                png, "%s SST anomaly (hot area %.0f%%)" % (ymd, frac))
    pred, err = predict(api, "marine_heatwave", png)
    model_lab = pred.get("severity")
    rows = [{
        "element": "marine_heatwave", "image": os.path.basename(png), "date": time_[:10],
        "truth_area_frac": round(frac, 1), "truth_label": lab,
        "model_detected": pred.get("detected"), "model_severity": model_lab,
        "model_area_frac": pred.get("area_frac"), "error": err,
        "match": (model_lab == lab),
    }]
    print("  mhw 热面积 %.1f%% 真值 %s 模型 %s %s" % (
        frac, lab, model_lab, "OK" if rows[0]["match"] else "MISS"))
    return rows



# ---------------- snowstorm（AMSR2 SWE → 雪深分级） ----------------
def run_snowstorm(mod, swe_dir, outdir, api, n_days=2):
    import glob as _g
    import re as _re
    files = sorted(_g.glob(os.path.join(swe_dir, "*.he5")))
    if not files:
        print("  no swe files")
        return []
    rows = []
    for f in reversed(files[-6:]):
        m = _re.search(r"(\d{8})", os.path.basename(f))
        stamp = m.group(1) if m else "x"
        try:
            swe, lat, lon = mod.load(f)
        except Exception as e:
            print("  load fail", stamp, str(e)[:50])
            continue
        H, W = swe.shape
        WIN = mod.WIN
        cand = []
        for y in range(0, H - WIN + 1, WIN // 2):
            for x in range(0, W - WIN + 1, WIN // 2):
                sub = swe[y:y + WIN, x:x + WIN]
                if np.isfinite(sub).mean() < 0.5:
                    continue
                lab, snow_cm, q75 = mod.grade_of_window(sub)
                if lab is None:
                    continue
                cand.append((snow_cm, y, x, lab, q75))
        if not cand:
            continue
        cand.sort(reverse=True)
        picks = [cand[0], cand[-1]]
        for snow_cm, y, x, lab, q75 in picks:
            sub = swe[y:y + WIN, x:x + WIN]
            png = os.path.join(outdir, "storm_%s_%d_%d.png" % (stamp, y, x))
            mod.render_grade(sub, png, stamp, lab, snow_cm, q75)
            pred, err = predict(api, "snowstorm", png)
            model_lab = pred.get("severity")
            rows.append({
                "element": "snowstorm", "image": os.path.basename(png), "stamp": stamp,
                "truth_snow_cm": round(float(snow_cm), 1), "truth_label": lab,
                "model_detected": pred.get("detected"), "model_severity": model_lab,
                "error": err, "match": (model_lab == lab),
            })
            print("  storm %-28s 真值 %-12s(%5.1f cm) 模型 %s/%s %s" % (
                rows[-1]["image"], lab, snow_cm, pred.get("detected"), model_lab,
                "OK" if rows[-1]["match"] else "MISS"))
        if len(rows) >= n_days * 2:
            break
    return rows



# ---------------- flare（SWPC XRS + SDO AIA193） ----------------
def run_flare(mod, outdir, api):
    import re as _re
    import datetime as _dt
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    r = _rq.get("https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
                proxies=proxy, timeout=120)
    if r.status_code != 200:
        print("  swpc fail", r.status_code)
        return []
    rows = r.json()
    peaks = [(float(x["flux"]), x["time_tag"]) for x in rows
             if x.get("energy") == "0.1-0.8nm" and x.get("flux")]
    if not peaks:
        print("  no flux rows")
        return []
    peaks.sort()
    picks = [peaks[-1], peaks[0]]     # 最强 / 最弱
    out = []
    for flux, ttag in picks:
        when = _dt.datetime.strptime(ttag[:19], "%Y-%m-%dT%H:%M:%S")
        day_path = when.strftime("%Y/%m/%d")
        base = "https://sdo.gsfc.nasa.gov/assets/img/browse/%s/" % day_path
        rl = _rq.get(base, proxies=proxy, timeout=120)
        frames = []
        for m in _re.finditer(r'href="(\d{8})_(\d{6})_1024_0193\.jpg"', rl.text):
            t = _dt.datetime.strptime(m.group(1)[:8] + m.group(2), "%Y%m%d%H%M%S")
            frames.append((abs((t - when).total_seconds()), m.group(0).split('"')[1]))
        if not frames:
            print("  no sdo frames for", day_path)
            continue
        frames.sort()
        fn = frames[0][1]
        img = os.path.join(outdir, "sdo_%s.jpg" % fn[-21:-4])
        if not os.path.exists(img):
            rb = _rq.get(base + fn, proxies=proxy, timeout=180)
            if rb.status_code != 200:
                print("  sdo download fail", rb.status_code)
                continue
            open(img, "wb").write(rb.content)
        png = os.path.join(outdir, "flare_%s.png" % fn[-21:-4])
        mod.render(img, png, None)
        lab = mod.flux_grade(flux)
        pred, err = predict(api, "flare", png)
        truth_det = (lab != "quiet")
        rows2 = {
            "element": "flare", "image": os.path.basename(png), "time": ttag,
            "truth_flux": float(flux), "truth_class": lab,
            "truth_detected": truth_det,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "error": err, "match": (pred.get("detected") == truth_det),
        }
        out.append(rows2)
        print("  flare %-26s 通量 %.2e 真值 %-8s 模型 detected=%s %s" % (
            rows2["image"], flux, lab, pred.get("detected"),
            "OK" if rows2["match"] else "MISS"))
    return out



# ---------------- heatwave_lst / frost（MOD11C3 月度 LST 百分位） ----------------
_LST_CACHE = {}

def _fetch_mod11c3_months(months):
    import re as _re
    import requests as _rq
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from credentials_local import EARTHDATA_TOKEN as TOK
    except ImportError:
        from config_local import EARTHDATA_TOKEN as TOK
    q = ("https://cmr.earthdata.nasa.gov/search/granules.json?short_name=MOD11C3"
         "&temporal=2023-01-01T00:00:00Z,2026-06-30T23:59:59Z&page_size=100")
    ent = _rq.get(q, proxies=proxy, timeout=120).json().get("feed", {}).get("entry", [])
    name_by_key = {}
    for e in ent:
        for l in e.get("links", []):
            h = l.get("href", "")
            if h.endswith(".hdf") and "MOD11C3.A" in h:
                nm = os.path.basename(h)
                m = _re.search(r"A(\d{4})(\d{3})\.", nm)
                if m:
                    name_by_key["%s%03d" % (m.group(1), int(m.group(2)))] = nm
                break
    cache_file = os.path.join(ROOT, "data", "interim", "mod11c3_china_0p1.npz")
    if os.path.exists(cache_file):
        z = np.load(cache_file, allow_pickle=True)
        ck = [tuple(x) for x in z["months"]]
        if ck == months:
            print("  复用磁盘缓存", cache_file)
            return z["stack"], [str(x) for x in z["stamps"]]
    grids, stamps = [], []
    for (y, mo) in months:
        doy = (__import__("datetime").date(y, mo, 1) - __import__("datetime").date(y, 1, 1)).days + 1
        key = "%d%03d" % (y, doy)
        nm = name_by_key.get(key)
        if not nm:
            continue
        url = ("https://ladsweb.modaps.eosdis.nasa.gov/opendap/RemoteResources/laads/allData/61/"
               "MOD11C3/%d/%03d/%s.ascii?LST_Day_CMG[720:2:1439][5060:2:6299]" % (y, doy, nm))
        r = None
        for attempt in range(3):
            try:
                r = _rq.get(url, headers={"Authorization": "Bearer " + TOK},
                            proxies=proxy, timeout=300)
                if r.status_code == 200:
                    break
                print("  fetch retry", y, mo, r.status_code)
            except Exception as e:
                print("  fetch retry", y, mo, str(e)[:60])
            __import__("time").sleep(3 + attempt * 4)
        if r is None or r.status_code != 200:
            print("  fetch fail", y, mo)
            continue
        grid = np.full((360, 620), np.nan)
        for line in r.text.splitlines():
            if not line.startswith("LST_Day_CMG["):
                continue
            try:
                head, rest = line.split("],", 1)
                iy = int(head.split("[")[-1])
                vals = [float(x) for x in rest.split(",")]
            except Exception:
                continue
            if 0 <= iy < 360:
                grid[iy, :min(620, len(vals))] = vals[:620]
        grid = grid * 0.02
        grid[(grid < 200) | (grid > 360)] = np.nan
        grids.append(grid.astype("float32"))
        stamps.append(key)
        print("  month %s ok" % key)
    if not grids:
        return None, None
    stack = np.stack(grids)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.savez_compressed(cache_file, stack=stack, stamps=np.array(stamps),
                        months=np.array(months, dtype=object))
    print("  已缓存", cache_file)
    return stack, stamps


def run_lst(mod, which, outdir, api):
    months = [(y, m) for y in range(2023, 2027) for m in range(1, 13)]
    months = [x for x in months if x <= (2026, 6)]
    cache_key = tuple(months)
    global _LST_CACHE
    if cache_key not in _LST_CACHE:
        print("  拉取 MOD11C3 月度 LST（约 20–35 MB）...")
        stack, stamps = _fetch_mod11c3_months(months)
        if stack is None:
            return []
        _LST_CACHE[cache_key] = (stack, stamps)
    stack, stamps = _LST_CACHE[cache_key]
    D = stack.shape[0]
    pct = np.full_like(stack, np.nan)
    for i in range(D):
        v = stack[i]
        lower = np.nansum(stack < v, axis=0)
        valid = np.sum(np.isfinite(stack), axis=0)
        pct[i] = np.where(valid >= 10, 100.0 * lower / np.maximum(valid, 1), np.nan)
    h, w = pct.shape[1], pct.shape[2]
    lats = 54.0 - np.arange(h) * 0.1
    lons = 73.0 + np.arange(w) * 0.1
    rows = []
    cand = []
    for i in range(D - 1, max(-1, D - 6), -1):
        p = pct[i]
        for r0 in range(0, h - 16 + 1, 8):
            for c0 in range(0, w - 16 + 1, 8):
                wp = p[r0:r0 + 16, c0:c0 + 16]
                if np.isfinite(wp).mean() < 0.5:
                    continue
                cand.append((float(np.nanmean(wp)), i, r0, c0))
    if not cand:
        return []
    if which == "heatwave_lst":
        cand.sort(reverse=True)
        picks = cand[:2]
    else:
        cand.sort()
        picks = cand[:2]
    for mp, i, r0, c0 in picks:
        la = float(lats[r0 + 8]); lo = float(lons[c0 + 8])
        png = os.path.join(outdir, "%s_%s_r%d_c%d.png" % (which[:3], stamps[i], r0, c0))
        mod.render(pct[i][r0:r0 + 16, c0:c0 + 16], png, stamps[i], la, lo)
        pred, err = predict(api, which, png)
        if which == "heatwave_lst":
            truth = "heatwave" if mp >= 90 else "none"
        else:
            truth = "frost" if mp <= 10 else "none"
        det_truth = (truth != "none")
        rows.append({
            "element": which, "image": os.path.basename(png), "stamp": stamps[i],
            "truth_pct": round(mp, 1), "truth_label": truth,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "error": err, "match": bool(pred.get("detected")) == det_truth,
        })
        print("  %s %-28s 百分位 %5.1f 真值 %-8s 模型 %s/%s %s" % (
            which, rows[-1]["image"], mp, truth, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- sea_ice_s2_arctic（Sentinel-2 L2A，NDSI） ----------------
def run_seaice_s2(mod, raw_dir, outdir, api):
    import glob as _g
    zips = sorted(_g.glob(os.path.join(raw_dir, "*.zip")))
    if not zips:
        print("  no s2 scene")
        return []
    zp = max(zips, key=os.path.getsize)
    bands, scl = mod.load_scene(zp)
    ok, msg = mod.valid_scene(bands)
    if not ok:
        print("  scene invalid:", msg)
        return []
    valid, land = mod.land_mask(bands, scl)
    ndsi = mod.ndsi_of(bands)
    water = valid & (~land)
    wins = mod.find_windows(water, 4)
    if not wins:
        print("  no windows")
        return []
    tag = os.path.basename(zp).split("_")[2][:8] + "_" + os.path.basename(zp).split("_")[5][:6]
    cand = []
    for y0, x0 in wins:
        w = water[y0:y0 + mod.WIN, x0:x0 + mod.WIN]
        n = ndsi[y0:y0 + mod.WIN, x0:x0 + mod.WIN]
        if w.sum() < 100:
            continue
        frac = float(np.mean(n[w] > mod.NDSI_SNOW))
        cand.append((frac, y0, x0))
    if not cand:
        print("  no valid windows")
        return []
    cand.sort(reverse=True)
    rows = []
    for frac, y0, x0 in [cand[0], cand[-1]]:
        lab = "ice" if frac >= 0.15 else "open_water"
        png = os.path.join(outdir, "s2ice_%s_%d_%d.png" % (tag, y0, x0))
        mod.render_ndsi(ndsi[y0:y0 + mod.WIN, x0:x0 + mod.WIN], png, tag, lab, frac)
        pred, err = predict(api, "sea_ice_s2_arctic", png)
        rows.append({
            "element": "sea_ice_s2_arctic", "image": os.path.basename(png), "scene": os.path.basename(zp),
            "truth_ice_frac": round(frac, 3), "truth_label": lab,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "error": err, "match": (pred.get("detected") == (lab == "ice")),
        })
        print("  s2ice %-30s 冰比例 %.1f%% 真值 %-10s 模型 %s/%s %s" % (
            rows[-1]["image"], frac * 100, lab, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- flood_water（Sentinel-1 GRD 暗区占比） ----------------
def run_grd(mod, grd_dir, outdir, api):
    import glob as _g
    import zipfile
    import tifffile
    zips = sorted(_g.glob(os.path.join(grd_dir, "*.zip")))
    if not zips:
        print("  no grd zip")
        return []
    zp = max(zips, key=os.path.getsize)
    stamp = os.path.basename(zp).split("_")[4][:8]
    with zipfile.ZipFile(zp) as z:
        vv = [n for n in z.namelist() if n.lower().endswith("vv-") or "-vv-" in n.lower()]
        tif = [n for n in z.namelist() if n.lower().endswith((".tiff", ".tif")) and "-vv-" in n.lower()]
        if not tif:
            print("  no vv tiff")
            return []
        tmp = os.path.join(ROOT, "data", "interim", "grd_vv_%s.tif" % stamp)
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        if not os.path.exists(tmp):
            with z.open(tif[0]) as f, open(tmp, "wb") as o:
                o.write(f.read())
    arr = tifffile.imread(tmp)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr[::4, ::4].astype("float32")          # 降采样，控制内存
    print("  vv 网格", arr.shape)
    lo, hi = np.percentile(arr, [1, 99])
    b = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    H, W = b.shape
    buckets = {"water": [], "land": []}
    for y in range(0, H - mod.WIN + 1, mod.WIN // 2):
        for x in range(0, W - mod.WIN + 1, mod.WIN // 2):
            win = b[y:y + mod.WIN, x:x + mod.WIN]
            df = float((win < 60).mean())
            if df > mod.HI:
                buckets["water"].append((df, x, y))
            elif df < mod.LO:
                buckets["land"].append((df, x, y))
    rows = []
    for lab in ("water", "land"):
        lst = buckets[lab]
        if not lst:
            continue
        lst.sort(key=lambda t: t[0], reverse=(lab == "water"))
        df, x, y = lst[0]
        crop = b[y:y + mod.WIN, x:x + mod.WIN].astype("uint8")
        png = os.path.join(outdir, "grd_%s_%d_%d.png" % (stamp, x, y))
        mod.render(crop, png, "Sentinel-1 GRD 相对后向散射 · flood_water · %s" % stamp, df, lab)
        pred, err = predict(api, "flood_water", png)
        rows.append({
            "element": "flood_water", "image": os.path.basename(png), "stamp": stamp,
            "truth_dark_frac": round(df, 3), "truth_label": lab,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "error": err, "match": (pred.get("severity") == lab),
        })
        print("  grd %-30s 暗区 %.1f%% 真值 %-6s 模型 %s/%s %s" % (
            rows[-1]["image"], df * 100, lab, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows




# ---------------- green_tide（Sentinel-2 L2A，FAI） ----------------
def run_greentide(mod, s2_dir, outdir, api):
    import glob as _g
    zips = sorted(_g.glob(os.path.join(s2_dir, "*.zip")))
    if not zips:
        print("  no s2 zip")
        return []
    zp = max(zips, key=os.path.getsize)
    b02, b03, b04, b08, b11, scl = mod.load_bands(zp)
    water = mod.water_mask(b02, b03, b04, scl)
    fai = mod.fai_of(b04, b08, b11)
    thr, med = mod.adaptive_threshold(fai, water)
    if thr is None:
        print("  threshold failed")
        return []
    wins = mod.find_windows(water, 3)
    cand = []
    for y0, x0 in wins:
        fw = fai[y0:y0 + mod.WIN, x0:x0 + mod.WIN]
        ww = water[y0:y0 + mod.WIN, x0:x0 + mod.WIN]
        lab, frac = mod.grade_of(fw, ww, thr)
        if lab is None:
            continue
        cand.append((frac, y0, x0, lab))
    if not cand:
        print("  no valid windows")
        return []
    cand.sort(reverse=True)
    tag = os.path.basename(zp).split("_")[2][:8]
    rows = []
    for frac, y0, x0, lab in [cand[0], cand[-1]]:
        png = os.path.join(outdir, "gtide_%s_%d_%d.png" % (tag, y0, x0))
        mod.render_fai(fai[y0:y0 + mod.WIN, x0:x0 + mod.WIN],
                       water[y0:y0 + mod.WIN, x0:x0 + mod.WIN], png, tag, lab, frac, thr)
        pred, err = predict(api, "green_tide", png)
        rows.append({
            "element": "green_tide", "image": os.path.basename(png), "scene": os.path.basename(zp),
            "truth_frac": round(frac, 3), "truth_label": lab,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "error": err, "match": (pred.get("severity") == lab),
        })
        print("  gtide %-30s 超阈占比 %.1f%% 真值 %-11s 模型 %s/%s %s" % (
            rows[-1]["image"], frac * 100, lab, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- hurricane_opt（VIIRS L1B 亮温） ----------------
def run_typhoon(mod, raw_dir, outdir, api):
    import glob as _g
    fps = sorted(_g.glob(os.path.join(raw_dir, "*.nc")))
    if not fps:
        print("  no viirs file")
        return []
    best = (999.0, None, None, None)
    for fp in fps:
        try:
            bt_i, rgb_i = mod.load_scene(fp)
        except Exception:
            continue
        mn = float(np.nanmin(bt_i))
        if mn < best[0]:
            best = (mn, fp, bt_i, rgb_i)
    if best[1] is None:
        print("  no readable scene")
        return []
    mn, fp, bt, rgb = best
    print("  选用 %s  min BT %.1f K" % (os.path.basename(fp), mn))
    cc = mod.find_cold_center(bt)
    if cc is None:
        cc = (bt.shape[0] // 2, bt.shape[1] // 2)
    cy, cx = cc
    tag = os.path.basename(fp).split(".")[1]
    cand = []
    for dy in (-600, 0, 600):
        for dx in (-600, 0, 600):
            bw, rw, (y0, x0) = mod.crop(bt, rgb, cy, cx, mod.WIN, (dy, dx))
            lab, frac = mod.grade_of(bw)
            if lab is None:
                continue
            cand.append((frac, lab, bw, y0, x0))
    if not cand:
        print("  no valid windows")
        return []
    cand.sort(key=lambda t: t[0], reverse=True)
    rows = []
    for frac, lab, bw, y0, x0 in [cand[0], cand[-1]]:
        png = os.path.join(outdir, "typh_%s_%d_%d.png" % (tag, y0, x0))
        mod.render_bt(bw, png, tag, lab, frac)
        pred, err = predict(api, "hurricane_opt", png)
        rows.append({
            "element": "hurricane_opt", "image": os.path.basename(png), "scene": os.path.basename(fp),
            "truth_cold_frac": round(frac, 3), "truth_label": lab,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "error": err, "match": (pred.get("severity") == lab),
        })
        print("  typh %-30s 冷云占比 %.1f%% 真值 %-10s 模型 %s/%s %s" % (
            rows[-1]["image"], frac * 100, lab, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- snow_fsc（VIIRS NDSI→FSC，流水线） ----------------
def run_snowfsc(mod, outdir, api, day=None):
    import datetime as _dt
    import collections
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from credentials_local import EARTHDATA_USER as EU, EARTHDATA_PASS as EP
    except ImportError:
        from config_local import EARTHDATA_USER as EU, EARTHDATA_PASS as EP
    proxy = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    mod.USER, mod.PASS = EU, EP
    mod.S.proxies.update(proxy)
    mod._TOK["v"] = None
    os.makedirs(mod.L1B, exist_ok=True)
    os.makedirs(mod.GEO, exist_ok=True)
    os.makedirs(mod.IMG, exist_ok=True)
    if day is None:
        day = _dt.date(2024, 2, 20)
    recs = []
    mod.process_region(day, "neimenggu", mod.REGIONS["neimenggu"], True, recs,
                       collections.Counter())
    rows = []
    for r in recs:
        if r.get("product") != "intrinsic_fractional_snow_cover":
            continue
        png = os.path.join(mod.IMG, r["image"])
        if not os.path.exists(png):
            continue
        pred, err = predict(api, "snow_fsc", png)
        rows.append({
            "element": "snow_fsc", "image": r["image"], "day": str(day),
            "truth_mean_fsc": r.get("mean_fsc"), "truth_label": r.get("label"),
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "error": err, "match": (pred.get("severity") == r.get("label")),
        })
        print("  fsc %-30s 平均FSC %.3f 真值 %-10s 模型 %s/%s %s" % (
            r["image"], r.get("mean_fsc") or 0, r.get("label"),
            pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- ground_subsidence / groundwater_overuse（InSAR ENVI 位移） ----------------
def run_insar(mod, products, element, outdir, api):
    import re as _re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    WIN = 128
    STRIDE = 64
    GRADE = [("stable", 0, 5), ("slight_subsidence", 5, 10 if element == "ground_subsidence" else 15),
             ("obvious_subsidence", 10 if element == "ground_subsidence" else 15, 1e9)]

    def grade_of(mm):
        a = abs(mm)
        for n, lo, hi in GRADE:
            if lo <= a < hi:
                return n
        return GRADE[-1][0]

    def render(arr, path, vmax_mm):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
        im = ax.imshow(arr * 1000, cmap="RdBu_r", vmin=-vmax_mm, vmax=vmax_mm)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("LOS displacement", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("mm", fontsize=7)
        fig.tight_layout(); fig.savefig(path, dpi=96); plt.close(fig)
        PILImage.open(path).convert("RGB").resize((384, 384), PILImage.BILINEAR).save(path)

    rows = []
    for pd in products:
        imgf = os.path.join(pd, "deformation_map.data", "displacement_VV.img")
        dimf = os.path.join(pd, "deformation_map.dim")
        if not os.path.isfile(imgf):
            continue
        ncol = nrow = None
        if os.path.isfile(dimf):
            t = open(dimf, encoding="utf-8", errors="replace").read()
            mc = _re.search(r"<NCOLS>(\d+)</NCOLS>", t)
            mr = _re.search(r"<NROWS>(\d+)</NROWS>", t)
            if mc and mr:
                ncol, nrow = int(mc.group(1)), int(mr.group(1))
        if not ncol:
            continue
        arr = np.fromfile(imgf, dtype="<f4").reshape(nrow, ncol)
        arr = np.where(np.abs(arr) < 100, arr, np.nan)
        cand = []
        for y in range(0, max(1, nrow - WIN + 1), STRIDE):
            for x in range(0, max(1, ncol - WIN + 1), STRIDE):
                sub = arr[y:y + WIN, x:x + WIN]
                if sub.shape != (WIN, WIN):
                    continue
                v = sub[~np.isnan(sub)]
                if v.size < WIN * WIN * 0.3:
                    continue
                if element == "ground_subsidence":
                    mm = float(np.nanpercentile(np.abs(v), 90)) * 1000
                else:
                    mm = float(np.nanmedian(sub)) * 1000
                cand.append((mm, y, x))
        if not cand:
            continue
        cand.sort(key=lambda t: abs(t[0]), reverse=True)
        tag = os.path.basename(pd)
        for mm, y, x in [cand[0], cand[-1]]:
            sub = arr[y:y + WIN, x:x + WIN]
            lab = grade_of(mm)
            vmax = max(0.02, float(np.nanmax(np.abs(sub)))) * 1000
            png = os.path.join(outdir, "%s_%s_%d_%d.png" % (element[:3], tag, y, x))
            render(sub, png, vmax)
            pred, err = predict(api, element, png)
            rows.append({
                "element": element, "image": os.path.basename(png), "product": tag,
                "truth_mm": round(mm, 1), "truth_label": lab,
                "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
                "error": err, "match": (pred.get("severity") == lab),
            })
            print("  %s %-30s 位移 %6.1f mm 真值 %-20s 模型 %s/%s %s" % (
                element, rows[-1]["image"], mm, lab, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- coastal_sla / coastal_nt（NASA SSH 格网 SSHA） ----------------
def _load_ssh_stack(raw_dir):
    import glob as _g
    import xarray as xr
    files = sorted(_g.glob(os.path.join(raw_dir, "*.nc")))
    if not files:
        return None, None, None, None
    arrs = []
    lat = lon = None
    for f in files:
        try:
            ds = xr.open_dataset(f)
        except Exception:
            continue
        v = None
        for name in ("ssha", "sla", "ssh"):
            if name in ds:
                v = np.array(ds[name].squeeze(), dtype="float32"); break
        if v is None:
            ds.close(); continue
        if lat is None:
            lat = np.array(ds["latitude"][:], dtype="float64")
            lon = np.array(ds["longitude"][:], dtype="float64")
        v[(v < -10) | (v > 10)] = np.nan
        arrs.append(v)
        ds.close()
    if not arrs:
        return None, None, None, None
    return np.stack(arrs), lat, lon, files[-1]


def run_coastal(mod, which, raw_dir, outdir, api):
    import datetime as _dt
    stack, lat, lon, last = _load_ssh_stack(raw_dir)
    if stack is None:
        print("  no ssh files")
        return []
    D = stack.shape[0]
    stamp = os.path.basename(last)
    m = __import__("re").search(r"(\d{8})", stamp)
    ymd = m.group(1) if m else "latest"

    if which == "coastal_sla":
        li = np.where((lat >= 15) & (lat <= 42))[0]
        lo = np.where((lon >= 105) & (lon <= 135))[0]
        sub = stack[:, li][:, :, lo]
        lower = np.nansum(sub < sub, axis=0)
        valid = np.sum(np.isfinite(sub), axis=0)
        pct = np.where(valid >= 10, 100.0 * lower / np.maximum(valid, 1), np.nan)
        h, w = pct.shape
        lats = lat[li]; lons = lon[lo]
        cand = []
        for y in range(0, h - mod.WIN + 1, mod.STRIDE):
            for x in range(0, w - mod.WIN + 1, mod.STRIDE):
                win = pct[y:y + mod.WIN, x:x + mod.WIN]
                if np.isfinite(win).mean() < 0.5:
                    continue
                cand.append((float(np.nanmean(win)), y, x))
        if not cand:
            print("  no windows")
            return []
        cand.sort(key=lambda t: t[0], reverse=True)
        rows = []
        for mp, y, x in [cand[0], cand[-1]]:
            png = os.path.join(outdir, "csla_%s_%d_%d.png" % (ymd, y, x))
            mod.render(pct[y:y + mod.WIN, x:x + mod.WIN], png, ymd,
                       float(lats[y + mod.WIN // 2]), float(lons[x + mod.WIN // 2]))
            sev = mod.severity_of(mp)
            pred, err = predict(api, "coastal_sla", png)
            rows.append({"element": "coastal_sla", "image": os.path.basename(png),
                         "truth_pct": round(mp, 1), "truth_label": sev,
                         "model_detected": pred.get("detected"),
                         "model_severity": pred.get("severity"), "error": err,
                         "match": (pred.get("severity") == sev)})
            print("  csla %-28s 百分位 %5.1f 真值 %-8s 模型 %s/%s %s" % (
                rows[-1]["image"], mp, sev, pred.get("detected"), pred.get("severity"),
                "OK" if rows[-1]["match"] else "MISS"))
        return rows

    # coastal_nt：30° 分区，按分区最大 SLA 分级
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    last_map = stack[-1]
    cand = []
    for la0 in range(-60, 60, 30):
        for lo0 in range(-180, 180, 30):
            li2 = np.where((lat >= la0) & (lat < la0 + 30))[0]
            lo2 = np.where((lon >= lo0) & (lon < lo0 + 30))[0]
            if not len(li2) or not len(lo2):
                continue
            tile = last_map[np.ix_(li2, lo2)]
            if np.isfinite(tile).mean() < 0.3:
                continue
            mx = float(np.nanmax(np.abs(tile)))
            lab = "high_water" if mx >= 0.30 else ("elevated" if mx >= 0.15 else "normal")
            cand.append((mx, la0, lo0, lab, tile))
    if not cand:
        print("  no tiles")
        return []
    cand.sort(key=lambda t: t[0], reverse=True)
    rows = []
    for mx, la0, lo0, lab, tile in [cand[0], cand[-1]]:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
        im = ax.imshow(tile, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("SLA (m)", fontsize=7)
        fig.tight_layout()
        tag = "%s_%s_%s" % (ymd, ("N" if la0 >= 0 else "S") + str(abs(la0)),
                            ("E" if lo0 >= 0 else "W") + str(abs(lo0)))
        png = os.path.join(outdir, "cnt_%s.png" % tag)
        fig.savefig(png, dpi=96); plt.close(fig)
        PILImage.open(png).convert("RGB").resize((384, 384), PILImage.BILINEAR).save(png)
        pred, err = predict(api, "coastal_nt", png)
        rows.append({"element": "coastal_nt", "image": os.path.basename(png), "tile": tag,
                     "truth_max_sla_m": round(mx, 3), "truth_label": lab,
                     "model_detected": pred.get("detected"),
                     "model_severity": pred.get("severity"), "error": err,
                     "match": (pred.get("severity") == lab)})
        print("  cnt %-30s 最大SLA %.3f m 真值 %-10s 模型 %s/%s %s" % (
            rows[-1]["image"], mx, lab, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows



# ---------------- earthquake_def（ARIA GUNW） ----------------
def run_eqd(mod, gunw_dir, outdir, api):
    import glob as _g
    import h5py
    fps = sorted(_g.glob(os.path.join(gunw_dir, "*.nc")))
    if not fps:
        print("  no gunw")
        return []
    fp = fps[-1]
    with h5py.File(fp, "r") as h:
        g = h["science/grids/data"]
        phase = np.array(g["unwrappedPhase"][:], dtype="float32")
        coh = np.array(g["coherence"][:], dtype="float32")
        cc = np.array(g["connectedComponents"][:], dtype="float32")
    lam = 0.0554657                      # Sentinel-1 C 波段波长 (m)
    disp = -phase * lam / (4 * np.pi) * 1000.0   # mm
    ok = np.isfinite(disp) & (cc >= 0) & (coh >= 0.2)
    disp = np.where(ok, disp, np.nan)
    disp[np.abs(disp) > 3000] = np.nan
    h_, w_ = disp.shape
    WIN = getattr(mod, "WIN", 192)
    cand = []
    for y in range(0, max(1, h_ - WIN + 1), WIN):
        for x in range(0, max(1, w_ - WIN + 1), WIN):
            sub = disp[y:y + WIN, x:x + WIN]
            v = sub[np.isfinite(sub)]
            if v.size < WIN * WIN * 0.3:
                continue
            cand.append((float(np.nanmedian(sub)), y, x))
    if not cand:
        print("  no windows")
        return []
    cand.sort(key=lambda t: abs(t[0]), reverse=True)
    tag = os.path.basename(fp).split("-tops-")[-1][:15]
    rows = []
    for med, y, x in [cand[0], cand[-1]]:
        sub = disp[y:y + WIN, x:x + WIN]
        lab = mod.grade_of(med)
        png = os.path.join(outdir, "eqd_%s_%d_%d.png" % (tag, y, x))
        mod.render(sub, png, "Sentinel-1 InSAR LOS 位移 · Turkey 2023",
                   "%s" % tag)
        pred, err = predict(api, "earthquake_def", png)
        rows.append({
            "element": "earthquake_def", "image": os.path.basename(png), "gunw": os.path.basename(fp),
            "truth_median_mm": round(med, 1), "truth_label": lab,
            "model_detected": pred.get("detected"), "model_severity": pred.get("severity"),
            "error": err, "match": (pred.get("severity") == lab),
        })
        print("  eqd %-30s 中位位移 %7.1f mm 真值 %-14s 模型 %s/%s %s" % (
            rows[-1]["image"], med, lab, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows

# ---------------- volcano ----------------
def run_volcano(mod, modvolc_dir, outdir, api, win_days=20, n_files=4):
    import glob as _g
    import datetime as _dt
    stats = []
    for f in _g.glob(os.path.join(modvolc_dir, "*.txt")):
        try:
            pts = mod.parse_points(f)
        except Exception:
            continue
        if len(pts) < 20:
            continue
        stats.append((max(p[0] for p in pts), len(pts), f))
    stats.sort(reverse=True)          # 最新时间优先
    rows = []
    for t_last, n_all, f in stats[:n_files]:
        vid = os.path.splitext(os.path.basename(f))[0]
        pts = mod.parse_points(f)
        wt = t_last - win_days * 86400
        seg = [p for p in pts if wt <= p[0] <= t_last + 1]
        if not seg:
            continue
        hot = sum(1 for p in seg if p[3] >= mod.EXCESS_THRESH)
        label = "yes" if hot >= mod.MIN_HOT else "no"
        png = os.path.join(outdir, "volc_%s_%d_excess.png" % (vid, int(wt)))
        mod.render(seg, "Volcano %s  excess radiance" % vid, png)
        pred, err = predict(api, "volcano_thermal", png)
        stamp = _dt.datetime.utcfromtimestamp(t_last).strftime("%Y-%m-%d")
        rows.append({
            "element": "volcano_thermal", "image": os.path.basename(png),
            "volcano": vid, "window_end": stamp,
            "truth_label": label, "truth_hot": hot,
            "model_detected": pred.get("detected"),
            "model_severity": pred.get("severity"),
            "error": err, "match": (pred.get("detected") == (label == "yes")),
        })
        print("  volcano %-30s %s 真值 %-4s 模型 %s/%s %s" % (
            rows[-1]["image"], stamp, label, pred.get("detected"), pred.get("severity"),
            "OK" if rows[-1]["match"] else "MISS"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", default="wildfire,volcano_thermal")
    ap.add_argument("--firms-dir", default=os.path.join(ROOT, "data", "raw", "03_thermal_infrared", "firms", "chn2025"))
    ap.add_argument("--modvolc-dir", default=os.path.join(ROOT, "data", "raw", "03_thermal_infrared", "modvolc"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs", "raw_pair1"))
    ap.add_argument("--api", default="http://127.0.0.1:8078")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    want = [x.strip() for x in args.elements.split(",") if x.strip()]
    allrows = []

    if "wildfire" in want:
        mod = load_module("wfv3", os.path.join(ROOT, "src", "preprocess", "wildfire_v3.py"))
        print("== wildfire：2025 年 FIRMS（训练只到 2024）==")
        allrows += run_wildfire(mod, args.firms_dir, args.outdir, args.api)

    if "snow_swe" in want:
        smod = load_module("ssw", os.path.join(ROOT, "src", "preprocess", "snow_swe_render.py"))
        print("== snow_swe：AMSR2 SWE（优先训练未见日期）==")
        allrows += run_snow_swe(smod, os.path.join(ROOT, "data", "raw", "04_passive_microwave", "swe_amsr2"),
                                args.outdir, args.api)

    if "waterlogging_pct" in want:
        wmod = load_module("wlp", os.path.join(ROOT, "src", "preprocess", "waterlogging_pct_render.py"))
        print("== waterlogging_pct：SMAP 土壤湿度百分位 ==")
        allrows += run_waterlogging(wmod, os.path.join(ROOT, "data", "raw", "04_passive_microwave", "smap2"),
                                    args.outdir, args.api)

    for _el, _mod in (("drought_grace", "drought_grace_render.py"),
                       ("groundwater_grace", "groundwater_grace_render.py")):
        if _el in want:
            gmod = load_module("gr_" + _el, os.path.join(ROOT, "src", "preprocess", _mod))
            print("== %s：GRACE-DA 周资料百分位 ==" % _el)
            allrows += run_grace(gmod, os.path.join(ROOT, "data", "raw", "04_passive_microwave", "grace_gw"),
                                 args.outdir, args.api, _el)

    if "wind_gale" in want:
        wmod = load_module("wg", os.path.join(ROOT, "src", "preprocess", "wind_gale_render.py"))
        print("== wind_gale：CMEMS 风场（训练未用日期）==")
        allrows += run_wind_gale(wmod, os.path.join(ROOT, "data", "raw", "07_scatterometer", "wind"),
                                 args.outdir, args.api)

    if "hurricane_rain" in want:
        hmod = load_module("hr", os.path.join(ROOT, "src", "preprocess", "gpm_hurricane_render.py"))
        print("== hurricane_rain：GPM IMERG 日累积 ==")
        allrows += run_hurricane_rain(hmod, os.path.join(ROOT, "data", "raw", "06_radar_precipitation", "gpm"),
                                      args.outdir, args.api)

    if "bloom_chla" in want:
        bmod = load_module("bm", os.path.join(ROOT, "src", "preprocess", "chla_bloom_render_json.py"))
        print("== bloom_chla：ERDDAP 叶绿素（训练未见日期）==")
        allrows += run_bloom(bmod, os.path.join(ROOT, "data", "raw", "01_optical_vnir", "chla_erddap"),
                             args.outdir, args.api)

    if "sea_ice_amsr2" in want:
        imod = load_module("si", os.path.join(ROOT, "src", "preprocess", "seaice_amsr2_render_json.py"))
        print("== sea_ice_amsr2：NSIDC CDR 海冰浓度 ==")
        allrows += run_sea_ice(imod, os.path.join(ROOT, "data", "raw", "04_passive_microwave", "ice_conc"),
                               args.outdir, args.api)

    if "gpm_nt" in want:
        nmod = load_module("gn", os.path.join(ROOT, "src", "preprocess", "gpm_precip_render.py"))
        print("== gpm_nt：GPM 日累积（训练未见日期）==")
        allrows += run_gpm_nt(nmod, os.path.join(ROOT, "data", "raw", "06_radar_precipitation", "gpm"),
                              args.outdir, args.api)

    if "wave" in want:
        vmod = load_module("wv", os.path.join(ROOT, "src", "preprocess", "render_infa_tracks.py"))
        print("== wave：Sentinel-3 沿轨波高 ==")
        allrows += run_wave(vmod, os.path.join(ROOT, "data", "raw", "08_altimetry", "infa"),
                            args.outdir, args.api)

    if "flood_precip_global" in want:
        fpmod = load_module("fp", os.path.join(ROOT, "src", "preprocess", "flood_precip_global_render.py"))
        print("== flood_precip_global：GPM 3 日累积 + GDACS ==")
        allrows += run_flood_precip(fpmod, os.path.join(ROOT, "data", "raw", "06_radar_precipitation", "gpm"),
                                    os.path.join(ROOT, "data", "raw", "14_gdacs_floods.json"),
                                    args.outdir, args.api)

    if "drought_ndvi_clim" in want:
        nmod2 = load_module("nd", os.path.join(ROOT, "src", "preprocess", "drought_ndvi_clim_render.py"))
        print("== drought_ndvi_clim：MODIS NDVI 百分位 ==")
        allrows += run_ndvi(nmod2, os.path.join(ROOT, "data", "raw", "01_optical_vnir", "modis_ndvi"),
                            args.outdir, args.api)

    if "geomagnetic" in want:
        gmod = load_module("gm", os.path.join(ROOT, "src", "preprocess", "geomagnetic_render.py"))
        print("== geomagnetic：WDC Dst 指数 ==")
        allrows += run_geomagnetic(gmod, os.path.join(ROOT, "data", "raw", "12_space_weather", "dst"),
                                   args.outdir, args.api)

    if "haze" in want:
        zmod = load_module("hz", os.path.join(ROOT, "src", "preprocess", "mod08_haze_render.py"))
        print("== haze：MOD08 AOD（OPeNDAP 取变量）==")
        allrows += run_haze(zmod, args.outdir, args.api)

    if "sanddust" in want:
        dmod = load_module("sd", os.path.join(ROOT, "src", "preprocess", "mod08_dust_render.py"))
        print("== sanddust：MOD08 Deep Blue AOD（OPeNDAP 取变量）==")
        allrows += run_sanddust(dmod, args.outdir, args.api)

    if "uvai_dust" in want:
        umod = load_module("uv", os.path.join(ROOT, "src", "preprocess", "omi_uvai_render_json.py"))
        print("== uvai_dust：OMI OMAERUVd ==")
        allrows += run_uvai(umod, os.path.join(ROOT, "data", "raw", "11_atmospheric", "omi_uvai"),
                            args.outdir, args.api)

    if "wildfire_smoke" in want:
        smod2 = load_module("ws", os.path.join(ROOT, "src", "preprocess", "wildfire_smoke_render.py"))
        print("== wildfire_smoke：FIRMS + OMI UVAI ==")
        allrows += run_smoke(smod2, os.path.join(ROOT, "data", "raw", "03_thermal_infrared", "firms", "chn2025"),
                             os.path.join(ROOT, "data", "raw", "11_atmospheric", "omi_uvai"),
                             args.outdir, args.api)

    if "coral_dhw" in want:
        cmod = load_module("cr", os.path.join(ROOT, "src", "preprocess", "coral_dhw_render_json.py"))
        print("== coral_dhw：NOAA CRW DHW（ERDDAP 截取）==")
        allrows += run_coral(cmod, args.outdir, args.api)

    if "volcano_so2" in want:
        omod = load_module("so", os.path.join(ROOT, "src", "preprocess", "omi_so2_render_json.py"))
        print("== volcano_so2：OMI OMSO2e ==")
        allrows += run_so2(omod, os.path.join(ROOT, "data", "raw", "11_atmospheric", "omi_so2"),
                           args.outdir, args.api)

    if "enso" in want:
        emod = load_module("en", os.path.join(ROOT, "scripts", "enso_render_dual.py"))
        print("== enso：OISST 距平（ERDDAP 截取）==")
        allrows += run_enso(emod, args.outdir, args.api)

    if "iod" in want:
        imod2 = load_module("io", os.path.join(ROOT, "src", "preprocess", "iod_render.py"))
        print("== iod：OISST 距平（ERDDAP 印度洋）==")
        allrows += run_iod(imod2, args.outdir, args.api)

    if "marine_heatwave" in want:
        hmod2 = load_module("mh", os.path.join(ROOT, "src", "preprocess", "mhw_render.py"))
        print("== marine_heatwave：OISST 距平 30°窗 ==")
        allrows += run_mhw(hmod2, args.outdir, args.api)

    if "snowstorm" in want:
        tmod = load_module("ss", os.path.join(ROOT, "src", "preprocess", "swe_snow_render.py"))
        print("== snowstorm：AMSR2 SWE → 雪深分级 ==")
        allrows += run_snowstorm(tmod, os.path.join(ROOT, "data", "raw", "04_passive_microwave", "swe_amsr2"),
                                 args.outdir, args.api)

    if "flare" in want:
        fmod = load_module("fl", os.path.join(ROOT, "src", "preprocess", "solar_flare_v3.py"))
        print("== flare：SWPC XRS + SDO AIA193 ==")
        allrows += run_flare(fmod, args.outdir, args.api)

    if "heatwave_lst" in want:
        lmod = load_module("lw", os.path.join(ROOT, "src", "preprocess", "heatwave_lst_render.py"))
        print("== heatwave_lst：MOD11C3 LST 百分位 ==")
        allrows += run_lst(lmod, "heatwave_lst", args.outdir, args.api)

    if "frost" in want:
        fmod2 = load_module("fr", os.path.join(ROOT, "src", "preprocess", "frost_lst_render.py"))
        print("== frost：MOD11C3 LST 百分位 ==")
        allrows += run_lst(fmod2, "frost", args.outdir, args.api)

    if "sea_ice_s2_arctic" in want:
        smod = load_module("s2", os.path.join(ROOT, "src", "preprocess", "sentinel2_seaice_render.py"))
        print("== sea_ice_s2_arctic：Sentinel-2 NDSI ==")
        allrows += run_seaice_s2(smod, os.path.join(ROOT, "data", "raw", "01_optical_vnir", "sentinel2_seaice"),
                                 args.outdir, args.api)

    if "flood_water" in want:
        gmod2 = load_module("gw", os.path.join(ROOT, "src", "preprocess", "grd_water_render.py"))
        print("== flood_water：Sentinel-1 GRD ==")
        allrows += run_grd(gmod2, os.path.join(ROOT, "data", "raw", "05_sar_insar", "grd", "flood_water"),
                           args.outdir, args.api)

    if "green_tide" in want:
        gtmod = load_module("gt", os.path.join(ROOT, "src", "preprocess", "sentinel2_green_tide_render.py"))
        print("== green_tide：Sentinel-2 FAI ==")
        allrows += run_greentide(gtmod, os.path.join(ROOT, "data", "raw", "01_optical_vnir", "sentinel2"),
                                 args.outdir, args.api)

    if "hurricane_opt" in want:
        tmod2 = load_module("ty", os.path.join(ROOT, "src", "preprocess", "viirs_typhoon_render.py"))
        print("== hurricane_opt：VIIRS I05 亮温 ==")
        allrows += run_typhoon(tmod2, os.path.join(ROOT, "data", "raw", "01_optical_vnir", "viirs_typhoon"),
                               args.outdir, args.api)

    if "snow_fsc" in want:
        fscmod = load_module("fsc", os.path.join(ROOT, "src", "preprocess", "snow_fsc_pipeline.py"))
        print("== snow_fsc：VIIRS NDSI→FSC ==")
        allrows += run_snowfsc(fscmod, args.outdir, args.api)

    if "ground_subsidence" in want or "groundwater_overuse" in want:
        prods = [os.path.join(ROOT, "outputs", "henshui_pair01"),
                 os.path.join(ROOT, "outputs", "henshui_6month")]
        print("== InSAR：衡水 ENVI 位移产品 ==")
        if "ground_subsidence" in want:
            allrows += run_insar(None, prods, "ground_subsidence", args.outdir, args.api)
        if "groundwater_overuse" in want:
            allrows += run_insar(None, prods, "groundwater_overuse", args.outdir, args.api)

    if "coastal_sla" in want or "coastal_nt" in want:
        csmod = load_module("cs", os.path.join(ROOT, "src", "preprocess", "coastal_sla_render.py"))
        print("== coastal：NASA SSH 格网（替代 CMEMS） ==")
        sshdir = os.path.join(ROOT, "data", "raw", "08_altimetry", "ssh_ref")
        if "coastal_sla" in want:
            allrows += run_coastal(csmod, "coastal_sla", sshdir, args.outdir, args.api)
        if "coastal_nt" in want:
            allrows += run_coastal(csmod, "coastal_nt", sshdir, args.outdir, args.api)

    if "earthquake_def" in want:
        eqmod = load_module("eq", os.path.join(ROOT, "scripts", "render_eqd.py"))
        print("== earthquake_def：ARIA GUNW ==")
        allrows += run_eqd(eqmod, os.path.join(ROOT, "data", "raw", "05_sar_insar", "aria_gunw"),
                           args.outdir, args.api)

    if "volcano_thermal" in want:
        mod = load_module("vth", os.path.join(ROOT, "src", "preprocess", "volcano_thermal_render.py"))
        print("== volcano_thermal：MODVOLC 最近时间窗 ==")
        allrows += run_volcano(mod, args.modvolc_dir, args.outdir, args.api)

    out = os.path.join(args.outdir, "pair_result.json")
    json.dump(allrows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ok = sum(1 for r in allrows if r["match"])
    print("\n合计 %d/%d 一致 -> %s" % (ok, len(allrows), out))


if __name__ == "__main__":
    main()

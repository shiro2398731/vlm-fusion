"""
IOD（印度洋偶极子）异常渲染 → VLM 训练集生成器

原理（与 thermal_sst_render.py 同源）:
  输入 = 同一份 OISST 原始数据（data/raw/03_thermal_infrared/oisst/）
  每天 d → 取 d-15~d+15 已下载日文件 → 逐格点平均 → 印度洋区域"准月均异常场" → 渲染PNG
  DMI = 西印度洋(50~70°E, 10°S~10°N) 距平均  −  东印度洋(90~110°E, 10°S~0°) 距平均
  标签 = 当月 DMI：≥+0.4 正IOD / ≤-0.4 负IOD / 之间 中性

规格（对齐处理方案验收门槛）:
  - 输出 datasets/03_thermal_infrared/iod/images/YYYYMMDD.png
  - 训练 json: datasets/03_thermal_infrared/iod/train_iod.json
  - 图片统一尺寸 / RGB / 单张 < 262144 px

用法:
  venv\\Scripts\\python.exe src\\preprocess\\iod_render.py --dates 19910115   # 单日验证
  venv\\Scripts\\python.exe src\\preprocess\\iod_render.py                    # 全量
"""
import os
import re
import sys
import glob
import json
import argparse
from datetime import date, timedelta

import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset

SST_DIR = r"C:\vlmdisaster\data\raw\03_thermal_infrared\oisst"
FIELD_DIR = r"C:\vlmdisaster\data\interim\oisst_fields"   # 区域场(.npz)；存在则优先读场
OUT_DIR = r"C:\vlmdisaster\datasets\03_thermal_infrared\iod"

# 印度洋渲染范围（0-360 经度制）
LAT_MIN, LAT_MAX = -30, 30
LON_MIN, LON_MAX = 30, 120
ANOM_LIMIT = 4.0

# DMI 计算的两个"盒子"
WEST = dict(lat=(-10, 10), lon=(50, 70))    # 西印度洋
EAST = dict(lat=(-10, 0),  lon=(90, 110))   # 东印度洋

FIG_SIZE = (7.2, 3.6)
DPI = 100

PROMPT = ("Analyze the sea surface temperature anomaly pattern in the Indian Ocean. "
          "Determine the Indian Ocean Dipole (IOD) phase: is it positive IOD, "
          "negative IOD, or neutral?")


def load_anom_masked(filepath):
    """读取 anom 场。

    若 FIELD_DIR 下当日 .npz 存在，则从场读（印度洋区域已预裁）；
    否则退回读原始 .nc（旧行为）。"""
    import os as _os
    if FIELD_DIR and _os.path.isdir(FIELD_DIR):
        m = re.search(r"(\d{8})", _os.path.basename(filepath))
        if m:
            npz = _os.path.join(FIELD_DIR, m.group(1) + ".npz")
            if _os.path.exists(npz):
                try:
                    z = np.load(npz)
                    return z["indian"], z["lat_i"], z["lon_i"]
                except Exception:
                    pass  # 坏场：退回读 nc（若也被删则抛错，由上层跳过）
    ds = Dataset(filepath)
    anom = np.asarray(ds.variables["anom"][0, 0])
    lat = np.asarray(ds.variables["lat"][:])
    lon = np.asarray(ds.variables["lon"][:])
    ds.close()
    return anom, lat, lon


def month_mean_anom(filepaths, lat, lon):
    """读窗口内所有日文件 → 印度洋区域逐格点均值（mask-aware）"""
    lat_idx = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    lon_idx = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]

    sum_field = None
    count_field = None
    n = 0
    for fp in filepaths:
        anom, _, _ = load_anom_masked(fp)
        sub = anom[np.ix_(lat_idx, lon_idx)]
        data = np.where(np.isfinite(sub) & (sub > -50) & (sub < 50), sub, np.nan)
        if sum_field is None:
            sum_field = np.zeros_like(data)
            count_field = np.zeros_like(data)
        v = ~np.isnan(data)
        sum_field[v] += data[v]
        count_field[v] += 1
        n += 1
    if n == 0:
        return None, None
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_field = np.where(count_field > 0, sum_field / np.maximum(count_field, 1), np.nan)
    return mean_field, (lat[lat_idx], lon[lon_idx])


def dmi_from_field(mean_field, lat, lon):
    """从区域均值场算 DMI = 西盒均值 − 东盒均值"""
    def box_mean(lat_rng, lon_rng):
        li = np.where((lat >= lat_rng[0]) & (lat <= lat_rng[1]))[0]
        lo = np.where((lon >= lon_rng[0]) & (lon <= lon_rng[1]))[0]
        sub = mean_field[np.ix_(li, lo)]
        vals = sub[np.isfinite(sub)]
        return float(np.mean(vals)) if vals.size else None
    w = box_mean(WEST["lat"], WEST["lon"])
    e = box_mean(EAST["lat"], EAST["lon"])
    if w is None or e is None:
        return None
    return w - e


def iod_phase(dmi):
    if dmi is None:
        return None
    if dmi >= 0.4:
        return "positive IOD"
    if dmi <= -0.4:
        return "negative IOD"
    return "neutral"


def render_mean(mean_field, coords, out_png, title=""):
    sub_lat, sub_lon = coords
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    im = ax.pcolormesh(sub_lon, sub_lat, mean_field, cmap="RdBu_r",
                       vmin=-ANOM_LIMIT, vmax=ANOM_LIMIT, shading="auto")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude")
    ax.set_title(title or "Monthly SST Anomaly (Indian Ocean)")
    # 画出 DMI 两个盒子
    for box, col in [(WEST, "k"), (EAST, "k")]:
        x0, x1 = box["lon"]; y0, y1 = box["lat"]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=col, lw=0.8, ls="--")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("SST Anomaly (°C)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", format="png")
    plt.close(fig)
    from PIL import Image
    im2 = Image.open(out_png)
    if im2.mode != "RGB":
        im2.convert("RGB").save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=31, help="滚动窗口天数（默认31≈月均）")
    ap.add_argument("--dates", nargs="*", default=None, help="指定日期 YYYYMMDD")
    ap.add_argument("--limit", type=int, default=0, help="本次最多渲多少张（0=不限；建议分批如 400）")
    ap.add_argument("--resume", action="store_true", help="跳过 images 里已存在的日期，断点续渲")
    ap.add_argument("--per-class-stop", type=int, default=0,
                    help="每类最多渲多少张就停（0=不限；用于均衡三类，如 150）")
    args = ap.parse_args()

    half = args.window // 2
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)

    all_files = glob.glob(os.path.join(SST_DIR, "oisst-avhrr-v02r01.*.nc"))
    if not all_files and FIELD_DIR and os.path.isdir(FIELD_DIR):
        all_files = glob.glob(os.path.join(FIELD_DIR, "*.npz"))
    by_date = {}
    for f in all_files:
        m = re.search(r"(\d{8})", os.path.basename(f))
        if not m:
            continue
        ymd = m.group(1)
        if ymd not in by_date or "_preliminary" not in os.path.basename(f):
            by_date[ymd] = f
    all_dates = sorted(by_date.keys())
    print(f"可用日文件: {len(all_dates)} 个")
    print(f"滚动窗口: {args.window} 天（±{half}）")

    targets = []
    for ymd in all_dates:
        d = date.fromisoformat(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}")
        win = [ (d + timedelta(days=k)).strftime("%Y%m%d") for k in range(-half, half + 1) ]
        win = [w for w in win if w in by_date]
        if len(win) >= max(3, args.window * 2 // 3):
            targets.append((ymd, win))
    if args.dates:
        targets = [t for t in targets if t[0] in args.dates]
    print(f"可渲染日期: {len(targets)} 个\n")

    records = []
    # 已有图片（resume 时跳过 + 从已有 json 续记）
    img_dir = os.path.join(OUT_DIR, "images")
    existing_dates = set()
    if args.resume and os.path.isdir(img_dir):
        existing_dates = {f[:-4] for f in os.listdir(img_dir) if f.endswith(".png")}
    # 载入已有 json（若有），保留历史记录
    out_json = os.path.join(OUT_DIR, "train_iod.json")
    if os.path.exists(out_json):
        try:
            records = json.load(open(out_json, encoding="utf-8"))
        except Exception:
            records = []
    done_dates = set()
    from collections import Counter
    dist = Counter()
    for r in records:
        for d0 in r.get("images", []):
            done_dates.add(os.path.basename(d0)[:-4])
        for k in ["positive IOD", "negative IOD", "neutral"]:
            if k in r["conversations"][1]["value"]:
                dist[k] += 1
                break

    n_new = 0
    for ymd, win in targets:
        if args.resume and ymd in existing_dates:
            continue
        if ymd in done_dates:
            continue   # 已有记录：跳过，避免重跑时重复写入
        if args.limit and n_new >= args.limit:
            print(f"\n  已达本次限量 {args.limit} 张，停止（用 --resume 继续）")
            break
        fps = [by_date[d] for d in win]
        try:
            _, lat0, lon0 = load_anom_masked(fps[0])
            mean_field, coords = month_mean_anom(fps, lat0, lon0)
        except Exception:
            continue  # 窗口内有坏场：跳过该日期
        if mean_field is None:
            continue
        dmi = dmi_from_field(mean_field, coords[0], coords[1])
        phase = iod_phase(dmi)
        if phase is None:
            continue
        if args.per_class_stop and dist[phase] >= args.per_class_stop:
            continue

        png = os.path.join(OUT_DIR, "images", f"{ymd}.png")
        render_mean(mean_field, coords, png,
                    title=f"{ymd} Rolling{args.window}d SST Anomaly (IOD: {phase})")
        answer = f"The Indian Ocean SST anomaly indicates a {phase} phase (DMI={dmi:+.2f})."
        records.append({
            "conversations": [
                {"from": "human", "value": f"<image>{PROMPT}"},
                {"from": "gpt", "value": answer},
            ],
            "images": [png],
        })
        dist[phase] += 1
        n_new += 1
        # 每 50 张就落一次盘，中断不丢
        if n_new % 50 == 0:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"  [{len(records)} 条，本批 {n_new}] {ymd} → {phase} (DMI={dmi:+.2f})  已落盘", flush=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n累计: {len(records)} 条样本 → {out_json}")
    print(f"标签分布: {dict(dist)}")


if __name__ == "__main__":
    main()

"""
海温异常渲染 → VLM 训练集生成器（v4 滚动窗口版）

原理（季节平均 + 样本量兼顾）:
  单日海温场 = ENSO信号 + 天气噪声
  滚动窗口平均（±15天≈31天窗口）≈ 月均 → 去噪声又保样本量

流程:
  每天 d → 取 d-15~d+15 已下载日文件 → 逐格点平均(mask-aware)
         → "准月均异常场" → 渲染PNG
  标签 = nina34 当月距平 (≥0.5=El Nino / ≤-0.5=La Nina / 之间=neutral)

用法:
  python src/preprocess/thermal_sst_render.py                  # 全部日期(滚动窗口)
  python src/preprocess/thermal_sst_render.py --window 31      # 自定义窗口(天)
  python src/preprocess/thermal_sst_render.py --dates 20251115 # 指定日期

输出:
  datasets/03_thermal_infrared/enso/images/YYYYMMDD.png
  datasets/03_thermal_infrared/enso/train_thermal_sst.json
"""
import os
import re
import sys
import glob
import json
import argparse
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from netCDF4 import Dataset

# 路径已于 2026-09-30 更新到新的目录规范（data\raw\<源号>_<源名>\）。
# 旧路径 data\thermal\... 已不存在，会导致"可用日文件: 0 个"。
SST_DIR = r"C:\vlmdisaster\data\raw\03_thermal_infrared\oisst"
FIELD_DIR = r"C:\vlmdisaster\data\interim\oisst_fields"   # 区域场(.npz)；存在则优先读场
OUT_DIR = r"C:\vlmdisaster\datasets\03_thermal_infrared\enso"
ENSO_FILE = r"C:\vlmdisaster\data\raw\03_thermal_infrared\enso_index\nina34.anom.data"

# 热带太平洋渲染范围（ENSO 主场）
LAT_MIN, LAT_MAX = -30, 30
LON_MIN, LON_MAX = 120, 300  # 120E ~ 60W（0-360制）
ANOM_LIMIT = 4.0

# 图像尺寸控制（< 262144 像素）
FIG_SIZE = (7.2, 3.6)
DPI = 100

PROMPT = ("Analyze the sea surface temperature anomaly pattern in the tropical Pacific Ocean. "
          "Determine the ENSO phase: is it El Nino, La Nina, or neutral?")


def load_nina34():
    """加载 nina34 月距平: {year: [12个月]}"""
    anom = {}
    with open(ENSO_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 13 and parts[0].isdigit():
                anom[int(parts[0])] = [float(x) for x in parts[1:]]
    return anom


def ensn_phase(nina34, year, month):
    if year not in nina34 or month > len(nina34[year]):
        return None
    v = nina34[year][month - 1]
    if v < -50:
        return None
    if v >= 0.5:
        return "El Nino"
    if v <= -0.5:
        return "La Nina"
    return "neutral"


def load_anom_masked(filepath):
    """读取 anom 场（masked array，netCDF4 自动 scale_factor+mask）

    若提供了区域场目录（FIELD_DIR）且当日 .npz 存在，则从场读（区域已预裁）；
    否则退回读原始 .nc（旧行为）。"""
    import os as _os
    if FIELD_DIR and _os.path.isdir(FIELD_DIR):
        m = re.search(r"(\d{8})", _os.path.basename(filepath))
        if m:
            npz = _os.path.join(FIELD_DIR, m.group(1) + ".npz")
            if _os.path.exists(npz):
                try:
                    z = np.load(npz)
                    anom = z["pacific"]            # 太平洋区域场（ENSO 主场）
                    lat = z["lat_i"]
                    lon = z["lon_p"]
                    return anom, lat, lon
                except Exception:
                    pass  # 坏场：退回读 nc
    ds = Dataset(filepath)
    anom = ds.variables["anom"][0, 0]
    lat = np.asarray(ds.variables["lat"][:])
    lon = np.asarray(ds.variables["lon"][:])
    ds.close()
    return anom, lat, lon


def month_mean_anom(filepaths, lat, lon):
    """读该月所有日文件 → 逐格点月均异常场（mask-aware）"""
    lat_idx = np.where((lat >= LAT_MIN) & (lat <= LAT_MAX))[0]
    lon_idx = np.where((lon >= LON_MIN) & (lon <= LON_MAX))[0]

    sum_field = None
    count_field = None
    n_valid_files = 0

    for fp in filepaths:
        anom, _, _ = load_anom_masked(fp)
        sub = anom[np.ix_(lat_idx, lon_idx)]
        data = np.ma.filled(sub, np.nan).astype(float)
        data = np.where(np.isfinite(data) & (data > -50) & (data < 50), data, np.nan)

        if sum_field is None:
            sum_field = np.zeros_like(data)
            count_field = np.zeros_like(data)
        valid = ~np.isnan(data)
        sum_field[valid] += data[valid]
        count_field[valid] += 1
        n_valid_files += 1

    if n_valid_files == 0:
        return None, None

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_field = np.where(count_field > 0, sum_field / np.maximum(count_field, 1), np.nan)

    sub_lat = lat[lat_idx]
    sub_lon = lon[lon_idx]
    return mean_field, (sub_lat, sub_lon)


def render_mean(mean_field, coords, out_png, title=""):
    """渲染月均异常场"""
    sub_lat, sub_lon = coords
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    im = ax.pcolormesh(sub_lon, sub_lat, mean_field, cmap="RdBu_r",
                       vmin=-ANOM_LIMIT, vmax=ANOM_LIMIT, shading="auto")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude")
    ax.set_title(title or "Monthly SST Anomaly (OISST)")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(190, color="gray", lw=0.5, ls="--")
    ax.axvline(240, color="gray", lw=0.5, ls="--")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("SST Anomaly (°C)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", format="png")
    plt.close(fig)

    from PIL import Image
    im2 = Image.open(out_png)
    if im2.mode != "RGB":
        im2 = im2.convert("RGB")
        im2.save(out_png)


def main():
    from datetime import date, timedelta

    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=31,
                        help="滚动窗口天数（默认31≈月均，可设7/15/31）")
    parser.add_argument("--dates", nargs="*", default=None,
                        help="指定日期 YYYYMMDD（默认全部）")
    args = parser.parse_args()

    half = args.window // 2
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    nina34 = load_nina34()

    # 收集所有日文件（日期去重：正式版优先）；若原始 nc 已删而场存在，则扫场目录
    all_files = glob.glob(os.path.join(SST_DIR, "oisst-avhrr-v02r01.*.nc"))
    if not all_files and FIELD_DIR and os.path.isdir(FIELD_DIR):
        all_files = glob.glob(os.path.join(FIELD_DIR, "*.npz"))
    by_date = {}
    for f in all_files:
        name = os.path.basename(f)
        m = re.search(r"(\d{8})", name)
        if not m:
            continue
        ymd = m.group(1)
        if ymd not in by_date or "_preliminary" not in name:
            by_date[ymd] = f

    all_dates = sorted(by_date.keys())
    print(f"可用日文件: {len(all_dates)} 个")
    print(f"滚动窗口: {args.window}天（±{half}天）≈ 月均效果")
    print()

    # 目标日期：窗口内至少有 2/3 日文件的日期
    targets = []
    for i, ymd in enumerate(all_dates):
        d = date.fromisoformat(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}")
        # 窗口内的日期集合
        win_dates = []
        for delta in range(-half, half + 1):
            wd = d + timedelta(days=delta)
            wkey = wd.strftime("%Y%m%d")
            if wkey in by_date:
                win_dates.append(wkey)
        min_need = max(3, args.window * 2 // 3)
        if len(win_dates) >= min_need:
            targets.append((ymd, win_dates))

    if args.dates:
        targets = [t for t in targets if t[0] in args.dates]

    print(f"可渲染日期: {len(targets)} 个（窗口数据充足）")
    records = []

    for ymd, win_dates in targets:
        year, mon = int(ymd[:4]), int(ymd[4:6])
        phase = ensn_phase(nina34, year, mon)
        if phase is None:
            continue

        filepaths = [by_date[d] for d in win_dates]
        try:
            anom0, lat0, lon0 = load_anom_masked(filepaths[0])
            mean_field, coords = month_mean_anom(filepaths, lat0, lon0)
        except Exception:
            continue  # 窗口内有坏场：跳过该日期
        if mean_field is None:
            continue

        valid_frac = np.isfinite(mean_field).mean()
        png = os.path.join(OUT_DIR, "images", f"{ymd}.png")
        render_mean(mean_field, coords, png,
                    title=f"{ymd} Rolling{args.window}d SST Anomaly ({phase})")

        answer = f"The tropical Pacific SST anomaly indicates a {phase} phase."
        records.append({
            "conversations": [
                {"from": "human", "value": f"<image>{PROMPT}"},
                {"from": "gpt", "value": answer},
            ],
            "images": [png],
        })
        if len(records) % 30 == 1 or len(records) == len(targets):
            print(f"  [{len(records)}/{len(targets)}] {ymd} → {phase} ({valid_frac*100:.0f}%)")

    out_json = os.path.join(OUT_DIR, "train_thermal_sst.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n已生成: {len(records)} 条样本 → {out_json}")

    from collections import Counter
    phases = Counter()
    for r in records:
        for k in ["El Nino", "La Nina", "neutral"]:
            if k in r["conversations"][1]["value"]:
                phases[k] += 1
                break
    print(f"标签分布: {dict(phases)}")


if __name__ == "__main__":
    main()

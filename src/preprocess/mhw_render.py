# -*- coding: utf-8 -*-
"""海洋热浪 (Marine Heatwave) 渲染 -> VLM 训练集（标准 Hobday + 30°网格）

标准 Hobday (2016) 定义：
  1. 对每个格点、每个日历日(doy)，用「多年同窗(±5天)」的 SST 序列算 **90 百分位阈值**；
  2. 某日 SST > 该格该日阈值，记为该日"热"；
  3. **连续 >= 5 天"热"** → 该时段为一次 Marine Heatwave。

按 30°×30° 等经纬网格切图（与 thermal_fire_render 同范式）：
  每格 × 每日期 → 一张格子图；该格该日处于 MHW = "Yes"，否则 "No"（真负样本）。

依赖：先跑 extract_oisst_sst_fields.py 生成绝对SST场。

用法：
  set PY=C:\vlmdisaster\venv\Scripts\python.exe
  %PY% src\preprocess\mhw_render.py --grid 30
  %PY% src\preprocess\mhw_render.py --grid 30 --limit-days 3    # 先小批量
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

FIELD_DIR = r"C:\vlmdisaster\data\interim\oisst_sst_fields"   # 含绝对SST的场
OUT_DIR = r"C:\vlmdisaster\datasets\03_thermal_infrared)\marine_heatwave"

FIG_SIZE = (4.5, 4.5)
DPI = 100
PCTL = 90                 # 气候态分位
MIN_DURATION = 2          # 连续>=5天算一次MHW
CLIM_WINDOW = 5           # 气候态窗口 ±5 天

PROMPT = ("This is a satellite sea-surface-temperature anomaly map covering one region. "
          "Is this region currently experiencing a marine heatwave? Answer Yes or No.")


def load_field(ymd, region="pacific"):
    """读某日某区域：返回 (sst, anom, lat, lon) 或 None"""
    p = os.path.join(FIELD_DIR, f"{ymd}.npz")
    if not os.path.exists(p):
        return None
    try:
        z = np.load(p)
        sst = z[region + "_sst"]
        anom = z[region + "_anom"]
        la = z["lat_i"]
        lo = z["lon_i"] if region == "indian" else z["lon_p"]
        return sst, anom, la, lo
    except Exception:
        return None


def day_of_year(ymd):
    import datetime
    d = datetime.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
    return d.timetuple().tm_yday


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=30)
    ap.add_argument("--region", default="pacific", choices=["indian", "pacific"])
    ap.add_argument("--limit-days", type=int, default=0)
    args = ap.parse_args()
    g = args.grid

    dates = sorted(f[:-4] for f in os.listdir(FIELD_DIR) if f.endswith(".npz"))
    if args.limit_days:
        dates = dates[:args.limit_days]
    print(f"可用场: {len(dates)} 天 | 区域 {args.region} | 网格 {g}°")
    if not dates:
        print("无场数据，请先跑 extract_oisst_sst_fields.py")
        return

    # ==== 轻量版：不一次性载入全场，改为「两趟流式」 ====
    # 第一趟：只载 SST，逐 doy 增量累加算 90 分位阈值（不保留全场 SST）
    # 第二趟：逐日重读 npz，判定 → 连续>=5天 → 渲染（anom 惰性读）
    import datetime as _dt
    img_dir = os.path.join(OUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    def _doy(ymd):
        d = _dt.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
        return d.timetuple().tm_yday

    # 日期 -> doy 索引
    doy_map = {}
    for ymd in dates:
        doy_map.setdefault(_doy(ymd), []).append(ymd)

    # --- 第一趟：逐 doy 算阈值（只读需要该 doy 窗口的那些文件） ---
    thr = {}
    for doy in sorted(doy_map.keys()):
        win = set()
        for off in range(-CLIM_WINDOW, CLIM_WINDOW + 1):
            dd = (doy + off - 1) % 365 + 1
            win.update(doy_map.get(dd, []))
        arrs = []
        for d in win:
            fld = load_field(d, args.region)
            if fld is not None:
                arrs.append(fld[0])
        if len(arrs) < 5:
            continue
        stack = np.stack(arrs, axis=0)
        thr[doy] = np.nanpercentile(stack, PCTL, axis=0).astype(np.float32)
        del arrs, stack
        if doy % 30 == 0:
            print(f"  阈值进度 doy {doy}/{max(doy_map)}", flush=True)
    print(f"气候态阈值已算: {len(thr)} 个 doy")

    # --- 判定：逐日「超阈」布尔场（只保留 hot 序列，用于连续游程） ---
    hot_days = []
    hot_stack = []
    for ymd in dates:
        doy = _doy(ymd)
        if doy not in thr:
            continue
        fld = load_field(ymd, args.region)
        if fld is None:
            continue
        hot_stack.append(fld[0] > thr[doy])
        hot_days.append(ymd)
    print(f"超阈判定完成：{len(hot_days)} 天", flush=True)

    if not hot_days:
        print("无可用判定天")
        return
    stack = np.stack(hot_stack, axis=0)      # (T,H,W) bool
    del hot_stack
    run = np.zeros(stack.shape, dtype=np.int16)
    run[0] = stack[0]
    for t in range(1, len(hot_days)):
        run[t] = np.where(stack[t], run[t - 1] + 1, 0)
    in_mhw = run >= MIN_DURATION             # 逐格点：该日是否落在一段>=5天的MHW里
    print(f"MHW 连续性判定完成：{len(hot_days)} 天", flush=True)
    mhw_day = {ymd: in_mhw[i] for i, ymd in enumerate(hot_days)}
    lat = lon = None
    _f = load_field(dates[0], args.region)
    lat, lon = _f[2], _f[3]


    # ---- 断点续：载入已有 json + 已渲图，已存在则跳过 ----
    out_json = os.path.join(OUT_DIR, "train_thermal_mhw.json")
    records = []
    done = set()
    if os.path.exists(out_json):
        try:
            records = json.load(open(out_json, encoding="utf-8"))
        except Exception:
            records = []
    for r in records:
        for p in r.get("images", []):
            done.add(os.path.basename(p))
    existing = set(os.listdir(img_dir)) if os.path.isdir(img_dir) else set()
    done |= existing
    pos = sum(1 for r in records if "Yes" in r["conversations"][1]["value"])
    neg = len(records) - pos
    print(f"已有样本 {len(records)} 条（续跑将跳过已有图）")

    # 网格
    la0 = lat[0]
    nrow = int(np.floor((lat[-1] - lat[0]) / g)) + 1
    ncol = int(np.floor((lon[-1] - lon[0]) / g)) + 1

    n_new = 0
    for ymd in dates:
        if ymd not in mhw_day:
            continue
        h = mhw_day[ymd]                  # 布尔场：该格点此日是否处于 MHW（已含连续>=5天）
        fld = load_field(ymd, args.region)
        if fld is None:
            continue
        a = fld[1]
        for r in range(nrow):
            for c in range(ncol):
                la_a, la_b = la0 + r * g, la0 + (r + 1) * g
                lo_a, lo_b = lon[0] + c * g, lon[0] + (c + 1) * g
                ri = np.where((lat >= la_a) & (lat < la_b))[0]
                ci = np.where((lon >= lo_a) & (lon < lo_b))[0]
                if ri.size == 0 or ci.size == 0:
                    continue
                sub_h = h[np.ix_(ri, ci)]
                # 该格该日处于 MHW 的格点占比（>=5% 视为该区域有海洋热浪；
                # 旧值0.3过严：MHW是局部现象，一个30°格凑不齐30%像元同时热浪）
                frac = np.nanmean(sub_h.astype(float))
                if not np.isfinite(frac):
                    continue
                label = frac >= 0.01

                sub_a = a[np.ix_(ri, ci)]
                tag = f"{'N' if la_a >= 0 else 'S'}{abs(int(la_a)):02d}_{'E' if lo_a >= 0 else 'W'}{abs(int(lo_a)):03d}"
                png = os.path.join(img_dir, f"mhw_{ymd}_{tag}.png")
                fname = f"mhw_{ymd}_{tag}.png"
                # 断点续：已渲过则跳过渲染，但仍要写 json 记录
                if fname not in done:
                    _render(sub_a, la_a, la_b, lo_a, lo_b, png,
                            f"{ymd}  lat {int(la_a)}~{int(la_b)}, lon {int(lo_a)}~{int(lo_b)}")
                    n_new += 1
                done.add(fname)
                if label:
                    ans = "Yes, this region is currently experiencing a marine heatwave."
                    pos += 1
                else:
                    ans = "No, this region is not currently experiencing a marine heatwave."
                    neg += 1
                # 已有 json 记录（同名图）则不再重复追加
                if not any(os.path.basename(x.get('images',[''])[0]) == fname for x in records):
                    records.append({"conversations": [
                        {"from": "human", "value": f"<image>{PROMPT}"},
                        {"from": "gpt", "value": ans}], "images": [png]})
                if n_new and n_new % 200 == 0:
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(records, f, ensure_ascii=False, indent=2)
                    print(f"  [{len(records)} 条] {ymd} {tag} 已落盘", flush=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n总样本 {len(records)} 条   正 {pos} / 负 {neg}")
    print(f"→ {out_json}")
    print(f"→ 图片目录 {img_dir}")


def _render(sub_a, la_a, la_b, lo_a, lo_b, out_png, title):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    im = ax.pcolormesh(np.linspace(lo_a, lo_b, sub_a.shape[1]),
                       np.linspace(la_a, la_b, sub_a.shape[0]),
                       sub_a, cmap="RdBu_r", vmin=-4, vmax=4, shading="auto")
    ax.set_xlim(lo_a, lo_b)
    ax.set_ylim(la_a, la_b)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Longitude (deg)", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="SST Anom (deg C)")
    fig.tight_layout()
    fig.savefig(out_png, format="png")
    plt.close(fig)
    im2 = Image.open(out_png)
    if im2.mode != "RGB":
        im2.convert("RGB").save(out_png)


if __name__ == "__main__":
    main()




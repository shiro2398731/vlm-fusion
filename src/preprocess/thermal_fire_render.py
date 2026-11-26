"""
FIRMS 火点渲染 → VLM 训练集（v3 · 区域网格版）

为什么改：v2 用「全球火点图 + 有/无野火」二分类 —— 全球图天天都有火，
所以实测正负 102:6，正类恒真、负类不存在，任务退化。

v3 改成**按 30°×30° 等经纬网格切图**：
  - 每个格子单独成图 → 某格当天有显著火点 = Yes，没有 = No（**真负样本出现了**）
  - 一天 × 多个格子 → 样本量 = 天数 × 格数
  - **不需要逆地理编码**（纯经纬度网格，零依赖、可复现）

「陆地格」怎么定（重要，无外部依赖）：先扫一遍全部输入，
**凡在整段时间里出现过火点的格子 → 视为陆地格**。从不出火的格子（海洋/冰盖）直接排除。
  ⚠️ 局限：会把"这 108 天里从没起火的陆地"也排除掉。这属于**偏保守**的选择，
     换来的是**零外部数据依赖**；后续若要更准，再引入真正的陆地掩膜。

用法：
  python src/preprocess/thermal_fire_render.py --grid 30 --limit-days 3   # 先小批量
  python src/preprocess/thermal_fire_render.py --grid 30                  # 全量
"""
import os
import sys
import csv
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

FIRE_DIR = r"C:\vlmdisaster\data\raw\03_thermal_infrared\firms\history"
OUT_DIR = r"C:\vlmdisaster\datasets\03_thermal_infrared\wildfire"

FIG_SIZE = (4.5, 4.5)
DPI = 100
FRP_MIN = 5.0
MIN_BYTES = 1024          # 小于此的 csv 视为空壳，跳过

PROMPT = ("This is a satellite fire-hotspot map covering one region. "
          "Are there significant active wildfires in this region? Answer Yes or No.")


def load_points(path):
    pts = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                pts.append((float(r["latitude"]), float(r["longitude"]),
                            float(r.get("frp", 0) or 0)))
            except (ValueError, KeyError, TypeError):
                continue
    return pts


def cell_of(lat, lon, g):
    la = int(np.floor((lat + 90) / g)) * g - 90
    lo = int(np.floor((lon + 180) / g)) * g - 180
    return la, lo


def render(pts, la, lo, g, out_png, title):
    """特色场：FRP 场（火点 FRP 累加到网格，替代旧散点图）"""
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    if pts:
        step = 0.25
        nx = max(1, int(round(g / step)))
        ny = max(1, int(round(g / step)))
        grid = np.zeros((ny, nx), dtype=float)
        for lat, lon, frp in pts:
            ix = int((lon - lo) / step)
            iy = int((lat - la) / step)
            if 0 <= ix < nx and 0 <= iy < ny:
                grid[iy, ix] += max(frp, 0.0)
        im = ax.imshow(grid, origin="lower", extent=[lo, lo + g, la, la + g],
                       cmap="YlOrRd", vmin=0, vmax=max(grid.max(), 1.0),
                       interpolation="bilinear", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("FRP (MW)", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No fire hotspot", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=12)
    ax.set_xlim(lo, lo + g)
    ax.set_ylim(la, la + g)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Longitude (deg)", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="lightgray", lw=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(out_png, format="png")     # 不用 bbox_inches=tight → 保证尺寸统一
    plt.close(fig)
    im = Image.open(out_png)
    if im.mode != "RGB":
        im.convert("RGB").save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=30, help="网格边长（度），默认 30")
    ap.add_argument("--threshold", type=int, default=3,
                    help="格子内显著火点(FRP>=5MW)数 >= 此值 → Yes，默认 3")
    ap.add_argument("--limit-days", type=int, default=0, help="只处理前 N 天（0=全部）")
    args = ap.parse_args()
    g = args.grid

    files = [f for f in sorted(glob.glob(os.path.join(FIRE_DIR, "*.csv")))
             if os.path.getsize(f) >= MIN_BYTES]
    if args.limit_days:
        files = files[:args.limit_days]
    print(f"输入 csv: {len(files)} 个 | 网格 {g}° | Yes 阈值 {args.threshold} 个显著火点")

    # 第一遍：找出「出现过火点的格子」= 陆地格
    day_points = []
    land = set()
    for f in files:
        pts = load_points(f)
        day_points.append((os.path.basename(f)[:-4], pts))
        for lat, lon, frp in pts:
            if frp >= FRP_MIN:
                land.add(cell_of(lat, lon, g))
    print(f"陆地格（整段出现过显著火点）: {len(land)} 个")

    img_dir = os.path.join(OUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    records, pos, neg = [], 0, 0
    for name, pts in day_points:
        for (la, lo) in sorted(land):
            sub = [p for p in pts if cell_of(p[0], p[1], g) == (la, lo)]
            sig = sum(1 for p in sub if p[2] >= FRP_MIN)
            label = sig >= args.threshold
            tag = f"{'N' if la >= 0 else 'S'}{abs(la):02d}_{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
            png = os.path.join(img_dir, f"fire_{name}_{tag}.png")
            render(sub, la, lo, g, png, f"{name}  lat {la}~{la+g}, lon {lo}~{lo+g}")
            if label:
                ans = (f"Yes, there are significant active wildfires in this region "
                       f"({sig} hotspots with FRP >= {FRP_MIN:.0f} MW).")
                pos += 1
            else:
                ans = (f"No significant active wildfire in this region "
                       f"(only {sig} hotspot(s) with FRP >= {FRP_MIN:.0f} MW).")
                neg += 1
            records.append({"conversations": [
                {"from": "human", "value": f"<image>{PROMPT}"},
                {"from": "gpt", "value": ans}], "images": [png]})

    out_json = os.path.join(OUT_DIR, "train_thermal_fire.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n总样本 {len(records)} 条   正 {pos} / 负 {neg}")
    print(f"→ {out_json}")
    print(f"→ 图片目录 {img_dir}（本版命名 fire_<日期>_<格>；旧的 fire_<日期>.png 已不被引用）")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""
Sentinel-1 GRD 水体渲染 → VLM 训练集（服务 SAR 的"洪水水体/涝灾渍涝"元素）。

原理：水面对雷达波做镜面反射，后向散射极低 → 影像上呈暗区。
      所以按 VV 波段的低值像元占比判"有没有水体"。

处理：
  1. 从 GRD zip 内读 measurement 的 VV tiff（16bit 强度，约 25206×23642）
  2. 每景先按分位数做归一化（消除不同景的绝对强度差异）
  3. 切 512×512 窗口，按低值像元占比判标签：
       water ：暗区占比 > HI（大面积水）
       land  ：暗区占比 < LO（基本陆地）
       中间区间丢弃（避免边界模糊样本）
  4. 渲染成 384×384 RGB 灰度图

输入：data\raw\05_sar_insar\grd\<灾害>\*.zip
输出：datasets\05_sar_insar\<灾害>\images\*.png + json (+ val)

用法：
  python src/preprocess/grd_water_render.py --disaster flood_water --limit 2   # 先试 2 景
  python src/preprocess/grd_water_render.py --disaster flood_water
"""
import os
import sys
import io
import json
import glob
import zipfile
import argparse
import numpy as np
from PIL import Image as PILImage

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Image_MAX = None
PILImage.MAX_IMAGE_PIXELS = None

GRD_ROOT = r"C:\vlmdisaster\data\raw\05_sar_insar\grd"
OUT_ROOT = r"C:\vlmdisaster\datasets\05_sar_insar"

WIN = 512
STRIDE = 768
OUT_SIZE = 384
VAL_PER_CLASS = 30
MAX_PER_FILE = 40
DARK_Q = 0.25        # 低值阈值取全图 25 分位
HI = 0.35            # 暗区占比 > HI → water
LO = 0.10            # 暗区占比 < LO → land

PROMPT = ("This is a Sentinel-1 SAR backscatter image (VV polarization, C-band ~5.6 cm) "
          "over a region. Dark areas = specular reflection from smooth water surface "
          "(low backscatter); bright areas = rough land (high backscatter). "
          "Is there a significant water body (flood/inundation) in this region? "
          "Answer Yes or No, and report the dark-area fraction.")

# 中文字体（与 render_eqd.py 同源）
def _zh_font(size):
    from matplotlib import font_manager
    import os as _os
    cands = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for p in cands:
        if _os.path.exists(p):
            return font_manager.FontProperties(fname=p, size=size)
    return font_manager.FontProperties(size=size)


def read_vv(zip_path):
    z = zipfile.ZipFile(zip_path)
    cand = [n for n in z.namelist() if "measurement" in n and "vv" in n.lower()
            and n.lower().endswith(".tiff")]
    if not cand:
        return None
    im = PILImage.open(io.BytesIO(z.read(cand[0])))
    return im


def render(arr, path, title, dark_frac, lab):
    """带物理量的水体图: 主图(相对后向散射) + colorbar(带阈值线) + 图例 + 常量 + 物理链"""
    import numpy as np
    fig = plt.figure(figsize=(4.48, 5.90), dpi=100)
    ax = fig.add_axes([0.04, 0.30, 0.78, 0.60])
    im = ax.imshow(arr, cmap="Blues_r", vmin=0, vmax=255)   # 越蓝越亮=越弱(水)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7.5, fontproperties=_zh_font(7.5))

    # colorbar: 标 “暗区阈值” 线
    cax = fig.add_axes([0.84, 0.30, 0.035, 0.60])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("相对后向散射强度 (0-255)", fontsize=6.5, fontproperties=_zh_font(6.5))
    cb.ax.tick_params(labelsize=6)
    cb.ax.axhline(60, color="red", lw=1.2)          # 暗区阈值 60
    cb.ax.text(1.9, 60, "暗区阈值60", color="red", fontsize=5.5,
               transform=cb.ax.get_yaxis_transform(), va="center", fontproperties=_zh_font(5.5))

    # 底部信息带
    legend = "图例: 暗(蓝)=水·镜面反射  亮=陆·粗糙"
    consts = ("常量: VV极化 · C波段 5.6cm · 景内归一化(未绝对定标) · "
              f"暗区占比 {dark_frac*100:.0f}%")
    chain = "物理链: 地表粗糙度/含水量 → 介电常数 → 后向散射 → 成像 → 暗区=水"
    concl = ("结论: 该区域存在显著水体(洪水/渍涝)。" if lab == "water"
             else "结论: 该区域无显著水体。")
    fig.text(0.04, 0.235, legend, fontsize=6.5, fontproperties=_zh_font(6.5))
    fig.text(0.04, 0.185, consts, fontsize=6.5, fontproperties=_zh_font(6.5))
    fig.text(0.04, 0.135, chain, fontsize=6.5, fontproperties=_zh_font(6.5))
    fig.text(0.04, 0.085, concl, fontsize=7, color="#b00", fontproperties=_zh_font(7))
    fig.savefig(path, dpi=100); plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def normalize(im):
    """分位数线性拉伸到 0~255，抑制绝对强度差异。"""
    small = im.resize((1024, 1024))
    a = np.array(small).astype("float32")
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disaster", required=True, choices=["flood_water", "waterlogging", "sea_ice"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = os.path.join(GRD_ROOT, args.disaster)
    out = os.path.join(OUT_ROOT, args.disaster)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    for old in glob.glob(os.path.join(img_dir, "grd_*.png")):
        os.remove(old)

    zips = sorted(glob.glob(os.path.join(src, "*.zip")))
    if args.limit:
        zips = zips[:args.limit]
    print(f"{args.disaster}: {len(zips)} 景")

    records = []
    for zp in zips:
        stamp = os.path.basename(zp).split("_")[4][:8]  # 日期
        try:
            im = read_vv(zp)
            if im is None:
                continue
            lo, hi = normalize(im)
        except Exception as e:
            print(f"  x {stamp}: {e}"); continue
        W, H = im.size
        buckets = {}
        for y in range(0, H - WIN + 1, STRIDE):
            for x in range(0, W - WIN + 1, STRIDE):
                crop = im.crop((x, y, x + WIN, y + WIN))
                a = np.array(crop).astype("float32")
                # 归一化到 0~255
                b = np.clip((a - lo) / (hi - lo) * 255, 0, 255)
                darkfrac = float((b < 60).mean())   # 暗区占比
                if darkfrac > HI:
                    lab = "water"
                elif darkfrac < LO:
                    lab = "land"
                else:
                    continue
                buckets.setdefault(lab, []).append((darkfrac, x, y))
        picked = []
        for lab, lst in buckets.items():
            lst.sort(key=lambda t: t[0], reverse=(lab == "water"))
            picked += [(lab, t) for t in lst[:MAX_PER_FILE]]
        for lab, (df, x, y) in picked:
            fn = f"grd_{args.disaster}_{stamp}_{x}_{y}.png"
            crop = im.crop((x, y, x + WIN, y + WIN))
            a = np.clip((np.array(crop).astype("float32") - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
            render(a, os.path.join(img_dir, fn),
                   f"Sentinel-1 GRD 相对后向散射 · {args.disaster} · {stamp}", df, lab)
            if lab == "water":
                ans = (f"Yes. There is a significant water body in this region. "
                       f"Backscatter dark-area fraction = {df*100:.0f}% "
                       f"(threshold {HI*100:.0f}%), indicating specular reflection from water "
                       f"(source: Sentinel-1 GRD, VV, C-band).")
            else:
                ans = (f"No. No significant water body. "
                       f"Backscatter dark-area fraction = {df*100:.0f}% "
                       f"(threshold {LO*100:.0f}%), region dominated by rough land "
                       f"(source: Sentinel-1 GRD, VV, C-band).")
            records.append({
                "image": fn, "stamp": stamp, "label": lab, "dark_frac": round(df, 3),
                "conversations": [
                    {"from": "human", "value": PROMPT},
                    {"from": "gpt", "value": ans},
                ],
            })
        print(f"  {stamp}: {len(picked)} 窗 ({dict((k,len(v)) for k,v in buckets.items())})", flush=True)

    from collections import Counter, defaultdict
    import random as rnd
    rnd.seed(7)
    print(f"\n合计 {len(records)}")
    print("标签分布:", dict(Counter(r["label"] for r in records)))
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        rnd.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 5))
        val.extend(lst[:k]); train.extend(lst[k:])
    json.dump(train, open(os.path.join(out, f"train_{args.disaster}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(out, f"train_{args.disaster}_val.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"train {len(train)} | val {len(val)}")


if __name__ == "__main__":
    main()

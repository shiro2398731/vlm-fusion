# -*- coding: utf-8 -*-
r"""
渤海海冰 · Sentinel-2 光学渲染（扩充 sea_ice 正样本）

## 这一格是什么

灾害 = 海冰/凌汛，源 = 01 光学 VNIR，阶段 = 灾中。
物理量：**地表反射率导出的冰/水分隔**，不是拍照片。

## 判据（有出处，不拍数）

**NDSI（归一化雪指数）= (B03 − B11) / (B03 + B11)**

这是 MODIS 雪产品（Hall et al. 1995）与 Sentinel-2 海冰监测文献里的通用做法：
  NDSI > 0.4  -> 雪/冰
  配合 B08（NIR）低反射率（< 0.15）排除裸地

**海冰密集度分档沿用 WMO 标准里的 15% 门槛**：
  - 窗口内 NDSI>0.4 且落在水体区域的像元占比
  - >= 15%  -> ice（有冰）
  - < 15%   -> open_water（开阔水面）

注意：S2 是 10~20 m 分辨率，算的是**像元级冰覆盖**，
不等于 WMO 的「海冰密集度」（那是卫星格网单元的冰占比）。
图注里会明确标“pixel ice fraction”，**不声称密集度**。

## 输出双产品（多角度，符合热/光两源的物理差异）

1. `intrinsic_ndsi`      NDSI 场（无量纲），固定范围 -0.5 ~ 0.9
   用 RdBu（红=冰/雪，蓝=水），便于跨图比较
2. `feature_true_color`  真彩合成，给人看实际场景（物理量为反射率）

## 用法

  cd C:\vlmdisaster
  .\venv\Scripts\python.exe src\preprocess\sentinel2_seaice_render.py --probe
  .\venv\Scripts\python.exe src\preprocess\sentinel2_seaice_render.py

`--probe` 只报告每景能切出多少窗口，不写图。
"""

import argparse
import glob
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\sentinel2_seaice"
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\sea_ice_s2"
IMG = os.path.join(OUT, "images")

SCALE_DIV = 2          # 10980 / 2 = 5490
WIN = 256              # 窗口边长（输出 384x384）
STRIDE = 256           # 不重叠
MAX_PER_SCENE = 6
OUT_SIZE = 384

NDSI_SNOW = 0.4        # Hall et al. 1995
ICE_FRAC_MIN = 0.15    # WMO 15% 门槛（此处为像元冰覆盖比）

PROMPT = ("This is a Sentinel-2 surface-reflectance composite over a sea-ice region "
          "in the Bohai Sea, China. Classify: ice or open_water.")


def load_scene(zp):
    """读 B02/B03/B04/B08/B11 与 SCL，统一到同一网格

    2026-11-02 修：Sentinel-2 L2A 自基线 04.00 起，波段的 DN 要减去
    BOA_ADD_OFFSET（本场景 -1000）再除 BOA_QUANTIFICATION_VALUE（10000）。
    旧版只除了 10000，所有反射率偏高 0.1，导致水体被判成陆地、窗口数 0。
    """
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    target = 10980 // SCALE_DIV

    # 从元数据读 BOA_ADD_OFFSET（不同基线不同，不写死）；L1C 没有 BOA offset
    is_l1c = any("MSIL1C" in x for x in names)
    offset = 0.0 if is_l1c else -1000.0
    qv = 10000.0
    mtd = [x for x in names if ("MTD_MSIL2A" in x or "MTD_MSIL1C" in x)]
    if mtd:
        txt = z.read(mtd[0]).decode("utf-8", errors="replace")
        m = re.search(r"<BOA_QUANTIFICATION_VALUE[^>]*>([^<]+)<", txt)
        if m:
            qv = float(m.group(1))
        m = re.search(r'<BOA_ADD_OFFSET band_id="0">([^<]+)<', txt)
        if m:
            offset = float(m.group(1))
        else:
            offs = re.findall(r'<BOA_ADD_OFFSET band_id="\d+">([^<]+)<', txt)
            if offs:
                offset = float(offs[0])

    def grab(tag):
        pat = re.compile(r"_%s(?:_20m|_60m|_10m)?\.jp2$" % tag)
        cands = [x for x in names
                 if pat.search(x) and "MSK_" not in x and "PRB_" not in x]
        if not cands:
            return None
        im = Image.open(io.BytesIO(z.read(cands[0])))
        im = im.resize((target, target), Image.BILINEAR)
        dn = np.asarray(im).astype("float32")
        return np.clip((dn - offset) / qv, 0.0, None)

    bands = {}
    for b in ("B02", "B03", "B04", "B08", "B11"):
        bands[b] = grab(b)

    scl = None
    cs = [x for x in names if re.search(r"_SCL_20m\.jp2$", x)] or \
         [x for x in names if "_SCL_" in x and x.endswith(".jp2")]
    if cs:
        im = Image.open(io.BytesIO(z.read(cs[0]))).resize((target, target),
                                                          Image.NEAREST)
        scl = np.asarray(im).astype("uint8")
    z.close()
    return bands, scl


def valid_scene(bands):
    """场景是否有真实影像（空产品检测）

    2026-11-01 下载的 11 景里，有 3 景是 CDSE 返回的空产品：
    zip 结构完整但 IMG_DATA 几乎为 0。这里用反射率方差判断。
    """
    b = bands.get("B08")
    if b is None:
        return False, "缺 B08"
    v = b[np.isfinite(b)]
    if v.size < 10000:
        return False, "有效像元过少"
    if float(np.std(v)) < 0.005:
        return False, "影像近似常数（空产品）"
    return True, "ok"


def land_mask(bands, scl):
    """陆地/水体掩膜

    2026-11-02 修：旧版只用 SCL==6（水体）。实测渤海冬季场景 SCL 分布为
    5=云 820万 / 9=云影 776万 / 8=云 657万 / 3=云影 331万 / 10=薄卷云 304万 /
    11=雪 93万 / **6=水 仅 4.5 万（0.15%）**——海冰在 SCL 里被判成了 11（雪）。
    只用 6 当水体会把所有海冰像元排掉，窗口数 0。

    改为：水 + 雪（11）都算"海面"，云（8/9/10）和陆地（4/5）排除。
    这样冰和水都是有效分析区，分类交给 NDSI。
    """
    if scl is not None:
        water = scl == 6
        sea = np.isin(scl, [6, 11])       # 开阔水面 + 雪/冰
        cloud = np.isin(scl, [8, 9, 10])  # 云、云影、薄卷云
        land = np.isin(scl, [4, 5, 7])    # 植被/裸地/低反射率
        valid = sea & ~cloud & ~land
        return valid, land
    b03 = bands.get("B03")
    b08 = bands.get("B08")
    if b03 is None or b08 is None:
        return None, None
    ndwi = (b03 - b08) / (b03 + b08 + 1e-6)
    return ndwi > -0.35, ndwi <= -0.35


def ndsi_of(bands):
    b03 = bands.get("B03")
    b11 = bands.get("B11")
    if b03 is None or b11 is None:
        return None
    return (b03 - b11) / (b03 + b11 + 1e-6)


def find_windows(water, max_n):
    """在水体密集区找窗口；没有水体就全图均匀铺"""
    H, W = water.shape
    if H < WIN or W < WIN:
        return []
    ys, xs = np.nonzero(water)
    out = []
    if ys.size > WIN * WIN:
        cy, cx = int(np.median(ys)), int(np.median(xs))
        for dy, dx in ((0, 0), (0, WIN), (-WIN, 0), (WIN, 0), (0, -WIN)):
            y0 = int(np.clip(cy + dy - WIN // 2, 0, H - WIN))
            x0 = int(np.clip(cx + dx - WIN // 2, 0, W - WIN))
            if water[y0:y0 + WIN, x0:x0 + WIN].mean() > 0.5:
                if (y0, x0) not in out:
                    out.append((y0, x0))
            if len(out) >= max_n:
                return out
    if not out:
        for y0 in range(0, H - WIN + 1, STRIDE * 2):
            for x0 in range(0, W - WIN + 1, STRIDE * 2):
                out.append((y0, x0))
                if len(out) >= max_n:
                    return out
    return out


def render_ndsi(win, path, tag, lab, frac):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.3, 4.3), dpi=100)
    im = ax.imshow(win, cmap="RdBu", vmin=-0.5, vmax=0.9, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("NDSI = (B03-B11)/(B03+B11)", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("%s  NDSI   ice fraction %.1f%%" % (tag, frac * 100), fontsize=8)
    fig.text(0.02, 0.055, "NDSI>%.1f = snow/ice (Hall 1995)" % NDSI_SNOW, fontsize=6)
    fig.text(0.02, 0.02, "Pixel ice fraction, not WMO ice concentration", fontsize=5.5,
             color="#b00")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE),
                                           Image.BILINEAR).save(path)


def render_true_color(bands, y0, x0, path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    rgb = np.dstack([bands["B04"][y0:y0 + WIN, x0:x0 + WIN],
                     bands["B03"][y0:y0 + WIN, x0:x0 + WIN],
                     bands["B02"][y0:y0 + WIN, x0:x0 + WIN]])
    rgb = np.clip(rgb * 2.5, 0, 1)
    fig, ax = plt.subplots(figsize=(4.3, 4.3), dpi=100)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("%s  true color" % tag, fontsize=8)
    fig.text(0.02, 0.05, "Surface reflectance RGB = B04/B03/B02 (x2.5 gain)", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE),
                                           Image.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    zips = sorted(glob.glob(os.path.join(RAW, "*.zip")))
    print("场景 %d 个" % len(zips))
    if not zips:
        print("没有 zip，检查:", RAW)
        return

    if not args.probe:
        os.makedirs(IMG, exist_ok=True)
        for f in glob.glob(os.path.join(IMG, "*.png")):
            os.remove(f)

    recs = []
    skipped = []
    from collections import Counter
    cnt = Counter()

    for i, zp in enumerate(zips, 1):
        name = os.path.basename(zp)
        stem = name.split("_")[2][:8]          # 20251201
        tile = re.search(r"_(T\d\d[A-Z]{3})_", name)
        tile = tile.group(1) if tile else "NA"

        try:
            bands, scl = load_scene(zp)
        except Exception as e:
            print("  x %s 读取失败 %s" % (name[:40], str(e)[:50]))
            skipped.append((name, "读取失败"))
            continue

        ok, why = valid_scene(bands)
        if not ok:
            print("  - %s %s" % (name[:40], why))
            skipped.append((name, why))
            continue

        ndsi = ndsi_of(bands)
        if ndsi is None:
            print("  x %s 缺波段" % name[:40])
            skipped.append((name, "缺波段"))
            continue

        water, land = land_mask(bands, scl)
        if water is None:
            water = np.ones_like(ndsi, dtype=bool)
        wins = find_windows(water, MAX_PER_SCENE)
        scene_ok = 0
        for (y0, x0) in wins:
            n_win = ndsi[y0:y0 + WIN, x0:x0 + WIN]
            w_win = water[y0:y0 + WIN, x0:x0 + WIN]
            if w_win.sum() < WIN * WIN * 0.5:
                continue
            valid = np.isfinite(n_win) & w_win
            if valid.sum() < 2000:
                continue
            frac = float(np.mean(n_win[valid] > NDSI_SNOW))
            lab = "ice" if frac >= ICE_FRAC_MIN else "open_water"

            tag = "%s_%s_%d_%d" % (stem, tile, y0, x0)
            if not args.probe:
                p1 = os.path.join(IMG, "%s_ndsi.png" % tag)
                render_ndsi(n_win, p1, tag, lab, frac)
                p2 = os.path.join(IMG, "%s_rgb.png" % tag)
                render_true_color(bands, y0, x0, p2, tag)

            recs.append({
                "image": os.path.join(IMG, "%s_ndsi.png" % tag),
                "image_rgb": os.path.join(IMG, "%s_rgb.png" % tag),
                "product": "intrinsic_ndsi",
                "label": lab,
                "ice_frac": round(frac, 4),
                "date": stem,
                "tile": tile,
                "window": [y0, x0],
            })
            cnt[lab] += 1
            scene_ok += 1
        print("  [%2d/%d] %s %s  窗口 %d" % (i, len(zips), stem, tile, scene_ok))

    print("\n=== 汇总 ===")
    print("窗口样本 %d，标签 %s" % (len(recs), dict(cnt)))
    if skipped:
        print("跳过场景 %d 个:" % len(skipped))

    if args.probe:
        print("（probe 模式，未写图）")
        return

    if not recs:
        print("无样本，不写 json")
        return

    # 自检
    bad = [r["image"] for r in recs if not os.path.exists(r["image"])]
    if bad:
        print("❌ 有图没写出来:", bad[:3])
        return
    if cnt.get("ice", 0) == 0 or cnt.get("open_water", 0) == 0:
        print("⚠️  某一类为空，检查判据")

    # 写入（追加到已有的 sea_ice 目录，不动原 json）
    json.dump(recs, open(os.path.join(OUT, "train_sea_ice_s2.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n输出:", OUT)
    print("注意：这是新增的一批，原 sea_ice 的 %d 条没动。"
          "合并由后续步骤决定。" % 376)


if __name__ == "__main__":
    main()

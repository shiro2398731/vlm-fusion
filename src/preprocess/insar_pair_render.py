# -*- coding: utf-8 -*-
r"""
衡水 InSAR 单对差分形变 · 样本渲染（路 B）

为什么不用 SBAS
-------------
  三条冬季链的 SBAS 年速率相差 174 mm（-24 / +150 / +105），
  参考点不稳、轨道误差未校正，结果不自洽，判定不可用。
  改回单对差分：数据本身真实，只是不给年速率。

物理限制（必须写进图）
-------------------
  衡水文献速率 141 mm/a，单对间隔 12 天 -> 形变约 4.6 mm。
  而 good 档噪声 2.07~4.69 mm，信号与噪声同量级。
  所以沉降区不会很显著，这是物理事实，不靠调参遮掩。

三档标签
------
  沉降疑似   |median| >= 3 sigma
  稳定       |median| <  1 sigma
  不可判读   其余（处在噪声带里，判不清）

双产品
------
  本征场  displacement 栅格（米 -> 毫米显示）
  特色场  相干系数图（数据质量可视化）

用法
----
  python src/preprocess/insar_pair_render.py --pair 20241209__20241221
  python src/preprocess/insar_pair_render.py --all
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from src.insar.envi_io import read_envi  # noqa: E402

PAIRS = r"C:\vlmdisaster\outputs\henshui_ts\pairs"
QC = r"C:\vlmdisaster\outputs\henshui_ts\pair_qc.json"
OUT = r"C:\vlmdisaster\datasets\05_sar_insar\ground_subsidence_henshui"
IMG = os.path.join(OUT, "images")

WIN = 192
STRIDE = 192
OUT_SIZE = 384


def load_qc():
    if not os.path.exists(QC):
        return {}
    return {x["pair"]: x for x in json.load(open(QC, encoding="utf-8"))}


def read_pair(pid):
    base = os.path.join(PAIRS, pid, "deformation_map.data")
    disp, meta = read_envi(os.path.join(base, "displacement_VV.img"))

    coh = None
    fdir = os.path.join(PAIRS, pid, "filtered.data")
    if os.path.isdir(fdir):
        for fn in sorted(os.listdir(fdir)):
            if fn.startswith("coh") and fn.endswith(".img"):
                coh, _ = read_envi(os.path.join(fdir, fn))
                break
    return disp, coh, meta


def grade(d_mm, sigma_mm):
    if not np.isfinite(d_mm):
        return "unreadable"
    if abs(d_mm) >= 3 * sigma_mm:
        return "subsidence"
    if abs(d_mm) < sigma_mm:
        return "stable"
    return "unreadable"


def render(arr, path, tag, lab, kind, sigma=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    fig = plt.figure(figsize=(4.6, 4.9), dpi=100)
    ax = fig.add_axes([0.03, 0.16, 0.86, 0.78])

    if kind == "disp":
        v = float(np.nanpercentile(np.abs(arr), 98))
        v = max(v, 1.0)
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-v, vmax=v,
                       interpolation="nearest")
        cax = fig.add_axes([0.90, 0.16, 0.03, 0.78])
        fig.colorbar(im, cax=cax, label="LOS displacement (mm)")
    else:
        im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=1,
                       interpolation="nearest")
        cax = fig.add_axes([0.90, 0.16, 0.03, 0.78])
        fig.colorbar(im, cax=cax, label="Coherence")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("%s  %s" % (tag, "displacement" if kind == "disp" else "coherence"),
                 fontsize=8)

    if kind == "disp":
        fig.text(0.03, 0.095, "12-day differential LOS, NOT an annual rate",
                 fontsize=6.0)
    else:
        fig.text(0.03, 0.095, "Interferometric coherence (quality)",
                 fontsize=6.0)
    fig.text(0.03, 0.055, "Sensor: Sentinel-1 IW SLC, Hengshui, 13.9 m",
             fontsize=6.0)
    fig.text(0.03, 0.018,
             "Reading: %s  (noise sigma %.1f mm)" % (lab, sigma or 0),
             fontsize=5.8, color="#b00")

    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE),
                                              PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    qc = load_qc()

    if args.all:
        pids = [p for p, v in qc.items() if v.get("grade") in ("good", "fair")]
    elif args.pair:
        pids = [args.pair]
    else:
        print("需要 --pair 或 --all")
        return

    pids.sort()
    print("待渲染 %d 对" % len(pids))

    recs = []
    for pid in pids:
        info = qc.get(pid, {})
        sigma = float(info.get("noise_std_mm") or 5.0)
        try:
            disp, coh, meta = read_pair(pid)
        except Exception as e:
            print("  x %s %s" % (pid, str(e)[:60]))
            continue

        d = disp * 1000.0
        d[~np.isfinite(d)] = np.nan
        d[np.abs(d) > 500] = np.nan
        H, W = d.shape

        kept = 0
        for y in range(0, max(1, H - WIN), STRIDE):
            for x in range(0, max(1, W - WIN), STRIDE):
                sub = d[y:y + WIN, x:x + WIN]
                if sub.size == 0:
                    continue
                ok = np.isfinite(sub)
                if ok.mean() < 0.9:
                    continue
                med = float(np.median(sub[ok]))
                lab = grade(med, sigma)
                tag = "%s_%d_%d" % (pid, y, x)

                p1 = os.path.join(IMG, "%s_disp.png" % tag)
                render(sub, p1, tag, lab, "disp", sigma)
                recs.append({"image": os.path.basename(p1), "stem": tag,
                             "label": lab, "pair": pid,
                             "product": "intrinsic_los_displacement",
                             "median_mm": round(med, 2),
                             "sigma_mm": round(sigma, 2)})

                if coh is not None:
                    cs = coh[y:y + WIN, x:x + WIN]
                    p2 = os.path.join(IMG, "%s_coh.png" % tag)
                    render(cs, p2, tag, lab, "coh", sigma)
                    recs.append({"image": os.path.basename(p2), "stem": tag,
                                 "label": lab, "pair": pid,
                                 "product": "intrinsic_coherence",
                                 "median_mm": round(med, 2),
                                 "sigma_mm": round(sigma, 2)})
                kept += 1

        print("  v %s  窗口 %d  累计 %d" % (pid, kept, len(recs)))

    if not recs:
        print("无样本")
        return

    json.dump(recs, open(os.path.join(OUT, "train_subsidence_henshui.json"),
                         "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n样本 %d 条" % len(recs))
    print("标签:", dict(Counter(r["label"] for r in recs)))
    print("输出:", IMG)


if __name__ == "__main__":
    main()

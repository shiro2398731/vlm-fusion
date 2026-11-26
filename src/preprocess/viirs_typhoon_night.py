# -*- coding: utf-8 -*-
r"""
光学台风 · 夜间轨道补渲（只有 I04/I05 热红外，无反射率波段）

背景
----
VJ102IMG 的夜间过境没有太阳光，产品里不含 I01/I02/I03 反射率波段，
主渲染脚本 viirs_typhoon_render.py 读 I01 时会报
"object 'I01' doesn't exist" 并跳过这些轨道。

这些轨道并非废数据：本征场用的就是 I05 亮温，不需要可见光。

本脚本
------
  只处理缺失 I01 的轨道；
  只出**单产品**：I05 云顶亮温场；
  标签与白天轨道一致（按窗口内 <=200 K 像元占比）。

判据（同白天轨道）
----------------
  >= 1.0%   mature
  0.3~1.0%  developing
  < 0.3%    weak

输出写入同一个数据集目录，json 与白天轨道合并去重。

用法
----
  python src/preprocess/viirs_typhoon_night.py
  python src/preprocess/viirs_typhoon_night.py --keep 2
"""
import os
import re
import sys
import glob
import json
import argparse
import collections
import random

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

PILImage.MAX_IMAGE_PIXELS = None

RAW = r"C:\vlmdisaster\data\raw\01_optical_vnir\viirs_typhoon"
OUT = r"C:\vlmdisaster\datasets\01_optical_vnir\hurricane"
IMG = os.path.join(OUT, "images")

WIN = 1024
OUT_SIZE = 384
FILL = 65535
VAL_PER_GROUP = 30

T_COLD = 200.0
T_CONV = 220.0
T_CLOUD = 260.0
G_MATURE = 0.010
G_DEVELOPING = 0.003

PROMPT_BT = ("This is a VIIRS 11.45 micrometre brightness temperature field (kelvin, 375 m) "
             "over a tropical cyclone, acquired at night. Cold values indicate high, deep "
             "convective cloud tops. Describe the cloud-top temperature structure.")
ANS = {
    "mature": "The window shows a mature tropical cyclone: a cold central dense overcast "
              "with a large fraction of cloud tops below 200 K, indicating an eyewall of "
              "deep convection.",
    "developing": "The window shows organized deep convection with a moderate fraction of "
                  "cloud tops below 200 K. The system appears to be developing.",
    "weak": "The window shows ordinary cloud cover with few very cold tops; no organized "
            "tropical cyclone structure is evident.",
}


def has_reflective(fp):
    """是否有反射率波段（I01）。有 = 白天轨道"""
    try:
        with h5py.File(fp, "r") as f:
            return "I01" in f["observation_data"]
    except Exception:
        return False


def load_bt(fp):
    with h5py.File(fp, "r") as f:
        od = f["observation_data"]
        i05 = np.array(od["I05"][:], dtype="uint16")
        lut = np.array(od["I05_brightness_temperature_lut"][:], dtype="float32")
    dn = np.clip(i05, 0, len(lut) - 1)
    bt = lut[dn].astype("float32")
    bt[bt < 100.0] = np.nan
    bt[i05 == FILL] = np.nan
    return bt


def find_cold_center(bt, thr=T_CONV):
    m = np.isfinite(bt) & (bt <= thr)
    if m.sum() < 200:
        m = np.isfinite(bt) & (bt <= np.nanpercentile(bt, 2))
    if m.sum() < 50:
        return None
    ys, xs = np.nonzero(m)
    return int(np.mean(ys)), int(np.mean(xs))


def crop(bt, cy, cx, size=WIN, shift=(0, 0)):
    H, W = bt.shape
    y0 = int(np.clip(cy + shift[0] - size // 2, 0, max(0, H - size)))
    x0 = int(np.clip(cx + shift[1] - size // 2, 0, max(0, W - size)))
    return bt[y0:y0 + size, x0:x0 + size], (y0, x0)


def grade_of(bt_win):
    v = bt_win[np.isfinite(bt_win)]
    if v.size < 5000:
        return None, 0.0
    frac = float(np.mean(v <= T_COLD))
    if frac >= G_MATURE:
        return "mature", frac
    if frac >= G_DEVELOPING:
        return "developing", frac
    return "weak", frac


def render_bt(bt_win, path, tag, lab, frac):
    fig = plt.figure(figsize=(4.6, 5.2), dpi=100)
    ax = fig.add_axes([0.04, 0.22, 0.80, 0.70])
    im = ax.imshow(bt_win, cmap="turbo_r", vmin=180, vmax=300, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("%s  %s (night)" % (tag, lab), fontsize=8)
    cax = fig.add_axes([0.86, 0.22, 0.035, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("cloud-top BT (K)", fontsize=7)
    for t in (T_COLD, T_CONV, T_CLOUD):
        cb.ax.axhline(t, color="w", lw=0.7, ls="--")
    fig.text(0.04, 0.165, "Sensor: VIIRS/JPSS-1 I05 11.45 um, 375 m (night pass)", fontsize=6.2)
    fig.text(0.04, 0.120, "Criteria: cloud-top BT 260/220/200 K (deep convection)",
             fontsize=6.2)
    fig.text(0.04, 0.075, "Fraction <=200 K in window: %.2f %%" % (frac * 100), fontsize=6.2)
    fig.text(0.04, 0.030, "Grade: %s   (no reflective bands at night)" % lab,
             fontsize=6.0, color="#b00")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    PILImage.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE), PILImage.BILINEAR).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(IMG, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(RAW, "*.nc")))
    night = [f for f in all_files if not has_reflective(f)]
    if args.keep:
        night = night[:args.keep]
    print("总景数 %d   夜间轨道 %d" % (len(all_files), len(night)))
    if not night:
        print("没有夜间轨道需要处理")
        return

    recs = []
    counter = collections.Counter()

    for i, f in enumerate(night):
        m = re.search(r"A(\d{7})\.(\d{4})", os.path.basename(f))
        stamp = ("%s_%s" % (m.group(1), m.group(2))) if m else str(i)
        try:
            bt = load_bt(f)
        except Exception as e:
            print("  %s 读取失败 %s" % (stamp, str(e)[:60]))
            continue

        c = find_cold_center(bt)
        if c is None:
            print("  %s 找不到冷云区" % stamp)
            continue
        cy, cx = c

        wins = [(cy, cx, (0, 0)),
                (cy, cx, (0, int(WIN * 0.9))),
                (cy, cx, (int(WIN * 0.9), 0))]
        for (_, _, sh) in wins:
            bt_w, (y0, x0) = crop(bt, cy, cx, WIN, shift=sh)
            if bt_w.shape[0] < WIN or bt_w.shape[1] < WIN:
                continue
            lab, frac = grade_of(bt_w)
            if lab is None:
                continue
            counter[lab] += 1
            tag = "%s_%d_%d" % (stamp, y0, x0)
            p1 = os.path.join(IMG, "%s_bt_night.png" % tag)
            render_bt(bt_w, p1, tag, lab, frac)
            recs.append({"image": os.path.basename(p1), "stem": tag, "label": lab,
                         "product": "intrinsic_cloud_top_bt",
                         "frac_cold": round(frac, 4),
                         "conversations": [
                             {"from": "human", "value": "<image>" + PROMPT_BT},
                             {"from": "gpt",
                              "value": ANS[lab] + " Fraction of cloud tops at or below "
                                       "200 K is %.2f percent." % (frac * 100)}]})
        print("  [%d/%d] %s  冷云中心 (%d,%d)  累计 %d" % (i + 1, len(night), stamp, cy, cx, len(recs)))

    if not recs:
        print("无样本")
        return

    # 与白天轨道的 json 合并
    tj = os.path.join(OUT, "train_hurricane.json")
    vj = os.path.join(OUT, "train_hurricane_val.json")
    old = []
    for p in (tj, vj):
        if os.path.exists(p):
            try:
                old += json.load(open(p, encoding="utf-8"))
            except Exception:
                pass
    seen = set()
    merged = []
    for r in old + recs:
        if r["image"] in seen:
            continue
        seen.add(r["image"])
        merged.append(r)

    random.seed(53)
    by = collections.defaultdict(list)
    for r in merged:
        by[(r["product"], r["label"])].append(r)
    val, train = [], []
    for k, lst in by.items():
        lst = lst[:]
        random.shuffle(lst)
        n = min(VAL_PER_GROUP, max(5, len(lst) // 5))
        val.extend(lst[:n])
        train.extend(lst[n:])

    json.dump(train, open(tj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(vj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n夜间新增 %d 条" % len(recs))
    print("夜间分布:", dict(counter))
    print("合并后 train %d | val %d" % (len(train), len(val)))


if __name__ == "__main__":
    main()

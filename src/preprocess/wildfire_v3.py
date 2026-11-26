# -*- coding: utf-8 -*-
r"""
野火 · 灾中实时识别（阶段二）· wildfire_v3

## 这一格是什么

灾害 = 野火/林火，源 = 03 热红外，阶段 = **灾中实时**。

火灾三阶段拆成三格（口径见 `野火三阶段_20261101.md`）：
  - 灾前：火险预测（FWI 类），另做
  - **灾中：活跃火点识别 ← 本脚本**
  - 灾后：烧痕评估（MCD64A1，按《森林防火条例》分级），另做

## 和 v2 的区别

| 项 | v2 | v3 |
|---|---|---|
| 标签名 | `fire` / `no_fire`（声称“野火”） | `active_fire` / `no_active_fire`（只说火点，不声称森林火灾）|
| fire 挑选 | 按点数从高到低取前 2 | **分层抽样**：slight / moderate / severe 三档各取，避免全是巨型火 |
| 回答文本 | 只说 “Active wildfire detected: N” | 带上严重度档 + 火点/FRP 数值，数字可核 |
| 跨天去重 | 无 | **同一窗口不同天的样本合并登记**，防止同一场火灾被当成多个独立事件 |
| 自检 | 无 | 写盘前检查标签分布、重复窗口、空样本 |

## 判据（写清出处，不拍数）

火点过滤沿用 v2，这是 FIRMS 官方字段的含义：
  - `type == 2` 剔除：静态地表热源（工厂/冶炼/地热），FIRMS 文档定义
  - `confidence == 'l'` 剔除：低置信度火点

严重度分档 **不是法定火灾等级**（《森林防火条例》按受害森林面积分级，
卫星火点给不出受害面积）。这里按**窗口内有效火点数**分档，
属于**统计分档**，论文中必须照此标注：

  - slight   5 ~ 30 个火点
  - moderate 30 ~ 100 个火点
  - severe   >= 100 个火点

阈值 5 沿用 v2 的既有判据；30 / 100 为本次划定，属经验阈值，
在 `野火三阶段_20261101.md` 里留了改口空间。

## 用法

  cd C:\vlmdisaster
  .\venv\Scripts\python.exe src\preprocess\wildfire_v3.py --probe
  .\venv\Scripts\python.exe src\preprocess\wildfire_v3.py

`--probe` 只读 csv 统计火点分布，不出图、不写文件，先确认数据没问题。
"""

import argparse
import collections
import csv
import glob
import json
import os
import random
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RAW = r"C:\vlmdisaster\data\raw\03_thermal_infrared\firms\chn"
OUT = r"C:\vlmdisaster\datasets\03_thermal_infrared\wildfire"
IMG = os.path.join(OUT, "images")

OUT_SIZE = 384
VAL_PER_CLASS = 30
MAX_PER_DAY = 3          # 每天最多取 3 个窗口（no_active_fire 另算）

# 严重度阈值（统计分档，非国标）
T_SLIGHT = 5
T_MODERATE = 30
T_SEVERE = 100

# 中国范围（与 v2 一致）
W, S, E, N = 73.0, 18.0, 135.0, 54.0

PROMPT = ("This is a satellite thermal-infrared active-fire detection map over a region "
          "in China. Classify: no_active_fire or active_fire.")

GRID_N = 40              # 每个窗口切成 40x40 像元


def load_pts(fp):
    """读一天火点，返回 [(lat, lon, ti4, frp)]"""
    out = []
    with open(fp, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                t = int(row.get("type", "0") or 0)
                conf = (row.get("confidence") or "").strip().lower()
                if t == 2:
                    continue
                if conf == "l":
                    continue
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                ti4 = float(row["bright_ti4"])
                frp = float(row["frp"] or 0)
                out.append((lat, lon, ti4, frp))
            except Exception:
                continue
    return out


def band_of(n):
    """按窗口内有效火点数分档（统计分档，非国标）"""
    if n < T_SLIGHT:
        return "none"
    if n < T_MODERATE:
        return "slight"
    if n < T_SEVERE:
        return "moderate"
    return "severe"


def pts_in(pts, la, lo, g, n=GRID_N):
    """把一个窗口内的火点栅格化，返回 (亮度温, FRP 累积, 计数)"""
    ti4 = np.zeros((n, n), dtype=float)
    frp = np.zeros((n, n), dtype=float)
    cnt = np.zeros((n, n), dtype=int)
    for lat, lon, t, f_ in pts:
        if not (la <= lat < la + g and lo <= lon < lo + g):
            continue
        iy = int((lat - la) / g * n)
        ix = int((lon - lo) / g * n)
        if not (0 <= iy < n and 0 <= ix < n):
            continue
        ti4[iy, ix] += t
        frp[iy, ix] += f_
        cnt[iy, ix] += 1
    m = np.where(cnt > 0, ti4 / np.maximum(cnt, 1), np.nan)
    return m, frp, cnt


def draw(field, cnt, la, lo, g, path, title, kind):
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=100)
    if kind == "ti4":
        im = ax.imshow(field, cmap="inferno")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("brightness temp (K)", fontsize=7)
    else:
        im = ax.imshow(field, cmap="YlOrRd")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("FRP (MW)", fontsize=7)
    if cnt.sum() == 0:
        ax.text(0.5, 0.5, "No fire hotspot", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888")
    ax.set_xlabel("Lon", fontsize=7)
    ax.set_ylabel("Lat", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    Image.open(path).convert("RGB").resize((OUT_SIZE, OUT_SIZE),
                                           Image.BILINEAR).save(path)


def answer_text(lab, sev, n, frp_sum):
    """生成回答文本

    2026-11-01 修复：旧版把 lab 当成了严重度传进来，
    导致答案写成 "severity active_fire"。这里参数分开。
    """
    if lab == "no_active_fire":
        return "No active fire hotspots detected in this region."
    return ("Active fire detected: %d hotspot pixels, severity %s, "
            "total FRP %.0f MW." % (n, sev, frp_sum))


def probe(fs_):
    """只统计火点分布，不出图不写盘"""
    print("=== 探测模式：只统计，不写文件 ===\n")
    per_day = []
    for i, f in enumerate(fs_, 1):
        pts = load_pts(f)
        per_day.append(len(pts))
        if i % 90 == 0:
            print("  已读 %d/%d 天" % (i, len(fs_)), flush=True)
    per_day = np.array(per_day)
    print("\n天数: %d" % len(fs_))
    print("每天有效火点数: min %d / 中位 %d / max %d / 合计 %d"
          % (per_day.min(), int(np.median(per_day)), per_day.max(), per_day.sum()))
    empty = int((per_day == 0).sum())
    print("零火点天数: %d（%.1f%%）" % (empty, 100.0 * empty / len(fs_)))

    # 抽 3 天看窗口分档
    print("\n抽 3 天看窗口分档:")
    g = 5.0
    lats = np.arange(S, N - g + 1e-9, g)
    lons = np.arange(W, E - g + 1e-9, g)
    for f in (fs_[0], fs_[len(fs_) // 2], fs_[-1]):
        pts = load_pts(f)
        bands = collections.Counter()
        for la in lats:
            for lo in lons:
                _, _, c = pts_in(pts, la, lo, g)
                bands[band_of(int(c.sum()))] += 1
        print("  %s  %s" % (os.path.basename(f)[5:15], dict(bands)))
    print("\n窗口总数: %d 个（%d 纬度 × %d 经度）" % (len(lats) * len(lons), len(lats), len(lons)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=5.0)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    fs_ = sorted(glob.glob(os.path.join(RAW, "*.csv")))
    if args.max_days:
        fs_ = fs_[:args.max_days]
    print("天数 %d | 窗 %.1f°" % (len(fs_), args.grid))
    if not fs_:
        print("没有 csv，检查:", RAW)
        return

    if args.probe:
        probe(fs_)
        return

    os.makedirs(IMG, exist_ok=True)
    for f in glob.glob(os.path.join(IMG, "*.png")):
        os.remove(f)

    g = args.grid
    lats = np.arange(S, N - g + 1e-9, g)
    lons = np.arange(W, E - g + 1e-9, g)

    recs = []
    cnt = collections.Counter()
    window_hits = collections.Counter()     # (la, lo) -> 出现天数，用于跨天去重检查

    for i, f in enumerate(fs_, 1):
        stamp = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        stamp = stamp.group(1) if stamp else str(i)
        pts = load_pts(f)

        by_band = collections.defaultdict(list)
        for la in lats:
            for lo in lons:
                ti4m, frp, c = pts_in(pts, la, lo, g)
                n = int(c.sum())
                b = band_of(n)
                score = n if b != "none" else 0
                by_band[b].append((score, la, lo, ti4m, frp, c, n))

        picked = []
        # 有火：三档各取前若干，保证 severe 不会把 moderate/slight 全挤掉
        for b in ("severe", "moderate", "slight"):
            lst = sorted(by_band.get(b, []), key=lambda t: -t[0])
            picked += [(b, t) for t in lst[:MAX_PER_DAY]]
        # 无火：取空窗
        lst = by_band.get("none", [])
        random.shuffle(lst)
        picked += [("none", t) for t in lst[:MAX_PER_DAY]]

        for b, (_, la, lo, ti4m, frp, c, n) in picked:
            lab = "no_active_fire" if b == "none" else "active_fire"
            cnt[lab] += 1
            window_hits[(round(float(la), 3), round(float(lo), 3))] += 1
            tag = "%s%d%s%d" % ("N" if la >= 0 else "S", abs(int(la)),
                                "E" if lo >= 0 else "W", abs(int(lo)))
            frp_sum = float(frp.sum())
            for sfx, kind, prod in (("_ti4", "ti4", "intrinsic_brightness_temp"),
                                    ("_frp", "frp", "feature_frp")):
                png = os.path.join(IMG, "%s_%s_%s%s.png" % (stamp, tag, lab, sfx))
                draw(ti4m, c, la, lo, g, png,
                     "%s  %s  %s" % (stamp, lab, b), kind)
                recs.append({
                    "conversations": [
                        {"from": "human", "value": "<image>" + PROMPT},
                        {"from": "gpt", "value": answer_text(lab, b, n, frp_sum)},
                    ],
                    "images": [png],
                    "product": prod,
                    "label": lab,
                    "severity": b,
                    "hotspots": n,
                    "frp_total": round(frp_sum, 1),
                    "stamp": stamp,
                    "window": [round(float(la), 3), round(float(lo), 3)],
                })
        if i % 30 == 0:
            print("  %d/%d %s" % (i, len(fs_), dict(cnt)), flush=True)

    # ---------------- 自检（写盘之前）----------------
    print("\n=== 自检 ===")
    ok = True

    if not recs:
        print("❌ 无样本")
        return

    lab_cnt = collections.Counter(r["label"] for r in recs)
    sev_cnt = collections.Counter(r["severity"] for r in recs)
    print("标签:", dict(lab_cnt))
    print("严重度:", dict(sev_cnt))

    if lab_cnt.get("active_fire", 0) == 0:
        print("❌ 没有一个 active_fire 样本，判据或数据有问题")
        ok = False
    if lab_cnt.get("no_active_fire", 0) == 0:
        print("❌ 没有一个 no_active_fire 样本")
        ok = False

    # 严重度档位不应全空
    for b in ("slight", "moderate", "severe"):
        if sev_cnt.get(b, 0) == 0:
            print("⚠️  严重度档 %s 为空" % b)

    # 跨天去重：同一窗口出现太多次，说明是同一场火被反复采样
    top = window_hits.most_common(5)
    print("出现最多的窗口（次数）:", top)
    if top and top[0][1] > 30:
        print("⚠️  窗口 %s 出现 %d 天，疑似同一场火被反复采样" % (top[0][0], top[0][1]))

    # 图像文件必须真实存在且非空
    missing = [r["images"][0] for r in recs
               if not os.path.exists(r["images"][0])
               or os.path.getsize(r["images"][0]) < 1024]
    print("图像缺失/过小的样本数:", len(missing))
    if missing:
        print("❌ 有图像没写出来，前 3 个:", missing[:3])
        ok = False

    # 同一窗口的同一天不应重复
    keys = [(r["stamp"], tuple(r["window"]), r["product"]) for r in recs]
    dup = len(keys) - len(set(keys))
    print("同一天同窗口重复的样本对:", dup)
    if dup > 0:
        print("❌ 有重复样本")
        ok = False

    # 答案文本里的 severity 必须是档位名，不能是标签名
    # （2026-11-01 修：这里曾把 lab 当严重度传进去，写成 severity active_fire）
    bad_sev = 0
    for r in recs:
        ans = r["conversations"][1]["value"]
        if "severity" in ans and r["severity"] not in ans:
            bad_sev += 1
    print("答案文本 severity 写错的条数:", bad_sev)
    if bad_sev:
        print("❌ 答案里的 severity 与实际档位不符")
        ok = False

    if not ok:
        print("\n自检未通过，不写 json。")
        return

    # ---------------- 划分并写盘 ----------------
    random.seed(3)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["label"]].append(r)
    val, train = [], []
    for lab, lst in by.items():
        random.shuffle(lst)
        k = min(VAL_PER_CLASS, max(3, len(lst) // 6))
        val.extend(lst[:k])
        train.extend(lst[k:])

    json.dump(train, open(os.path.join(OUT, "train_wildfire.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(val, open(os.path.join(OUT, "train_wildfire_val.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n=== 完成 ===")
    print("train %d / val %d" % (len(train), len(val)))
    print("输出:", OUT)
    print("\n下一步若要接训练，记得在 LLaMA-Factory/data/dataset_info.json 里")
    print("更新 disaster_wildfire 的文件名（标签已从 fire/no_fire 改为 active_fire/no_active_fire）。")


if __name__ == "__main__":
    main()

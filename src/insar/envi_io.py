# -*- coding: utf-8 -*-
r"""
SNAP ENVI 读取修正版

踩过的坑
-------
SNAP 导出的 .hdr 里写 `byte order = 1`，但数据实际是大端。
按小端读会得到一堆 ±3.4e38（float32 极值），看起来像"整幅无效"或
"位移 128 公里"，从而误判为"相干性全为 0""解缠失败"。

ENVI 官方定义：byte order = 0 为小端(LSF)，= 1 为大端(MSF)。
SNAP 导出的文件按 MSF 写（即大端），读取时不要按 `1=小端` 理解。

本模块统一按"大端优先，数值合理性校验通过才采用"的方式读取。
"""
import os
import re

import numpy as np

DT = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2"}


def read_hdr(hdr_path):
    h = open(hdr_path, encoding="utf-8", errors="replace").read()

    def kv(k):
        m = re.search(r"%s\s*=\s*(.+)" % re.escape(k), h)
        return m.group(1).strip() if m else None

    return {
        "samples": int(kv("samples")),
        "lines": int(kv("lines")),
        "bands": int(kv("bands") or 1),
        "dtype": DT.get(int(kv("data type")), "f4"),
        "byte_order_raw": kv("byte order"),
        "description": kv("description") or "",
        "map_info": kv("map info") or "",
    }


def read_envi(img_path, prefer="big"):
    """读 SNAP 导出的 ENVI 文件，自动判断字节序

    返回 (array[nrow,ncol], meta)
    """
    meta = read_hdr(img_path.replace(".img", ".hdr"))
    ncol, nrow, nb = meta["samples"], meta["lines"], meta["bands"]
    base = np.dtype(meta["dtype"])

    order = ["big", "little"] if prefer == "big" else ["little", "big"]
    best = None
    for bo in order:
        dt = base.newbyteorder(">" if bo == "big" else "<")
        a = np.fromfile(img_path, dtype=dt, count=ncol * nrow * nb)
        if a.size < ncol * nrow * nb:
            continue
        a = a.reshape(nb, nrow, ncol)
        first = a[0]
        fin = np.isfinite(first)
        if fin.sum() == 0:
            continue
        v = first[fin]
        # 合理性判据：中位数不能是天文数字
        med = float(np.median(v))
        if abs(med) < 1e6:
            meta["byte_order_used"] = bo
            return first, meta
        if best is None:
            best = (first, meta, bo)

    if best:
        meta["byte_order_used"] = best[2]
        return best[0], meta
    raise RuntimeError("无法读取 %s" % img_path)


def stats(a, name="", unit_scale=1.0, unit=""):
    good = np.isfinite(a) & (np.abs(a) < 1e30)
    v = a[good]
    if v.size == 0:
        print("  %s: 无有效像元" % name)
        return
    print("  %s  有效 %.1f%%" % (name, good.mean() * 100))
    print("     min %.5g  1%% %.5g  中位 %.5g  99%% %.5g  max %.5g %s"
          % (v.min() * unit_scale, np.percentile(v, 1) * unit_scale,
             np.median(v) * unit_scale, np.percentile(v, 99) * unit_scale,
             v.max() * unit_scale, unit))

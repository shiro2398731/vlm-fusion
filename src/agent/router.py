# -*- coding: utf-8 -*-
r"""
路由表：遥感源 → 候选元素

## 为什么不用数据里的 source 字段

各元素答案 JSON 里的 "source" 是答案模板里手写的，不可信：
  flare / cme 标成 OPT（其实是空间天气）
  geomagnetic 标成 MW
  hurricane_opt 标成 TIR（其实是光学）
拿它做路由会大面积错配。

## 采用的口径

每个元素的图都躺在 datasets\<XX_源>\<子目录>\images 里，
**图片实际所在的源目录**才是这个元素真实的遥感源。
本模块直接读训练数据第一条的 image 路径推导，不手写。

## 缓存

29 个 JSON 全量解析约 1~2 秒，结果缓存到 outputs\route_table.json。
加 --rebuild 强制重建。
"""

import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\vlmdisaster"
LF_DATA = os.path.join(ROOT, "LLaMA-Factory", "data")
INFO = os.path.join(LF_DATA, "dataset_info.json")
OUT_ROOT = os.path.join(ROOT, "outputs")
CACHE = os.path.join(OUT_ROOT, "route_table.json")

# datasets 目录前缀 → 源代号 / 中文名
CATEGORIES = {
    "01_optical_vnir": ("OPT", "光学可见光近红外"),
    "02_hyperspectral": ("HSI", "高光谱"),
    "03_thermal_infrared": ("TIR", "热红外"),
    "04_passive_microwave": ("MW", "被动微波"),
    "05_sar_insar": ("SAR", "SAR/InSAR"),
    "06_radar_precipitation": ("PRECIP", "雷达降水"),
    "07_scatterometer": ("SCAT", "散射计"),
    "08_altimeter": ("ALT", "测高仪"),
    "09_gravity": ("GRV", "重力卫星"),
    "10_lightning": ("LIT", "闪电成像"),
    "11_atmospheric": ("ATM", "大气成分"),
    "12_space_weather": ("SPACE", "空间天气"),
}

# 元素 → 这个元素的图长什么样（界面和命令行都用它提示用户选择）
NOTE = {
    "lst": "MODIS 地表温度昼夜图（区域），判寒潮/热浪",
    "drought_tir": "MODIS 昼夜地表温度（区域），判干旱",
    "volcano_thermal": "火山口亮温热点图，判热异常",
    "wildfire": "火点/火辐射功率分布图，判野火",
    "enso": "赤道太平洋海温距平全球图，判厄尔尼诺",
    "iod": "印度洋海温距平全球图，判偶极",
    "marine_heatwave": "海表温度距平全球图，判海洋热浪",
    "snow_fsc": "积雪覆盖比例（SCA）图",
    "snowstorm": "被动微波积雪深度/雪水当量图",
    "smap": "土壤湿度分布图",
    "green_tide": "光学真彩色海岸图，判绿潮",
    "flood_opt": "光学水体指数图，判洪水/内涝（与 SAR 的 flood_water 不同源）",
    "hurricane_opt": "光学台风云系图，判台风形态",
    "landslide": "光学灾前灾后对比，判滑坡痕迹",
    "sand_dust": "VIIRS 沙尘指数图",
    "sanddust": "MODIS AOD 沙尘图",
    "haze": "MODIS 气溶胶光学厚度图，判霾",
    "volcano_ndvi": "火山区 NDVI 变化图，判火山灰/亮羽流",
    "earthquake_def": "InSAR LOS 形变图，判地震形变",
    "landslide_def": "InSAR LOS 形变图，判滑坡形变",
    "groundwater_overuse": "InSAR 形变时间序列/速率图，判地下水超采",
    "flood_water": "SAR 后向散射图，判洪水淹没",
    "waterlogging": "SAR 影像，判城市内涝",
    "gpm": "GPM 累积降水分布图",
    "hurricane_rain": "台风区域累积降水图",
    "wave": "测高有效波高分布图",
    "flare": "GOES XRS 通量时间曲线",
    "cme": "日冕抛射白光影像",
    "geomagnetic": "Dst 指数时间曲线",
}


def category_of(image_path):
    """从图片绝对路径推出源类别代号"""
    parts = str(image_path).replace("/", "\\").split("\\")
    for p in parts:
        if p in CATEGORIES:
            return CATEGORIES[p][0]
    return "UNKNOWN"


def first_gold(rec):
    for t in rec.get("conversations") or []:
        if t.get("from") == "gpt":
            try:
                return json.loads(t.get("value", ""))
            except Exception:
                return None
    return None


def first_question(rec):
    for t in rec.get("conversations") or []:
        if t.get("from") == "human":
            return t.get("value", "")
    return ""


# 2026-11-17：gpm / coastal_high_water 两个原版的图上有「印答案」泄漏，
# 训练数据里仍有，但论文与演示一律用去标题重训的 gpm_nt / coastal_nt。
# 这里不把它们放进路由，避免演示时同一灾害出现两个候选、且其中一个引用泄漏模型。
LEGACY = {"gpm", "coastal_high_water"}


def build():
    """扫描全部已注册元素，产出 el -> 元信息"""
    info = json.load(open(INFO, encoding="utf-8"))
    table = {}
    for key in sorted(info):
        m = re.fullmatch(r"disaster_(.+)_json", key)
        if not m:
            continue
        el = m.group(1)
        if el in LEGACY:
            continue
        train_p = os.path.join(LF_DATA, info[key]["file_name"])
        if not os.path.exists(train_p):
            continue
        recs = json.load(open(train_p, encoding="utf-8"))
        if not recs:
            continue
        img = recs[0].get("image") or (recs[0].get("images") or [None])[0]
        gold = first_gold(recs[0]) or {}
        table[el] = {
            "element": el,
            "category": category_of(img),
            "question": first_question(recs[0]),
            "n_train": len(recs),
            "note": NOTE.get(el, ""),
            "n_val": (len(json.load(open(os.path.join(LF_DATA, info["disaster_%s_json_val" % el]["file_name"]),
                                        encoding="utf-8")))
                      if "disaster_%s_json_val" % el in info else 0),
            "adapter": os.path.join(OUT_ROOT, "qwen3vl-lora-%s-json" % el),
            "physical_quantity": gold.get("physical_quantity"),
        }
    return table


def load(rebuild=False):
    # 缓存要能自动失效：dataset_info.json 比缓存新（新增了元素）就重建，
    # 否则会出现「元素明明训好了，agent 却说没有可用 adapter」的错觉。
    if not rebuild and os.path.exists(CACHE):
        try:
            if os.path.getmtime(CACHE) >= os.path.getmtime(INFO):
                table = json.load(open(CACHE, encoding="utf-8"))
                if table:
                    return table
        except Exception:
            pass
    table = build()
    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump(table, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return table


def available(table):
    """只保留 adapter 真的存在的元素"""
    return {el: v for el, v in table.items()
            if os.path.exists(os.path.join(v["adapter"], "adapter_model.safetensors"))}


def candidates(table, category):
    """给定源代号，返回该源下的候选元素"""
    return sorted(el for el, v in available(table).items() if v["category"] == category)


def categories_in_use(table):
    out = {}
    for el, v in available(table).items():
        out.setdefault(v["category"], []).append(el)
    return {k: sorted(v) for k, v in sorted(out.items())}


def main():
    table = load(rebuild="--rebuild" in sys.argv)
    usable = available(table)
    print("=" * 74)
    print("  路由表：共 %d 个注册元素，其中 %d 个有 adapter" % (len(table), len(usable)))
    print("=" * 74)
    for cat, els in categories_in_use(table).items():
        cn = next((v[1] for k, v in CATEGORIES.items() if v[0] == cat), cat)
        print("\n[%s] %s —— %d 个" % (cat, cn, len(els)))
        for el in els:
            v = table[el]
            print("   %-20s 训练 %5d 条 / 验证 %4d 条   %s"
                  % (el, v["n_train"], v["n_val"], v.get("note", "")))
    miss = [el for el in table if el not in usable]
    if miss:
        print("\n[无 adapter] %s" % ", ".join(sorted(miss)))
    print("\n缓存: %s" % CACHE)


if __name__ == "__main__":
    main()

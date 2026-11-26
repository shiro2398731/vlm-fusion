"""
均衡采样脚本 v2 — 按灾害事件类型均衡采样

确保每种灾害类型（地震/野火/洪水/飓风/龙卷风/火山/海啸/滑坡/爆炸）都有代表，
避免之前 max_samples 取前25000条导致的类型缺失。
"""
import json
import random
import re
from collections import defaultdict

DATA_PATH = r"C:\vlmdisaster\data\DisasterM3\DisasterM3_Instruct\train_release.json"
OUT_PATH = r"C:\vlmdisaster\data\DisasterM3\DisasterM3_Instruct\train_balanced_25000.json"

# 事件前缀 → 灾害类型映射
EVENT_TYPE = {
    "turkey_earthquake": "earthquake",
    "morocco_earthquake": "earthquake",
    "morroco_earthquake": "earthquake",
    "haiti_earthquake": "earthquake",
    "portugal_wildfire": "wildfire",
    "socal_fire": "wildfire",
    "pinery_bushfire": "wildfire",
    "woolsey_fire": "wildfire",
    "santa_rosa_wildfire": "wildfire",
    "marshall_wildfire": "wildfire",
    "hawaii_wildfire": "wildfire",
    "nepal_flooding": "flood",
    "midwest_flooding": "flood",
    "libya_flood": "flood",
    "kalehe_flooding": "flood",
    "hurricane_michael": "hurricane",
    "hurricane_harvey": "hurricane",
    "hurricane_florence": "hurricane",
    "hurricane_matthew": "hurricane",
    "mexico_hurricane": "hurricane",
    "myanmar_hurricane": "hurricane",
    "tuscaloosa_tornado": "tornado",
    "moore_tornado": "tornado",
    "spain_volcano": "volcano",
    "la_palma_volcano": "volcano",
    "lower_puna_volcano": "volcano",
    "rwanda_volcano": "volcano",
    "palu_tsunami": "tsunami",
    "sunda_tsunami": "tsunami",
    "shovi_landslide": "landslide",
    "bata_explosion": "explosion",
    "beriut_explosion": "explosion",
    "ukraine_conflict": "conflict",
}

# 各灾害类型的采样配额（总计约25000）
# 火山/海啸/滑坡/爆炸等少样本类型全取，保证覆盖
QUOTA = {
    "earthquake": 4000,
    "wildfire": 4000,
    "flood": 4000,
    "hurricane": 4000,
    "tornado": 2500,
    "volcano": 4300,      # 全取（4430），火山是测试重点
    "tsunami": 1000,      # 全取（1014）
    "landslide": 126,     # 全取
    "explosion": 1200,    # 全取（1242）
    "conflict": 1500,     # 全取（1546）
}


def get_event_type(record):
    """根据图像路径判断灾害事件类型"""
    p = record.get("pre_image_path", "") or record.get("image_path", "")
    name = p.split("\\")[-1]
    # 去掉编号后缀
    m = re.match(r"([a-z_]+?)(?:_\d|_\d+_|_pre|_post|_sar)", name)
    prefix = m.group(1) if m else name.split("_")[0]
    # 归一化（morroco vs morocco）
    prefix = prefix.rstrip("_")
    return EVENT_TYPE.get(prefix, "unknown")


def main():
    random.seed(42)
    
    print("=" * 55)
    print("  均衡采样 v2 — 按灾害类型 ♪")
    print("=" * 55)
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 按灾害类型分组
    grouped = defaultdict(list)
    unknown = []
    for r in data:
        t = get_event_type(r)
        if t == "unknown":
            unknown.append(r)
        else:
            grouped[t].append(r)
    
    # 采样
    sampled = []
    for dtype, quota in QUOTA.items():
        pool = grouped.get(dtype, [])
        n = min(quota, len(pool))
        chosen = random.sample(pool, n) if n < len(pool) else pool
        sampled.extend(chosen)
        print(f"  {dtype}: 取 {n}/{len(pool)}")
    
    if unknown:
        print(f"  unknown: {len(unknown)} 条未分类")
    
    random.shuffle(sampled)
    print(f"\n  总计采样: {len(sampled)} 条")
    
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()


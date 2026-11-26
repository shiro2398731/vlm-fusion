"""
灾害类型识别数据生成器（v3 均衡版）

方案（白默提出）：
  1. 把 DisasterM3 全部数据按灾害类型分为 9 类
  2. 每类采样到"差距可接受"的目标数量（滑坡样本少则全用）
  3. 严格双图成对：只保留 pre+post 都存在的记录
  4. 混合少量其他任务样本防灾难性遗忘

修复的 bug：
  - 事件前缀提取：turkey_earthquake0_pre_343.png → turkey_earthquake（原脚本提取成 turkey，漏掉1.2万条地震）

用法: python src/preprocess/type_label_generator.py
输出: data/DisasterM3/DisasterM3_Instruct/train_balanced_all.json
"""
import json
import re
import os
import sys
import random
from collections import defaultdict

# Windows GBK 控制台兼容
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA_PATH = r"C:\vlmdisaster\data\DisasterM3\DisasterM3_Instruct\train_release.json"
IMG_BASE = r"C:\vlmdisaster\data\DisasterM3\DisasterM3_Instruct"
OUT_PATH = r"C:\vlmdisaster\data\DisasterM3\DisasterM3_Instruct\train_balanced_all.json"

# 每类类型识别目标样本数（滑坡/海啸样本少则全用）
# 预算控制：后续热红外 + InSAR/SAR 还要各练1-2次，本次训练需省时
# v4: 弱类增量——地震/洪水/飓风(v2混淆类)提量到2000，其他1000
TARGET_PER_TYPE = {
    "earthquake": 2000,   # v2: 0/2 ❌ 地震→飓风/洪水
    "flood": 2000,        # v2: 0/1 ❌ 洪水→地震
    "hurricane": 2000,    # v2: 0/1 ❌ 飓风→龙卷风
    "wildfire": 1000,
    "tornado": 1000,
    "volcano": 1000,
    "explosion": 1000,
    "tsunami": 1000,
    "landslide": 1000,
}
# 其他任务每类防遗忘采样数
TARGET_OTHER_PER_TASK = 300

# 类型识别 prompt（对齐训练格式）
PROMPT = "What kind of calamity has this area experienced?"

# 事件 → 灾害类型（完整映射，覆盖全部9类）
EVENT_TYPE = {
    # 地震
    "turkey_earthquake": "earthquake", "morroco_earthquake": "earthquake",
    "morocco_earthquake": "earthquake", "haiti_earthquake": "earthquake",
    # 野火
    "portugal_wildfire": "wildfire", "socal_fire": "wildfire",
    "pinery_bushfire": "wildfire", "woolsey_fire": "wildfire",
    "santa_rosa_wildfire": "wildfire", "hawaii_wildfire": "wildfire",
    "marshall_wildfire": "wildfire",
    # 飓风
    "hurricane_michael": "hurricane", "hurricane_harvey": "hurricane",
    "hurricane_florence": "hurricane", "hurricane_matthew": "hurricane",
    "mexico_hurricane": "hurricane", "myanmar_hurricane": "hurricane",
    # 洪水
    "nepal_flooding": "flood", "midwest_flooding": "flood",
    "libya_flood": "flood", "kalehe_flooding": "flood",
    # 龙卷风
    "tuscaloosa_tornado": "tornado", "moore_tornado": "tornado",
    # 火山
    "spain_volcano": "volcano", "la_palma_volcano": "volcano",
    "lower_puna_volcano": "volcano", "rwanda_volcano": "volcano",
    # 爆炸（含战争冲突）
    "bata_explosion": "explosion", "beriut_explosion": "explosion",
    "ukraine_conflict": "explosion",
    # 海啸
    "palu_tsunami": "tsunami", "sunda_tsunami": "tsunami",
    # 滑坡
    "shovi_landslide": "landslide",
}

# 灾害类型 → 答案模板（对齐评估关键词）
ANSWER_TEMPLATE = {
    "earthquake": "This area has been affected by an earthquake.",
    "wildfire": "This area has been affected by a wildfire.",
    "hurricane": "This area has been affected by a hurricane.",
    "flood": "This area has been affected by a flood.",
    "tornado": "This area has been affected by a tornado.",
    "volcano": "This area has been affected by a volcanic eruption.",
    "explosion": "This area has been affected by an explosion.",
    "tsunami": "This area has been affected by a tsunami.",
    "landslide": "This area has been affected by a landslide.",
}

# 防遗忘的其他任务（保留原问答格式）
OTHER_TASKS = [
    "Disaster Bearing Bodies Recognition",
    "Building Damage Counting",
    "disaster caption",
    "disaster restoration advice",
    "Road Damage Counting",
    "Disaster Scene Recognition",
]


def get_event_prefix(name):
    """提取事件前缀（修复版：兼容 turkey_earthquake0_pre_343.png 等格式）"""
    name = name.rsplit(".", 1)[0]  # 去扩展名
    # 截断到 _pre/_post/_sar 标记
    for marker in ("_pre", "_post", "_sar"):
        idx = name.find(marker)
        if idx != -1:
            name = name[:idx]
            break
    # 去掉尾部数字（如 turkey_earthquake0 → turkey_earthquake）
    name = re.sub(r"\d+_?$", "", name).rstrip("_")
    return name


def is_paired(record):
    """严格双图成对检查：pre+post 都存在且文件存在"""
    pre = record.get("pre_image_path", "")
    post = record.get("post_image_path", "")
    if not pre or not post:
        return False
    pre_full = os.path.join(IMG_BASE, pre)
    post_full = os.path.join(IMG_BASE, post)
    return os.path.exists(pre_full) and os.path.exists(post_full)


def main():
    random.seed(42)
    print("=" * 60)
    print("  灾害类型识别数据生成器 v3（均衡版）♪")
    print("=" * 60)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  总记录: {len(data)}")

    # ========== 1. 按灾害类型分组（只收双图记录） ==========
    grouped = defaultdict(list)  # 灾害类型 -> 记录列表
    skipped_single = 0
    unknown_events = defaultdict(int)
    for r in data:
        if not is_paired(r):
            skipped_single += 1
            continue
        p = r.get("pre_image_path", "")
        name = p.replace("\\", "/").split("/")[-1]
        prefix = get_event_prefix(name)
        dtype = EVENT_TYPE.get(prefix)
        if dtype:
            grouped[dtype].append(r)
        else:
            unknown_events[prefix] += 1

    print(f"  双图成对记录: {len(data) - skipped_single}（排除单图 {skipped_single} 条）")
    print(f"  未识别事件: {sum(unknown_events.values())} 条")
    for k, v in sorted(unknown_events.items(), key=lambda x: -x[1])[:10]:
        print(f"    {k}: {v}")

    # ========== 2. 每类均衡采样 + 生成类型识别问答对 ==========
    print("\n  === 各类样本量（v4 弱类增量采样） ===")
    type_records = []
    for dtype in sorted(grouped.keys()):
        samples = grouped[dtype]
        target = TARGET_PER_TYPE.get(dtype, 1000)  # dict 兼容
        n = min(target, len(samples))
        chosen = random.sample(samples, n)
        for r in chosen:
            type_records.append({
                "task": "Disaster Type Recognition",
                "prompts": PROMPT,
                "training_answer": ANSWER_TEMPLATE[dtype],
                "pre_image_path": r["pre_image_path"],
                "post_image_path": r["post_image_path"],
            })
        print(f"    {dtype:<12} 可用{len(samples):>6} → 采样{n:>5}")

    # ========== 3. 其他任务防遗忘（保留原问答） ==========
    print("\n  === 其他任务防遗忘采样 ===")
    other_records = []
    for task in OTHER_TASKS:
        samples = [r for r in data if r.get("task") == task and is_paired(r)]
        n = min(TARGET_OTHER_PER_TASK, len(samples))
        chosen = random.sample(samples, n)
        for r in chosen:
            other_records.append({
                "task": r["task"],
                "prompts": r.get("prompts", ""),
                "training_answer": r.get("training_answer") or r.get("ground_truth", ""),
                "pre_image_path": r["pre_image_path"],
                "post_image_path": r["post_image_path"],
            })
        print(f"    {task[:40]:<42} 可用{len(samples):>6} → 采样{n:>4}")

    # ========== 4. 合并输出 ==========
    all_records = type_records + other_records
    random.shuffle(all_records)
    print(f"\n  总计: {len(all_records)} 条（类型识别 {len(type_records)} + 防遗忘 {len(other_records)}）")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {OUT_PATH}")
    print()
    print("  下一步：")
    print("  1. 转换: python src/preprocess/data_converter.py --skip_extract --input train_balanced_all.json")
    print("  2. 续训: 加载 full 模型，输出 typeboost-v2")


if __name__ == "__main__":
    main()

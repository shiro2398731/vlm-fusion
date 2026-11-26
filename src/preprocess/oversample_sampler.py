"""
灾害类型识别任务 oversample 采样

将 Disaster Type Recognition 任务（1627条）重复采样4倍，
用于续训强化模型的灾害类型判断能力（地震/海啸/滑坡等少样本类别）。
"""
import json
import random
from collections import defaultdict

DATA_PATH = r"C:\vlmdisaster\data\DisasterM3\DisasterM3_Instruct\train_release.json"
OUT_PATH = r"C:\vlmdisaster\data\DisasterM3\DisasterM3_Instruct\train_type_oversample.json"

# 重复倍数
MULTIPLIER = 4


def main():
    random.seed(42)
    
    print("=" * 55)
    print("  灾害类型识别 oversample ♪")
    print("=" * 55)
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 分离灾害类型识别任务和其他任务
    type_rec = [r for r in data if r.get("task") == "Disaster Type Recognition"]
    others = [r for r in data if r.get("task") != "Disaster Type Recognition"]
    
    print(f"  灾害类型识别: {len(type_rec)} 条 × {MULTIPLIER} 倍 = {len(type_rec)*MULTIPLIER} 条")
    print(f"  其他任务: {len(others)} 条（少量采样4000条防遗忘）")
    
    # 灾害类型识别重复采样
    oversampled = type_rec * MULTIPLIER
    
    # 其他任务少量采样（防灾难性遗忘，不占大头）
    others_sampled = random.sample(others, min(4000, len(others)))
    
    # 合并
    combined = oversampled + others_sampled
    random.shuffle(combined)
    
    print(f"  总计: {len(combined)} 条")
    
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    
    print(f"  已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()


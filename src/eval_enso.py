# -*- coding: utf-8 -*-
"""
ENSO 识别快速验收脚本

用训练集外的日期测试（或随机抽几张），让模型判断海温异常图对应相位。
"""
import sys
import io
import os
import glob
import random

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = r"C:\vlmdisaster\models\Qwen3-VL-4B\models\Qwen--Qwen3-VL-4B-Instruct\snapshots\master"
LORA_PATH = r"C:\vlmdisaster\outputs\qwen3vl-thermal-enso"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference.vlm_inference import DisasterVLM

PROMPT = ("Analyze the sea surface temperature anomaly pattern in the tropical Pacific Ocean. "
          "Determine the ENSO phase: is it El Nino, La Nina, or neutral?")

# 测试日期（选不同相位的代表日）
TEST_DAYS = [
    ("20251115", "La Nina"),   # 2025-11 拉尼娜
    ("20260415", "El Nino"),   # 2026-04 厄尔尼诺
    ("20250915", "neutral"),   # 2025-09 中性
    ("20260615", "El Nino"),   # 2026-06 厄尔尼诺
]

def main():
    print("=" * 60)
    print("  ENSO 识别验收（qwen3vl-thermal-enso）")
    print("=" * 60)

    vlm = DisasterVLM(MODEL_PATH, LORA_PATH)

    img_dir = r"C:\vlmdisaster\datasets\03_thermal_infrared\enso\images"
    correct = 0
    for day, true_phase in TEST_DAYS:
        png = os.path.join(img_dir, f"{day}.png")
        if not os.path.exists(png):
            print(f"[跳过] {day} 无图片")
            continue
        print(f"\n[{day}] 正确={true_phase}")
        result = vlm.predict([png], prompt=PROMPT, max_new_tokens=100)
        print(f"  模型: {result}")

        # 判断
        rl = result.lower()
        if true_phase == "La Nina" and "la nina" in rl:
            correct += 1
            print("  ✅")
        elif true_phase == "El Nino" and "el nino" in rl:
            correct += 1
            print("  ✅")
        elif true_phase == "neutral" and ("neutral" in rl or "neither" in rl):
            correct += 1
            print("  ✅")
        else:
            print("  ❌")

    print(f"\n结果: {correct}/{len(TEST_DAYS)}")

if __name__ == "__main__":
    main()

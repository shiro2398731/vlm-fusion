"""
批量灾害识别评估脚本（双时相版）

关键改进：
1. 使用 pre + post 双时相图像（对齐训练数据分布）
2. 使用英文简洁 prompt（对齐训练格式）
3. 自动根据 post 文件名推导 pre 文件名（post → pre）
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.vlm_inference import DisasterVLM

# 对齐训练的英文 prompt
PROMPT = "What kind of calamity has this area experienced? Please identify the disaster type and describe the affected area briefly."


def find_pre_image(post_path: str) -> str:
    """根据 post 文件名推导 pre 文件名"""
    name = os.path.basename(post_path)
    pre_name = name.replace("post_disaster", "pre_disaster").replace("post_", "pre_")
    pre_path = os.path.join(os.path.dirname(post_path), pre_name)
    if os.path.exists(pre_path):
        return pre_path
    return None


def batch_evaluate(
    image_dir: str,
    model_path: str,
    lora_path: str,
    train_images_dir: str,
    output_file: str = "batch_eval_results.txt",
):
    """批量双图评估"""
    
    exts = ("*.png", "*.jpg", "*.jpeg")
    post_files = []
    for ext in exts:
        post_files.extend(glob.glob(os.path.join(image_dir, ext)))
    post_files.sort()
    
    print("=" * 60)
    print(f"  批量灾害识别评估（双时相版）")
    print(f"  图像数量: {len(post_files)}")
    print("=" * 60)
    
    vlm = DisasterVLM(model_path, lora_path)
    
    results = []
    for i, post_path in enumerate(post_files, 1):
        name = os.path.basename(post_path)
        
        pre_path = find_pre_image(post_path)
        if pre_path is None:
            train_pre = os.path.join(train_images_dir, os.path.basename(post_path).replace("post_disaster", "pre_disaster").replace("post_", "pre_"))
            if os.path.exists(train_pre):
                pre_path = train_pre
        
        print(f"\n[{i}/{len(post_files)}] {name}")
        print("-" * 60)
        
        if pre_path:
            print(f"  [双图] pre+post")
            result = vlm.predict([pre_path, post_path], prompt=PROMPT)
        else:
            print(f"  [单图] 未找到pre图，仅用post")
            result = vlm.predict([post_path], prompt=PROMPT)
        
        results.append(f"{name}\t{result}")
        print(result)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(results))
    
    print("\n" + "=" * 60)
    print(f"  评估完成！结果: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=r"datasets\test_images")
    parser.add_argument("--train-images", default=r"C:\vlmdisaster\data\DisasterM3\DisasterM3_Instruct\train_images")
    parser.add_argument("--model", default="C:/vlmdisaster/models/Qwen3-VL-4B/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master")
    parser.add_argument("--lora", default="C:/vlmdisaster/outputs/qwen3vl-disaster-full")
    parser.add_argument("--output", default="C:/vlmdisaster/batch_eval_results.txt")
    args = parser.parse_args()
    
    batch_evaluate(args.image_dir, args.model, args.lora, args.train_images, args.output)


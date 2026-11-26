"""
DisasterM3 → LLaMA-Factory ShareGPT 多模态格式转换

DisasterM3_Instruct 结构:
  train_release.json   — 123k 指令对
  train_images.zip      — 双时相(pre/post)卫星图像
  box_train_images.zip  — 边界框
  masks.zip             — 分割掩码

LLaMA-Factory ShareGPT mllm 格式:
  {
    "conversations": [
      {"from": "human", "value": "<image><image>指令"},
      {"from": "gpt", "value": "回答"}
    ],
    "images": ["pre.png", "post.png"]
  }
"""
import json
import os
import sys
import argparse
from pathlib import Path

# Windows GBK 控制台兼容
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def extract_zips(data_dir: str):
    """解压数据集中的zip文件"""
    import zipfile
    data_path = Path(data_dir)
    zips = list(data_path.glob("*.zip"))
    
    for zp in zips:
        extract_to = data_path
        print(f"[*] 解压 {zp.name} → {extract_to}")
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(extract_to)
        print(f"    ✓ 完成 ({len(zf.namelist())} 个文件)")

def convert_to_sharegpt(records, image_base_dir: str, max_samples: int = None):
    """
    将 DisasterM3 记录转为 LLaMA-Factory ShareGPT mllm 格式。
    每条含双时相图像（pre + post）。
    """
    converted = []
    
    for i, record in enumerate(records):
        if max_samples and i >= max_samples:
            break
        
        try:
            # 兼容两种图像格式
            if "pre_image_path" in record:
                # 双时相图像（大多数任务）
                pre_img = os.path.join(image_base_dir, record["pre_image_path"])
                post_img = os.path.join(image_base_dir, record["post_image_path"])
                images = [pre_img, post_img]
                image_tag = "<image><image>"
            elif "image_path" in record:
                # 单图像（relational reasoning）
                # 标注路径可能误含 train_images 前缀，修正为独立 box_train_images 目录
                img_rel = record["image_path"].replace(
                    "train_images\\box_train_images", "box_train_images"
                ).replace("train_images/box_train_images", "box_train_images")
                img = os.path.join(image_base_dir, img_rel)
                images = [img]
                image_tag = "<image>"
            else:
                print(f"[!] 跳过第 {i} 条: 无图像路径")
                continue
            
            prompt = record["prompts"]
            # 不同任务用不同字段作为答案
            answer = record.get("training_answer") or record.get("ground_truth", "")
            
            human_msg = f"{image_tag}{prompt}"
            
            # 可选：附加选项
            if "options_str" in record and record["options_str"]:
                human_msg += f"\n\n{record['options_str']}"
            
            sample = {
                "conversations": [
                    {"from": "human", "value": human_msg},
                    {"from": "gpt", "value": answer}
                ],
                "images": images
            }
            
            converted.append(sample)
            
        except Exception as e:
            print(f"[!] 跳过第 {i} 条: {e}")
            continue
    
    return converted

def save_for_llama_factory(data, output_dir: str, filename: str = "disaster_m3_train.json"):
    """保存为 LLaMA-Factory 兼容格式 + dataset_info.json"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[✓] 数据: {output_path} ({len(data)} 条)")
    
    info = {
        "disaster_m3_train": {
            "file_name": filename,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt"
            }
        }
    }
    
    info_path = os.path.join(output_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(info)
        info = existing
    
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[✓] 配置: {info_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="DisasterM3 → LLaMA-Factory")
    parser.add_argument("--data_dir", default=r"C:\vlmdisaster\data\DisasterM3\DisasterM3_Instruct")
    parser.add_argument("--output_dir", default=r"C:\vlmdisaster\LLaMA-Factory\data")
    parser.add_argument("--input", default="train_balanced_25000.json", help="输入JSON文件名")
    parser.add_argument("--max_samples", type=int, default=None, help="限制样本数(测试用)")
    parser.add_argument("--skip_extract", action="store_true", help="跳过解压")
    args = parser.parse_args()
    
    print("=" * 55)
    print("  DisasterM3 → LLaMA-Factory 数据转换 ♪")
    print("=" * 55)
    
    # 1. 解压
    if not args.skip_extract:
        extract_zips(args.data_dir)
    
    # 2. 加载
    json_path = os.path.join(args.data_dir, args.input)
    print(f"[*] 加载 {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"    共 {len(records)} 条记录")
    
    # 3. 转换
    converted = convert_to_sharegpt(records, args.data_dir, args.max_samples)
    
    # 4. 保存
    save_for_llama_factory(converted, args.output_dir)
    
    print("=" * 55)
    print("  转换完成! Step 4 就绪 ♪")
    print("=" * 55)

if __name__ == "__main__":
    main()


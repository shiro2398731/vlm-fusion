"""
VLM 灾害识别推理模块

加载 Qwen3-VL-4B + LoRA 微调权重，对卫星遥感图像进行灾害识别。
"""
import os
import sys
from typing import List, Optional

# Windows GBK 控制台兼容：统一 UTF-8 输出，避免 ✓/♪ 等字符编码崩溃
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# 灾害识别提示词模板
DISASTER_DETECTION_PROMPT = (
    "你是一个专业的灾害监测专家。请仔细观察这张卫星遥感图像，"
    "完成以下任务：\n"
    "1. 判断图像中是否存在自然灾害（洪水、地震、飓风、海啸、火山、野火、山体滑坡等）\n"
    "2. 如果存在，指出灾害类型\n"
    "3. 描述受灾区域的位置和范围\n"
    "4. 评估灾害严重程度（轻微/中等/严重/毁灭性）\n"
    "5. 描述可见的建筑或基础设施损坏情况"
)


class DisasterVLM:
    """灾害识别视觉语言模型"""

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        quantize_4bit: bool = True,
    ):
        self.device = device
        
        print(f"[*] 加载 VLM 模型: {model_path}")
        
        # 4bit 量化配置（节省显存）
        quant_config = None
        if quantize_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载 LoRA 微调权重
        if lora_path and os.path.exists(lora_path):
            print(f"[*] 加载 LoRA 权重: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        
        self.model.eval()
        print("[✓] VLM 模型加载完成")

    def predict(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """
        对卫星图像进行灾害识别。

        Args:
            image_paths: 图像路径列表（支持1张单图或2张双时相图）
            prompt: 自定义提示词（默认使用灾害检测模板）
            max_new_tokens: 最大生成token数
            temperature: 采样温度

        Returns:
            灾害识别结果文本
        """
        if prompt is None:
            prompt = DISASTER_DETECTION_PROMPT
        
        # 构建多模态消息
        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": f"file://{img_path}"})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # 处理图像和文本
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
            )
        
        # 解码输出
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response

if __name__ == "__main__":
    """命令行测试入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VLM灾害识别推理")
    parser.add_argument("--image", nargs="+", required=True, help="图像路径（可多张）")
    parser.add_argument("--model", default="C:/vlmdisaster/models/Qwen3-VL-4B/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master")
    parser.add_argument("--lora", default="C:/vlmdisaster/outputs/qwen3vl-disaster")
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()
    
    vlm = DisasterVLM(args.model, args.lora)
    result = vlm.predict(args.image, args.prompt)
    print("\n" + "=" * 55)
    print("  灾害识别结果")
    print("=" * 55)
    print(result)
    print("=" * 55)


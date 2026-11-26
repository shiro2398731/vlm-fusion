# -*- coding: utf-8 -*-
r"""
灾害识别 VLM 演示前端（Gradio）。

功能：
  1. 上传图片（或直接用样例图）；
  2. 手动选择遥感模态（SAR 形变 / 热红外 / 微波 / 自动）；
  3. 一键识别，显示模型答案。
  也支持"其它信息"输入框（如地理位置、日期），拼进 prompt 一起送模型。

模型：Qwen3-VL-4B + 三源 LoRA adapter（常驻显存，只加载一次）

启动：
  python src/app_demo.py
  python src/app_demo.py --lora outputs/qwen3vl-multisource-sar-tir-mw --port 7860
"""
import os
import sys
import glob
import argparse

import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import PeftModel

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_MODEL = "C:/vlmdisaster/models/Qwen3-VL-4B/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"

# 各模态对应的提问（对齐训练格式）
PROMPTS = {
    "SAR 形变（地震/沉降/滑坡）":
        "This is a satellite InSAR LOS displacement map. Classify the ground deformation level "
        "of this area: stable, slight subsidence, or obvious subsidence.",
    "热红外 野火":
        "This is a satellite fire-hotspot map covering one region. "
        "Are there significant active wildfires in this region? Answer Yes or No.",
    "热红外 温度灾害（寒潮/霜冻/热浪）":
        "This is a MODIS land-surface-temperature map (day and night) over a region. "
        "Classify the temperature hazard: cold_wave, frost, heatwave, or normal.",
    "微波 土壤湿度干旱":
        "This is a soil-moisture map over a region. Classify: dry soil, normal soil moisture, "
        "or wet soil?",
    "光学 灾害识别（灾种）":
        "What kind of disaster has this area experienced? Identify the disaster type.",
    "自动（不指定模态）":
        "This is a remote-sensing disaster image. Identify what disaster or hazard is shown, "
        "and describe the key evidence briefly.",
}

MODEL = None
PROC = None


def load_model(lora_path):
    global MODEL, PROC
    print("加载模型（只加载一次）...")
    m = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True)
    if lora_path and os.path.isdir(lora_path):
        m = PeftModel.from_pretrained(m, lora_path)
        print(f"已挂载 LoRA: {lora_path}")
    m = m.to("cuda:0")
    m.eval()
    devs = {p.device.type for p in m.parameters()}
    print(f"模型设备: {devs}")
    p = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    MODEL, PROC = m, p


def predict(image, modality, extra_info):
    import time
    if image is None:
        return "请先上传一张遥感图片。"
    if MODEL is None:
        return "模型尚未就绪。"
    t0 = time.time()
    prompt = PROMPTS.get(modality, PROMPTS["自动（不指定模态）"])
    if extra_info and extra_info.strip():
        prompt += f" Additional context: {extra_info.strip()}"

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}]}]
    t1 = time.time()
    text = PROC.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = PROC(text=[text], images=image_inputs, return_tensors="pt").to("cuda:0")
    t2 = time.time()
    with torch.inference_mode():
        out = MODEL.generate(**inputs, max_new_tokens=40, do_sample=False, use_cache=True)
    t3 = time.time()
    ans = PROC.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    timing = (f"\n\n—— 耗时：预处理 {t2-t1:.1f}s ｜ 生成 {t3-t2:.1f}s ｜ "
              f"图像tokens {inputs['input_ids'].shape[1]} ｜ 共 {t3-t0:.1f}s")
    return ans.strip() + timing


def build_demo(lora_path, samples):
    with gr.Blocks(title="多源遥感灾害识别 Demo") as demo:
        gr.Markdown("## 多源遥感灾害识别 Demo\n"
                    "Qwen3-VL-4B + 三源 LoRA｜SAR 形变 · 热红外 · 微波土壤湿度")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="filepath", label="上传遥感图片", height=360)
                modality = gr.Dropdown(list(PROMPTS.keys()),
                                       value="自动（不指定模态）", label="遥感模态 / 问题类型")
                extra = gr.Textbox(label="附加信息（可选）",
                                   placeholder="如：2020-07 长江中下游 / 震中区域 ...")
                btn = gr.Button("识别", variant="primary")
                gr.Markdown("小提示：**第一张图会慢一些（模型预热），先跑一张热身**，后面就快了。")
            with gr.Column():
                out = gr.Textbox(label="识别结果", lines=8)
                gr.Markdown("**样例图**（点击填入）")
                if samples:
                    gr.Examples(examples=[[s] for s in samples], inputs=[img])
        btn.click(predict, inputs=[img, modality, extra], outputs=[out])
        img.change(lambda: "已载入图片，选好模态后点【识别】", outputs=[out])
    return demo


def collect_samples():
    # 优先用人工挑好的演示样例目录
    demo_dir = r"C:\vlmdisaster\demo_samples"
    if os.path.isdir(demo_dir):
        fs = sorted(glob.glob(os.path.join(demo_dir, "*.png")))
        if fs:
            return fs
    # 回退：从数据集里抽
    roots = [
        r"C:\vlmdisaster\datasets\05_sar_insar\earthquake_deformation",
        r"C:\vlmdisaster\datasets\03_thermal_infrared\wildfire",
        r"C:\vlmdisaster\datasets\04_passive_microwave\drought",
    ]
    samples = []
    for r in roots:
        for sub in ["images", "images_fine"]:
            p = os.path.join(r, sub)
            if os.path.isdir(p):
                fs = glob.glob(os.path.join(p, "*.png"))
                samples += fs[::max(1, len(fs)//3)][:3]
                break
    return samples[:9]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="C:/vlmdisaster/outputs/qwen3vl-multisource-sar-tir-mw")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    load_model(args.lora)
    samples = collect_samples()
    print(f"样例图 {len(samples)} 张")
    demo = build_demo(args.lora, samples)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)

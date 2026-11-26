"""
灾害监测智能体 WebUI（Gradio）

启动: python src/webui/app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from agent.disaster_agent import DisasterAgent


# 配置（可改为从 config.yaml 读取）
MODEL_PATH = "C:/vlmdisaster/models/Qwen3-VL-4B/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"
LORA_PATH = "C:/vlmdisaster/outputs/qwen3vl-disaster"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# 全局智能体实例（懒加载）
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = DisasterAgent(
            vlm_model_path=MODEL_PATH,
            lora_path=LORA_PATH,
            deepseek_api_key=DEEPSEEK_KEY,
        )
    return _agent

def analyze_images(image1, image2, lat, lon, with_advice):
    """WebUI 处理函数"""
    if image1 is None:
        return "请至少上传一张卫星图像", "", ""
    
    agent = get_agent()
    
    image_paths = [image1]
    if image2 is not None:
        image_paths.append(image2)
    
    result = agent.analyze(
        image_paths,
        lat=float(lat) if lat else None,
        lon=float(lon) if lon else None,
        with_advice=with_advice,
    )
    
    return result["disaster"], result["location"], result["advice"]


# 构建界面
with gr.Blocks(title="灾害监测智能体") as demo:
    gr.Markdown("# 🌋 灾害监测智能体")
    gr.Markdown("上传卫星遥感图像，自动识别自然灾害并生成防灾建议")
    
    with gr.Row():
        image1 = gr.Image(type="filepath", label="灾前/单张图像")
        image2 = gr.Image(type="filepath", label="灾后图像（可选）")
    
    with gr.Row():
        lat = gr.Textbox(label="纬度（可选）", placeholder="如 31.23")
        lon = gr.Textbox(label="经度（可选）", placeholder="如 104.56")
    
    with_advice = gr.Checkbox(label="生成防灾建议（需DeepSeek API Key）", value=True)
    analyze_btn = gr.Button("开始分析", variant="primary")
    
    with gr.Row():
        disaster_out = gr.Textbox(label="灾害识别结果", lines=6)
    
    with gr.Row():
        location_out = gr.Textbox(label="受灾位置", lines=2)
        advice_out = gr.Textbox(label="防灾建议", lines=10)
    
    analyze_btn.click(
        analyze_images,
        inputs=[image1, image2, lat, lon, with_advice],
        outputs=[disaster_out, location_out, advice_out],
    )


if __name__ == "__main__":
    demo.launch()


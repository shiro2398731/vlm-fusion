# -*- coding: utf-8 -*-
r"""
灾害识别智能体 · 界面（Gradio）

## 与旧 app_demo.py 的区别

旧版把六段提问**硬编码**在界面里，和训练用的提问是两套文字，必然漂移。
本界面不写任何提问，全部由 agent.run() 从 router（训练数据）里读。

## 模型加载策略

点第一次「开始识别」时才加载基座，**一次性挂载全部可用 adapter**（约 145 MB），
之后每次识别只做 set_adapter 切换，不重载。

## 用法

  .\venv\Scripts\python.exe src\webui\app_v2.py
  .\venv\Scripts\python.exe src\webui\app_v2.py --port 7861

## 不需要显卡的自检

界面勾选「模拟结果」即可走完整条链路（不加载模型），用来验证界面接线。
"""

import argparse
import json
import os
import sys

import gradio as gr

ROOT = r"C:\vlmdisaster"
sys.path.insert(0, os.path.join(ROOT, "src", "agent"))

import agent  # noqa: E402
import router  # noqa: E402

RUNNER = None


def get_runner():
    """基座 + 全部 adapter，只加载一次"""
    global RUNNER
    if RUNNER is None:
        from vlm_runner import VLMRunner
        table = router.load()
        els = sorted(router.available(table))
        RUNNER = VLMRunner(els, table)
    return RUNNER


def source_choices():
    table = router.load()
    out = []
    for code, els in router.categories_in_use(table).items():
        cn = next((v[1] for k, v in router.CATEGORIES.items() if v[0] == code), code)
        out.append("%s（%s）" % (code, cn))
    return out


def element_choices():
    """(显示名, 元素名) —— 显示名带上「这张图长什么样」，方便用户选择"""
    table = router.load()
    out = []
    for el in sorted(router.available(table)):
        note = table[el].get("note") or ""
        out.append(("%s · %s" % (el, note) if note else el, el))
    return out


def _num(v):
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def ui_run(image, source, elements, lat, lon, bbox, use_advice, mock, api_key):
    if not image:
        return "请先上传影像", ""
    code = (source or "").split("（")[0].strip()
    try:
        runner = None if mock else get_runner()
        report = agent.run(
            image, source=code, elements=elements or "",
            lat=_num(lat), lon=_num(lon), bbox=(bbox or "").strip(),
            no_advice=not use_advice, api_key=api_key or "",
            mock=bool(mock), runner=runner, verbose=False)
    except Exception as e:
        return "✗ 运行异常：%s" % e, ""
    return agent.render(report), json.dumps(report, ensure_ascii=False, indent=1)


def build():
    with gr.Blocks(title="多源遥感灾害识别") as demo:
        gr.Markdown(
            "## 多源遥感灾害识别 · 智能体\n"
            "上传影像 → 选遥感源 → 自动选出该源下的候选元素，逐个判识 → 定位 → 处置建议\n\n"
            "> 没有显卡时勾选「模拟结果」，可以走完整条链路验证接线。")

        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(label="影像", type="filepath")
                src = gr.Dropdown(choices=source_choices(), value=source_choices()[0],
                                  label="遥感源")
                els = gr.CheckboxGroup(choices=element_choices(), value=[],
                                       label="指定元素（留空则按源自动路由）")
                with gr.Row():
                    lat = gr.Textbox(label="纬度（可选）", placeholder="30.5")
                    lon = gr.Textbox(label="经度（可选）", placeholder="104.1")
                bbox = gr.Textbox(label="影像范围 bbox（可选）",
                                  placeholder="minlon,minlat,maxlon,maxlat")
                with gr.Row():
                    use_advice = gr.Checkbox(value=True, label="生成处置建议")
                    mock = gr.Checkbox(value=False, label="模拟结果（不加载模型）")
                api_key = gr.Textbox(label="DeepSeek Key（可留空，读环境变量）", type="password")
                btn = gr.Button("开始识别", variant="primary")

            with gr.Column(scale=1):
                out_md = gr.Markdown(label="结果")
                out_json = gr.Code(label="结构化 JSON", language="json")

        btn.click(ui_run,
                  inputs=[img, src, els, lat, lon, bbox, use_advice, mock, api_key],
                  outputs=[out_md, out_json])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo = build()
    demo.launch(server_name="127.0.0.1", server_port=a.port, show_api=False)


if __name__ == "__main__":
    main()

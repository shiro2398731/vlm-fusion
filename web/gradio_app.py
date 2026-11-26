# -*- coding: utf-8 -*-
r"""
灾害识别 · Gradio 界面（第二前端）

与 `web/index.html` 共用同一套后端接口（默认 http://127.0.0.1:8078），
不自己加载模型，所以只要服务在跑，这个界面就能用。

用法：
    # 1) 先启动主服务
    python -m uvicorn scripts.serve:app --host 127.0.0.1 --port 8078
    # 2) 再启动本界面
    python web/gradio_app.py --port 7860 --api http://127.0.0.1:8078
"""
import argparse
import io
import json
import os
import sys

import requests
import gradio as gr

API = os.environ.get("VLM_API", "http://127.0.0.1:8078")
STATE = {}  # element -> prediction（用于汇总研判）


def _get(path):
    return requests.get(API + path, timeout=20).json()


def catalog_groups():
    try:
        return _get("/api/catalog").get("groups", [])
    except Exception as e:
        return [{"label": "（连接失败：%s）" % e, "elements": []}]


GROUPS = []


def refresh():
    global GROUPS
    GROUPS = catalog_groups()


def group_choices():
    if not GROUPS:
        refresh()
    return [g["label"] for g in GROUPS]


def element_choices(group_label):
    for g in GROUPS:
        if g["label"] == group_label:
            return ["%s（%s）" % (e.get("zh", e["name"]), e["name"]) for e in g.get("elements", [])]
    return []


def samples_of(group_label, element_label):
    el = element_label.split("（")[-1].rstrip("）")
    for g in GROUPS:
        if g["label"] != group_label:
            continue
        for e in g.get("elements", []):
            if e["name"] == el:
                out = []
                for s in e.get("samples", []):
                    tag = s.get("tag") or ""
                    label = os.path.basename(s.get("url", ""))
                    out.append("%s%s" % (label, ("｜" + tag) if tag else ""))
                return out
    return []


def sample_url(group_label, element_label, sample_label):
    el = element_label.split("（")[-1].rstrip("）")
    for g in GROUPS:
        if g["label"] != group_label:
            continue
        for e in g.get("elements", []):
            if e["name"] != el:
                continue
            for s in e.get("samples", []):
                if os.path.basename(s.get("url", "")) == sample_label.split("｜")[0]:
                    return s.get("url", "")
    return ""


def status_md():
    try:
        st = _get("/api/status")
        e2e = st.get("last_e2e") or {}
        return "服务：%s　模型：%s　端到端：%s/%s" % (
            API, "已暖机" if st.get("model_loaded") else "未暖机",
            e2e.get("ok", "?"), e2e.get("total", "?"))
    except Exception as e:
        return "服务不可用：%s" % e


def run_predict(group_label, element_label, sample_label, upload_file):
    el = element_label.split("（")[-1].rstrip("）")
    if not el:
        return None, "先选元素", status_md()
    try:
        if upload_file:
            data = open(upload_file, "rb").read()
            name = os.path.basename(upload_file)
        elif sample_label:
            u = sample_url(group_label, element_label, sample_label)
            if not u:
                return None, "找不到样例", status_md()
            r = requests.get(API + u, timeout=60)
            data = r.content
            name = os.path.basename(u)
        else:
            return None, "选一个样例，或上传一张渲染图", status_md()
        files = {"file": (name, data, "image/png")}
        r = requests.post(API + "/api/predict", data={"element": el}, files=files, timeout=300)
        d = r.json()
        if "error" in d:
            return None, "识别失败：%s" % d["error"], status_md()
        pred = d.get("prediction") or {}
        STATE[el] = pred
        rep = d.get("report") or {}
        md = "**%s**　`%s`\n\n" % (element_label, el)
        if isinstance(rep, dict) and rep.get("summary"):
            md += rep["summary"] + "\n\n"
        if isinstance(rep, dict):
            for k in ("risk", "advice", "suggestion", "detail"):
                if rep.get(k):
                    md += "- **%s**：%s\n" % (k, rep[k])
        md += "\n已识别元素：%d 个（可点「汇总研判」）" % len(STATE)
        return pred, md, status_md()
    except Exception as e:
        return None, "请求失败：%s" % e, status_md()


def run_aggregate():
    if not STATE:
        return "还没有识别结果，先跑一个元素", status_md()
    items = []
    for el, p in STATE.items():
        q = dict(p)
        q.setdefault("element", el)
        items.append(q)
    try:
        r = requests.post(API + "/api/report_aggregate",
                          json={"items": items, "backend": "template"}, timeout=120)
        rep = r.json().get("report") or {}
        md = ""
        if rep.get("summary"):
            md += rep["summary"] + "\n\n"
        for k in ("risk", "advice", "suggestion"):
            if rep.get(k):
                md += "- **%s**：%s\n" % (k, rep[k])
        if not md:
            md = "```json\n%s\n```" % json.dumps(rep, ensure_ascii=False, indent=2)
        return md, status_md()
    except Exception as e:
        return "汇总失败：%s" % e, status_md()


def clear_state():
    STATE.clear()
    return None, "已清空", status_md()


def build():
    global GROUPS
    GROUPS = catalog_groups()
    with gr.Blocks(title="遥感灾害识别 · Gradio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 遥感灾害识别 · Gradio 界面\n"
                    "与 `web/index.html` 共用同一后端；先启动服务，再打开本界面。")
        st = gr.Markdown(status_md())

        with gr.Row():
            with gr.Column(scale=1):
                g = gr.Dropdown(choices=group_choices(), value=(group_choices() or [None])[0],
                                label="采集方式")
                el = gr.Dropdown(choices=element_choices(g.value), value=None, label="元素")
                samp = gr.Dropdown(choices=[], value=None, label="样例（训练集之外优先）")
                up = gr.Image(type="filepath", label="或上传渲染图", height=220)
                with gr.Row():
                    btn = gr.Button("识别", variant="primary")
                    agg = gr.Button("汇总研判")
                    clr = gr.Button("清空")
            with gr.Column(scale=1):
                out_json = gr.JSON(label="结构化判定")
                out_md = gr.Markdown(label="中文研判")

        g.change(fn=lambda gv: (gr.update(choices=element_choices(gv), value=None),
                                gr.update(choices=[], value=None)),
                 inputs=g, outputs=[el, samp])
        el.change(fn=lambda gv, ev: gr.update(choices=samples_of(gv, ev or ""), value=None),
                  inputs=[g, el], outputs=samp)
        btn.click(fn=run_predict, inputs=[g, el, samp, up], outputs=[out_json, out_md, st])
        agg.click(fn=run_aggregate, outputs=[out_md, st])
        clr.click(fn=clear_state, outputs=[out_json, out_md, st])
        gr.Markdown("接口：`/api/catalog`、`/api/predict`、`/api/report_aggregate`、`/api/status`。"
                    "上传图需为渲染后的指数图（带色标）。")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=API)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    API = args.api.rstrip("/")
    build().launch(server_name=args.host, server_port=args.port)

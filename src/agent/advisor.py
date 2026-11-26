# -*- coding: utf-8 -*-
r"""
研判层：结构化识别结果 + 地名 → 处置建议

## 与旧 llm_advisor.py 的区别

旧版把 VLM 的一段中文散文丢给大模型，物理量全丢了。
这一版**直接喂结构化 JSON**，模型能看到 detected / severity / 数值 / 判据，
建议才有依据。

## 输出约定

固定三段，方便直接放论文和界面：
  一、研判    引用识别到的灾害类型、等级、物理量
  二、处置    紧急避险 / 人员疏散 / 物资准备 / 后续恢复
  三、局限    明确说明本结果来自遥感影像识别，不含实时灾情与人口信息

## Key

优先用参数，其次环境变量 DEEPSEEK_API_KEY。没有 key 时返回 (None, 原因)，不伪造内容。
"""

import json
import os
from typing import List, Optional, Tuple

SYSTEM_PROMPT = (
    "你是一名防灾减灾专家。你会收到一份来自遥感影像自动识别的结构化结果，"
    "包含灾害类型、等级、从图上读出的物理量与判据。请据此给出处置建议。\n"
    "要求：\n"
    "1. 分三段输出，标题依次为「一、研判」「二、处置」「三、局限」；\n"
    "2. 研判段必须引用给出的物理量数值与等级，不要泛泛而谈；\n"
    "3. 处置段给出可执行动作，区分紧急避险 / 人员疏散 / 物资准备 / 后续恢复；\n"
    "4. 局限段必须声明：结果来自单幅遥感影像的模型识别，不含实时灾情、"
    "人口分布与地面核查，需人工复核；\n"
    "5. 不要编造未提供的灾情数据（伤亡、面积、经济损失）。"
)


def build_prompt(detected: List[dict], location: str, image_name: str) -> str:
    return (
        "影像：%s\n"
        "位置：%s\n"
        "识别结果（JSON）：\n%s\n\n"
        "请按约定格式给出处置建议。"
        % (image_name, location,
           json.dumps(detected, ensure_ascii=False, indent=1))
    )


def advise(detected: List[dict], location: str, image_name: str,
           api_key: str = "", model: str = "deepseek-chat",
           base_url: str = "https://api.deepseek.com",
           max_tokens: int = 1024, temperature: float = 0.3) -> Tuple[Optional[str], str]:
    """
    返回 (建议文本, 状态说明)。没有 key 或调用失败时文本为 None。
    """
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return None, "未配置 DEEPSEEK_API_KEY，跳过建议生成"
    if not detected:
        return None, "未检出灾害，不生成建议"

    try:
        from openai import OpenAI
    except ImportError:
        return None, "未安装 openai 包（pip install openai）"

    try:
        client = OpenAI(api_key=key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(detected, location, image_name)},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content, "ok"
    except Exception as e:
        return None, "建议生成失败：%s" % e

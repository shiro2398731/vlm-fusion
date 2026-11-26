"""
DeepSeek 大语言模型防灾建议模块

使用 DeepSeek API（兼容 OpenAI SDK）生成防灾/抗灾建议。
API Key 申请: https://platform.deepseek.com/
"""
from typing import Optional
from openai import OpenAI


class LLMAdvisor:
    """防灾建议生成器（DeepSeek）"""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_advice(
        self,
        disaster_info: str,
        location: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """
        根据灾害识别结果和位置生成防灾建议。

        Args:
            disaster_info: VLM识别的灾害信息
            location: 受灾位置
            max_tokens: 最大token数
            temperature: 采样温度

        Returns:
            防灾/抗灾建议文本
        """
        system_prompt = (
            "你是一个专业的防灾减灾专家。请根据给定的灾害信息，"
            "提供专业、具体、可操作的防灾或抗灾建议。"
            "建议应包含：紧急避险、人员疏散、物资准备、后续恢复等方面。"
        )
        user_prompt = (
            f"受灾位置：{location}\n"
            f"灾害信息：{disaster_info}\n\n"
            f"请给出防灾/抗灾建议。"
        )
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[!] 防灾建议生成失败: {e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--disaster", required=True, help="灾害信息")
    parser.add_argument("--location", default="未知位置")
    args = parser.parse_args()
    
    advisor = LLMAdvisor(args.api_key)
    print(advisor.generate_advice(args.disaster, args.location))


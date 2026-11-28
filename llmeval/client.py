# -*- coding: utf-8 -*-
"""评估用 LLM 客户端：本地 transformers 或 OpenAI 兼容 API，二者统一接口。

用法：
    from llmeval import LLMClient
    c = LLMClient(backend="api", base_url="https://api.example.com/v1",
                  api_key="...", model="gpt-x")
    print(c.chat("请判断以下判读是否合理：..."))
    c2 = LLMClient(backend="local", model_path=r"C:\\models\\Qwen3-VL-4B", device="cuda")
"""
from __future__ import annotations

import json
import os
from typing import List, Optional


class LLMClient:
    def __init__(self, backend: str = "api", **cfg) -> None:
        self.backend = backend
        self.cfg = cfg
        self._model = None
        self._tok = None

    # ---------- API ----------
    def _chat_api(self, messages: List[dict], temperature: float, max_tokens: int) -> str:
        import requests
        base = self.cfg.get("base_url", "").rstrip("/")
        key = self.cfg.get("api_key") or os.environ.get("LLM_API_KEY", "")
        model = self.cfg.get("model", "")
        r = requests.post(base + "/chat/completions",
                          headers={"Authorization": "Bearer " + key,
                                   "Content-Type": "application/json"},
                          json={"model": model, "messages": messages,
                                "temperature": temperature, "max_tokens": max_tokens},
                          timeout=self.cfg.get("timeout", 120))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # ---------- 本地 ----------
    def _load_local(self):
        if self._model is not None:
            return
        import torch
        path = self.cfg.get("model_path")
        if not path:
            raise ValueError("local backend requires model_path")
        device = self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Qwen3-VL 等视觉语言模型
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self._tok = AutoProcessor.from_pretrained(path)
            self._model = AutoModelForImageTextToText.from_pretrained(
                path, torch_dtype="auto", device_map=device)
        except Exception:
            # 纯文本因果模型
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype="auto", device_map=device)
        self._model.eval()

    def _chat_local(self, messages: List[dict], temperature: float, max_tokens: int) -> str:
        self._load_local()
        import torch
        tok = getattr(self._tok, "tokenizer", self._tok)
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inp, max_new_tokens=max_tokens,
                                       do_sample=temperature > 0,
                                       temperature=max(temperature, 1e-5))
        return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ---------- 统一 ----------
    def chat(self, prompt: str, system: Optional[str] = None,
             temperature: float = 0.0, max_tokens: int = 512) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if self.backend == "api":
            return self._chat_api(messages, temperature, max_tokens)
        if self.backend == "local":
            return self._chat_local(messages, temperature, max_tokens)
        raise ValueError("unknown backend: %s" % self.backend)

    # ---------- 视觉输入 ----------
    def chat_image(self, prompt: str, image_path: str, system: Optional[str] = None,
                   temperature: float = 0.0, max_tokens: int = 512) -> str:
        """带一张图的对话；API 走 image_url(base64)，本地走 Qwen3-VL 处理器。"""
        if self.backend == "api":
            import base64
            b64 = base64.b64encode(open(image_path, "rb").read()).decode()
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b64}}]
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": content})
            return self._chat_api(messages, temperature, max_tokens)

        if self.backend == "local":
            from PIL import Image
            self._load_local()
            import torch
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}]}]
            tok = getattr(self._tok, "tokenizer", self._tok)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            proc = self._tok
            if hasattr(proc, "image_processor"):
                inp = proc(text=[text], images=[Image.open(image_path).convert("RGB")],
                           return_tensors="pt")
            else:
                inp = tok(text, return_tensors="pt")
            inp = {k: v.to(self._model.device) for k, v in inp.items()}
            with torch.no_grad():
                out = self._model.generate(**inp, max_new_tokens=max_tokens,
                                           do_sample=temperature > 0,
                                           temperature=max(temperature, 1e-5))
            return tok.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
        raise ValueError("unknown backend: %s" % self.backend)

    @classmethod
    def from_config(cls, path: str) -> "LLMClient":
        import yaml
        cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
        backend = cfg.pop("backend", "api")
        return cls(backend=backend, **cfg)

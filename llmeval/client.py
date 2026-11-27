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
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = self.cfg.get("model_path")
        if not path:
            raise ValueError("local backend requires model_path")
        device = self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype="auto", device_map=device)
        self._model.eval()

    def _chat_local(self, messages: List[dict], temperature: float, max_tokens: int) -> str:
        self._load_local()
        import torch
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inp, max_new_tokens=max_tokens,
                                       do_sample=temperature > 0,
                                       temperature=max(temperature, 1e-5))
        return self._tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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

    @classmethod
    def from_config(cls, path: str) -> "LLMClient":
        import yaml
        cfg = yaml.safe_load(open(path, encoding="utf-8")) or {}
        return cls(backend=cfg.get("backend", "api"), **cfg)

# -*- coding: utf-8 -*-
r"""
VLM 执行器：一个基座 + 多个 LoRA adapter 热切换

## 为什么不逐个加载

交接文档记过一个坑：PeftModel.delete_adapter("snow_fsc-json") 名字带连字符清理不干净，
下一个 adapter 加载时报 "Already found a peft_config"。

## 本模块的做法

反过来：**一次全部加载**，adapter 名统一成 a0/a1/a2（不带连字符），
切换时只调 set_adapter，不删除、不重载。

26 个 adapter × 5.6 MB ≈ 145 MB，内存压力可以忽略，切换是微秒级。

## 用法

  runner = VLMRunner(elements=["wildfire", "lst"])
  out = runner.predict("wildfire", r"C:\...\img.png")
  # out -> {"raw": "...", "json": {...} or None}
"""

import json
import os
import sys
from typing import Dict, List, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\vlmdisaster"
MODEL_PATH = os.path.join(
    ROOT, "models", "Qwen3-VL-4B", "models",
    "Qwen--Qwen3-VL-4B-Instruct", "snapshots", "master")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from eval_single import parse_json  # noqa: E402  与评估口径一致


class VLMRunner:
    def __init__(self, elements: List[str], router_table: Dict, model_path: str = MODEL_PATH,
                 quantize_4bit: bool = True):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
        from peft import PeftModel

        self.table = router_table
        self.elements = [e for e in elements
                         if os.path.exists(os.path.join(router_table[e]["adapter"],
                                                        "adapter_model.safetensors"))]
        if not self.elements:
            raise ValueError("没有任何可用的 adapter")

        print("[*] 加载基座模型 ...", flush=True)
        qc = None
        if quantize_4bit:
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, quantization_config=qc, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True)
        self.proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.proc.tokenizer.padding_side = "left"

        print("[*] 挂载 %d 个 adapter ..." % len(self.elements), flush=True)
        self.names = {}
        model = None
        for i, el in enumerate(self.elements):
            name = "a%d" % i
            path = router_table[el]["adapter"]
            if model is None:
                model = PeftModel.from_pretrained(base, path, adapter_name=name)
            else:
                model.load_adapter(path, adapter_name=name)
            self.names[el] = name
        model.eval()
        self.model = model
        self.base = base
        self._torch = torch
        print("[✓] 就绪：%s" % ", ".join(self.elements), flush=True)

    def predict(self, element: str, image_path: str, max_new_tokens: int = 200) -> dict:
        from qwen_vl_utils import process_vision_info

        if element not in self.names:
            raise KeyError("未加载的元素: %s" % element)
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        question = self.table[element]["question"].replace("<image>", "").strip()
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": "file://%s" % image_path},
            {"type": "text", "text": question}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(msgs)
        inp = self.proc(text=[text], images=images, videos=videos,
                        return_tensors="pt").to(self.base.device)

        self.model.set_adapter(self.names[element])
        with self._torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens,
                                      do_sample=False, repetition_penalty=1.05)
        new = out[0][inp["input_ids"].shape[1]:]
        raw = self.proc.tokenizer.decode(new, skip_special_tokens=True).strip()
        return {"element": element, "raw": raw, "json": parse_json(raw)}

    def close(self):
        del self.model
        del self.base
        import gc
        gc.collect()
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass

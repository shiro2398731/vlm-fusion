# -*- coding: utf-8 -*-
"""projector 可行性：服务运行中再加载一份 4bit 基座，测显存余量。"""
import os, sys, traceback
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch

MODEL = os.environ.get(
    "VLM_MODEL",
    r"C:\vlmdisaster\models\Qwen3-VL-4B\models\Qwen--Qwen3-VL-4B-Instruct\snapshots\master")

def free():
    return torch.cuda.mem_get_info()[0] / 1e9

print("free GB before: %.2f" % free())
try:
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    m = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qc, torch_dtype=torch.bfloat16, device_map={"": 0})
    print("loaded; free GB after: %.2f" % free())
    print("second 4bit instance OK")
    del m
    torch.cuda.empty_cache()
except Exception as e:
    print("LOAD FAILED:", type(e).__name__, str(e)[:300])
    traceback.print_exc(limit=1)
print("free GB end: %.2f" % free())

# -*- coding: utf-8 -*-
"""projector 训练可行性：一次前向+反向的显存/时间测试（文本样本）。"""
import os, sys, time, traceback
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch

MODEL = os.environ.get(
    "VLM_MODEL",
    r"C:\vlmdisaster\models\Qwen3-VL-4B\models\Qwen--Qwen3-VL-4B-Instruct\snapshots\master")

print("free GB before: %.2f" % (torch.cuda.mem_get_info()[0] / 1e9))
try:
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    m = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qc, torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoProcessor.from_pretrained(MODEL).tokenizer
    text = ("Classify this remote-sensing disaster map and output JSON. "
            '{"disaster_type": "coral_bleaching", "detected": true, "severity": "alert1"}')
    ids = tok(text, return_tensors="pt").input_ids.to(0)
    m.train()
    t0 = time.time()
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print("one fwd+bwd OK: loss %.3f, %.2fs, peak alloc %.2f GB" % (
        float(out.loss), dt, torch.cuda.max_memory_allocated() / 1e9))
except Exception as e:
    print("TRAIN STEP FAILED:", type(e).__name__, str(e)[:200])
    traceback.print_exc(limit=1)
print("free GB end: %.2f" % (torch.cuda.mem_get_info()[0] / 1e9))

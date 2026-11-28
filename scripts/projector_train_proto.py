# -*- coding: utf-8 -*-
"""projector 原型：CLIP 向量 → K 个软 token → Qwen，文本教师强制，训练几步。"""
import json, os, sys, traceback
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch

MODEL = os.environ.get(
    "VLM_MODEL",
    r"C:\vlmdisaster\models\Qwen3-VL-4B\models\Qwen--Qwen3-VL-4B-Instruct\snapshots\master")
DATA = r"C:\vlmdisaster\LLaMA-Factory\data\train_coral_dhw_json_val.json"
K = 4

try:
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from features.extractors import ClipExtractor
    import joblib

    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qc, torch_dtype=torch.bfloat16, device_map={"": 0})
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    for p in model.parameters():
        p.requires_grad_(False)

    H = model.config.text_config.hidden_size
    clip = ClipExtractor(); clip.load()
    projector = torch.nn.Linear(512, K * H).to("cuda")
    opt = torch.optim.AdamW(projector.parameters(), lr=1e-3)

    rows = json.load(open(DATA, encoding="utf-8"))[:3]
    emb_layer = model.get_input_embeddings()
    for step, r in enumerate(rows):
        img = r["image"]
        prompt = r["conversations"][0]["value"].replace("<image>", "").strip()
        answer = r["conversations"][1]["value"]
        from PIL import Image
        inp = clip.proc(images=Image.open(img).convert("RGB"), return_tensors="pt")
        inp = {k: v.to(clip.device) for k, v in inp.items()}
        with torch.no_grad():
            feats = clip.model.get_image_features(pixel_values=inp["pixel_values"])
        vec = feats.pooler_output[0].detach().float().to("cuda")
        soft = projector(vec).view(1, K, H).to(torch.bfloat16)

        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(answer, add_special_tokens=False).input_ids
        ids = torch.tensor([p_ids + a_ids], device=0)
        with torch.no_grad():
            emb = emb_layer(ids)
        inputs_embeds = torch.cat([soft, emb], dim=1)
        attn = torch.ones(1, K + ids.shape[1], dtype=torch.long, device=0)
        labels = ids.clone()
        labels[:, :len(p_ids)] = -100
        labels = torch.cat([torch.full((1, K), -100, dtype=torch.long, device=0), labels], dim=1)

        out = model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
        out.loss.backward()
        opt.step(); opt.zero_grad()
        print("step %d loss %.3f" % (step, float(out.loss)), flush=True)
    print("PROJECTOR PROTOTYPE OK; peak %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
except Exception as e:
    print("PROJECTOR PROTOTYPE FAILED:", type(e).__name__, str(e)[:300])
    traceback.print_exc(limit=2)
print("free GB end: %.2f" % (torch.cuda.mem_get_info()[0] / 1e9))

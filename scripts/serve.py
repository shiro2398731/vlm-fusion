# -*- coding: utf-8 -*-
r"""
遥感灾害识别服务（FastAPI）

功能：
    GET  /                    简单前端（上传图 → 选元素 → 判定 + 研判）
    GET  /api/elements        可用元素（有 adapter 的）
    POST /api/predict         上传图片 + 元素 + 可选经纬度 → 结构化判定 + 研判文字

启动：
    .\venv\Scripts\python.exe -m uvicorn scripts.serve:app --host 127.0.0.1 --port 8000

说明：
    - 基座 4bit 懒加载一次，adapter 按元素缓存
    - 同一进程串行推理（GPU 8GB，不并发）
"""
import glob
import json
import os
import re
import sys
import tempfile
import threading

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:\vlmdisaster\scripts")

from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from report_llm import report as make_report, aggregate_report  # noqa: E402
from reverse_geocode import geocode  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get(
    "VLM_MODEL",
    os.path.join(_REPO, "models", "Qwen3-VL-4B", "models",
                 "Qwen--Qwen3-VL-4B-Instruct", "snapshots", "master"))
OUT_ROOT = os.environ.get("VLM_OUTPUTS", os.path.join(_REPO, "outputs"))
LF_DATA = os.environ.get("VLM_LF_DATA", os.path.join(_REPO, "LLaMA-Factory", "data"))
STATIC = r"C:\vlmdisaster\web"
DATASETS = r"C:\vlmdisaster\datasets"
DEMO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_samples")
APPENDIX = os.environ.get("VLM_APPENDIX",
                          os.path.join(_REPO, "docs", "appendix_elements.md"))
CAPABILITY = os.environ.get("VLM_CAPABILITY",
                            os.path.join(_REPO, "elements_capability.json"))
ARTIFACTS = r"C:\vlmdisaster\处理方案\工程清单_20261125.json"

SOURCE_ZH = {"OPT": "光学", "TIR": "热红外", "MW": "被动微波", "PRECIP": "雷达降水",
             "ALT": "测高", "GRACE": "重力", "SCAT": "散射计", "SPACE": "空间天气",
             "SAR": "SAR"}

app = FastAPI(title="遥感灾害识别")
app.mount("/images", StaticFiles(directory=DATASETS), name="images")
if os.path.isdir(DEMO_ROOT):
    app.mount("/samples", StaticFiles(directory=DEMO_ROOT), name="samples")
_lock = threading.Lock()
_state = {"base": None, "proc": None, "adapters": {}, "current": None}


def adopted_elements():
    """只暴露附录里采用的元素（去掉 holdout / 失败尝试的 adapter）。"""
    p = r"C:\vlmdisaster\处理方案\论文附录_元素总表_20261117.md"
    if not os.path.exists(p):
        return set()
    txt = open(p, encoding="utf-8").read()
    return set(re.findall(r"^\| \d+ \| (\w+) \|", txt, re.M))


def list_elements():
    keep = adopted_elements()
    els = []
    for d in glob.glob(os.path.join(OUT_ROOT, "qwen3vl-lora-*-json")):
        if os.path.exists(os.path.join(d, "adapter_config.json")):
            name = os.path.basename(d)[len("qwen3vl-lora-"):-len("-json")]
            if keep and name not in keep:
                continue
            els.append(name)
    return sorted(els)


def prompt_of(name):
    for suf in ("_val", ""):
        p = os.path.join(LF_DATA, "train_%s_json%s.json" % (name, suf))
        if os.path.exists(p):
            recs = json.load(open(p, encoding="utf-8"))
            if recs:
                txt = recs[0]["conversations"][0]["value"].replace("<image>", "").strip()
                return txt
    return "Describe the hazard in this image and output JSON."


def load_base():
    if _state["base"] is not None:
        return _state["base"], _state["proc"]
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qc, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    _state["base"], _state["proc"] = base, proc
    return base, proc


def use_adapter(name):
    from peft import PeftModel

    base, proc = load_base()
    if _state["current"] == name and _state.get("peft") is not None:
        return _state["peft"], proc
    lora = os.path.join(OUT_ROOT, "qwen3vl-lora-%s-json" % name)
    peft = _state.get("peft")
    if peft is None:
        peft = PeftModel.from_pretrained(base, lora, adapter_name=name)
        _state["peft"] = peft
        _state["adapters"][name] = peft
    else:
        if name not in getattr(peft, "peft_config", {}):
            peft.load_adapter(lora, adapter_name=name)
            _state["adapters"][name] = peft
    peft.set_adapter(name)
    peft.eval()
    _state["current"] = name
    return peft, proc


def parse_json(t):
    blocks = []
    depth, start = 0, None
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(t[start:i + 1])
    for b in blocks:
        try:
            return json.loads(b)
        except Exception:
            continue
    return None


def predict(name, img_path, extra_text=""):
    import torch
    from qwen_vl_utils import process_vision_info

    model, proc = use_adapter(name)
    prompt = prompt_of(name)
    if extra_text.strip():
        prompt = prompt + (
            "\n\nAdditional measurements and features. Use them together with the image; "
            "they come from physical statistics, a deep-network visual feature map and a "
            "semantic (image-text) model:\n" + extra_text.strip())
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": "file://%s" % img_path},
        {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inp = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=320, do_sample=False, repetition_penalty=1.05)
    raw = proc.tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return parse_json(raw), raw


@app.get("/api/elements")
def api_elements():
    return {"elements": list_elements()}


def _capability():
    if not os.path.exists(CAPABILITY):
        return {}
    try:
        return json.load(open(CAPABILITY, encoding="utf-8"))
    except Exception:
        return {}


def _meta():
    out = {}
    if not os.path.exists(APPENDIX):
        return out
    for line in open(APPENDIX, encoding="utf-8"):
        m = re.match(r"\| \d+ \| (\w+) \| ([^|]+) \| ([^|]+) \|", line)
        if m:
            out[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
    return out


def _samples(el, k=3):
    """优先返回「按日期留出」的验证图（真正训练集之外），否则退回随机验证集。"""
    plan = [
        ("train_%s_hold_json_val.json" % el, "未见日期"),
        ("train_%s_json_val.json" % el, "验证集"),
    ]
    for fname, tag in plan:
        p = os.path.join(LF_DATA, fname)
        if not os.path.exists(p):
            continue
        try:
            recs = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        urls = []
        for r in recs:
            img = r.get("image")
            if not img and isinstance(r.get("images"), list):
                img = r["images"][0]
            if img:
                rel = os.path.relpath(img, DATASETS).replace("\\", "/")
                urls.append({"url": "/images/" + rel, "tag": tag,
                             "label": r.get("label", ""), "_is_rgb": "_rgb" in rel})
        # 指数图优先（避免样例取到 RGB 产品图）
        urls.sort(key=lambda x: x["_is_rgb"])
        # 尽量按标签各取一张（正反例都覆盖），再补到 k 张
        picked, seen = [], set()
        for u in urls:
            if u["label"] not in seen:
                seen.add(u["label"])
                picked.append(u)
            if len(picked) >= k:
                break
        for u in urls:
            if len(picked) >= k:
                break
            if u not in picked:
                picked.append(u)
        NEG = {"none", "normal", "quiet", "no_snow", "dry", "stable", "land", "blue", "calm", "clear", "open_water", "clean_water", "no", "no_disaster"}
        picked.sort(key=lambda x: 1 if str(x.get("label")) in NEG else 0)
        for u in picked:
            u.pop("_is_rgb", None)
        if picked:
            return picked
    return []


@app.get("/api/route")
def api_route(name: str = ""):
    """按文件名/产品名推荐元素（规则路由，供前端自动选中）。"""
    n = (name or "").lower()
    rules = [
        ("ndwi", "flood_opt"), ("drg_", "drought_grace"), ("active_fire", "wildfire"),
        ("_ti4", "wildfire"), ("iod_anom", "iod"), ("ndp_", "drought_ndvi_clim"),
        ("mhw_anom", "marine_heatwave"), ("sla_", "coastal_sla"), ("coral_", "coral_dhw"),
        ("enso_anom", "enso"), ("_fai", "green_tide"), ("flare", "flare"),
        ("_fsc", "snow_fsc"), ("wlp_", "waterlogging_pct"), ("swe_", "snow_swe"),
        ("_precip", "hurricane_rain"), ("gpm_", "gpm_nt"), ("gw_", "groundwater_grace"),
        ("bloom_", "bloom_chla"), ("fp_", "flood_precip_global"), ("eqd_", "earthquake_def"),
        ("seaice_", "sea_ice_amsr2"), ("gwp90", "groundwater_overuse"), ("volc_", "volcano_thermal"),
        ("smoke_", "wildfire_smoke"), ("_seg1_anom", "wave"), ("uvai_", "uvai_dust"),
        ("so2_", "volcano_so2"), ("_storm_grade", "geomagnetic"), ("dust_", "sanddust"),
        ("_ndsi", "sea_ice_s2_arctic"), ("hw_", "heatwave_lst"), ("_disp", "ground_subsidence"),
        ("flood_water", "flood_water"), ("haze_", "haze"), ("_speed", "wind_gale"),
        ("_rgb", "hurricane_opt"),
    ]
    cands = []
    for tok, el in rules:
        if tok in n and el not in cands:
            cands.append(el)
    # 没有词根时按命名结构猜
    if not cands:
        if re.match(r"^\d{8}_\d+_\d+_grade", n):
            cands.append("snowstorm")
        elif re.match(r"^\d{8}_[ns]\d+_[ew]\d+_grade", n):
            cands.append("coastal_nt")
    meta = _meta()
    out = [{"element": el, "zh": meta.get(el, (el, ""))[0]} for el in cands if el in list_elements()]
    return {"name": name, "candidates": out}



@app.get("/api/artifacts")
def api_artifacts():
    if not os.path.exists(ARTIFACTS):
        return {"scripts": [], "docs": [], "endpoints": []}
    try:
        return json.load(open(ARTIFACTS, encoding="utf-8"))
    except Exception:
        return {"scripts": [], "docs": [], "endpoints": []}
@app.get("/api/status")
def api_status():
    """演示前确认「模型已暖机」。"""
    last = None
    try:
        import glob as _glob
        files = sorted(_glob.glob(os.path.join(OUT_ROOT, "e2e_all_*.json")))
        if files:
            d = json.load(open(files[-1], encoding="utf-8"))
            last = {"ok": d.get("ok"), "total": d.get("total"),
                    "file": os.path.basename(files[-1])}
    except Exception:
        last = None
    return {
        "last_e2e": last,
        "model_loaded": _state["base"] is not None,
        "current_element": _state["current"],
        "adapters_cached": len(_state["adapters"]),
        "elements": len(list_elements()),
    }


@app.get("/api/catalog")
def api_catalog():
    meta = _meta()
    keep = list_elements()
    groups = {}
    for el in keep:
        zh, src = meta.get(el, (el, "其他"))
        g = groups.setdefault(src, {"source": src, "label": SOURCE_ZH.get(src, src), "elements": []})
        g["elements"].append({"name": el, "zh": zh, "samples": _samples(el),
                            "cap": _capability().get(el)})
    order = ["OPT", "TIR", "MW", "PRECIP", "ALT", "GRACE", "SCAT", "SPACE", "SAR"]
    out = [groups[s] for s in order if s in groups]
    out += [g for s, g in groups.items() if s not in order]
    return {"groups": out}


@app.post("/api/predict")
async def api_predict(element: str = Form(...), lat: str = Form(""), lon: str = Form(""),
                      extra_text: str = Form(""),
                      report_backend: str = Form("template"),
                      file: UploadFile = File(...)):
    if element not in list_elements():
        return JSONResponse({"error": "unknown element"}, status_code=400)
    suffix = os.path.splitext(file.filename or "x.png")[1] or ".png"
    path = os.path.join(tempfile.gettempdir(), "dsh_predict_%d%s" % (os.getpid(), suffix))
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        with _lock:
            pred, raw = predict(element, path, extra_text)
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            return JSONResponse({"error": "显存不足：关掉其他 GPU 程序后重试（服务是串行推理，不并发）"},
                                status_code=507)
        return JSONResponse({"error": "推理失败：%s" % msg[:200]}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": "%s: %s" % (type(e).__name__, str(e)[:200])}, status_code=500)
    if pred is None:
        return JSONResponse({"error": "模型输出无法解析", "raw": raw}, status_code=502)
    pred["element"] = element
    if lat.strip() and lon.strip():
        try:
            pred["lat"], pred["lon"] = float(lat), float(lon)
            pred["location"] = geocode(float(lat), float(lon))
        except ValueError:
            pass
    backend = "openai" if report_backend == "llm" else "template"
    return {"prediction": pred, "report": make_report(pred, backend)}


# ---------------- LLM 配置接口 ----------------

@app.get("/api/llm")
def api_llm_get():
    from llm_config import load, mask
    c = load()
    return {"base_url": c["base_url"], "model": c["model"],
            "api_key": mask(c["api_key"]),
            "configured": bool(c["api_key"] and c["model"])}


@app.post("/api/llm")
async def api_llm_set(payload: dict):
    from llm_config import load, save, mask
    c = load()
    for k in ("base_url", "api_key", "model"):
        v = (payload or {}).get(k)
        if isinstance(v, str) and v.strip():
            c[k] = v.strip()
    save(c)
    return {"ok": True, "base_url": c["base_url"], "model": c["model"],
            "api_key": mask(c["api_key"]), "configured": True}


@app.post("/api/llm/test")
async def api_llm_test():
    from llm_config import load
    import urllib.request
    c = load()
    if not c["api_key"]:
        return {"ok": False, "error": "未配置 api_key"}
    url = c["base_url"].rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + c["api_key"]})
        d = json.loads(urllib.request.urlopen(req, timeout=30).read())
        ids = [m.get("id") for m in d.get("data", [])]
        return {"ok": True, "models": ids[:20], "model_in_list": c["model"] in ids}
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}



@app.post("/api/report_aggregate")
async def api_report_aggregate(payload: dict):
    """多元素汇总研判：{items:[pred...], lat?, lon?, backend?}"""
    items = (payload or {}).get("items") or []
    if not items:
        return JSONResponse({"error": "items 为空"}, status_code=400)
    lat = payload.get("lat")
    lon = payload.get("lon")
    backend = "openai" if (payload.get("backend") or "").lower() in ("llm", "openai") else "template"
    return {"report": aggregate_report(items, lat, lon, backend)}
@app.get("/", response_class=HTMLResponse)
def index():
    p = os.path.join(STATIC, "index.html")
    return HTMLResponse(open(p, encoding="utf-8").read() if os.path.exists(p) else "<h1>web/index.html 缺失</h1>")

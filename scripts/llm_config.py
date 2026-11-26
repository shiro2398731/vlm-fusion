# -*- coding: utf-8 -*-
r"""
LLM 配置接口（研判后端）

配置优先级：环境变量 > config/llm.json > 默认值

    .\venv\Scripts\python.exe scripts\llm_config.py show
    .\venv\Scripts\python.exe scripts\llm_config.py set --api-key sk-xxx --model deepseek-chat
    .\venv\Scripts\python.exe scripts\llm_config.py test
    .\venv\Scripts\python.exe scripts\llm_config.py clear

说明：
    - API key 只写入 config/llm.json（已 gitignore），不回显完整值
    - 任何 OpenAI 兼容接口都能用（DeepSeek / 自建网关 / vLLM）
    - 模型名不确定时用 `test` 会顺带列出可用模型（GET /models）
"""
import argparse
import json
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\vlmdisaster"
CFG_DIR = os.path.join(ROOT, "config")
CFG = os.path.join(CFG_DIR, "llm.json")
DEFAULT_BASE = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-flash"


def load():
    cfg = {"base_url": os.environ.get("LLM_BASE_URL", DEFAULT_BASE),
           "api_key": os.environ.get("LLM_API_KEY", ""),
           "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL)}
    if os.path.exists(CFG):
        try:
            d = json.load(open(CFG, encoding="utf-8"))
            for k in ("base_url", "api_key", "model"):
                if d.get(k):
                    cfg[k] = d[k]
        except Exception:
            pass
    # 环境变量最优先
    for k, env in (("base_url", "LLM_BASE_URL"), ("api_key", "LLM_API_KEY"), ("model", "LLM_MODEL")):
        if os.environ.get(env):
            cfg[k] = os.environ[env]
    return cfg


def mask(k):
    if not k:
        return "(未设置)"
    return k[:6] + "…" + k[-4:] if len(k) > 12 else "***"


def save(cfg):
    os.makedirs(CFG_DIR, exist_ok=True)
    json.dump(cfg, open(CFG, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    try:
        os.chmod(CFG, 0o600)
    except Exception:
        pass


def api(cfg, path, data=None):
    url = cfg["base_url"].rstrip("/") + path
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + cfg["api_key"]})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def cmd_show(a):
    c = load()
    print("base_url :", c["base_url"])
    print("model    :", c["model"])
    print("api_key  :", mask(c["api_key"]))
    print("config   :", CFG if os.path.exists(CFG) else "(未创建)")


def cmd_set(a):
    c = load()
    if a.base_url:
        c["base_url"] = a.base_url
    if a.api_key:
        c["api_key"] = a.api_key
    if a.model:
        c["model"] = a.model
    save(c)
    print("已保存：", CFG)
    print("base_url :", c["base_url"])
    print("model    :", c["model"])
    print("api_key  :", mask(c["api_key"]))


def cmd_clear(a):
    if os.path.exists(CFG):
        os.remove(CFG)
        print("已删除", CFG)
    else:
        print("没有配置文件")


def cmd_test(a):
    c = load()
    if not c["api_key"]:
        print("✗ 没有 api_key，先 set")
        return
    try:
        models = api(c, "/models")
        ids = [m.get("id") for m in models.get("data", [])]
        print("✓ 连通；可用模型：", ", ".join(ids[:12]) or "(接口未返回列表)")
        if ids and c["model"] not in ids:
            print("⚠ 当前 model=%s 不在列表里，请从上面挑一个：llm_config.py set --model <id>" % c["model"])
    except Exception as e:
        print("✗ 连接失败：", type(e).__name__, e)
        return
    body = json.dumps({"model": c["model"], "messages": [
        {"role": "user", "content": "只回复两个字：收到"}], "temperature": 0}).encode()
    try:
        r = api(c, "/chat/completions", body)
        print("✓ 推理测试：", r["choices"][0]["message"]["content"][:40])
    except Exception as e:
        print("✗ 推理失败：", type(e).__name__, e)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("show")
    p = sub.add_parser("set")
    p.add_argument("--base-url", default="")
    p.add_argument("--api-key", default="")
    p.add_argument("--model", default="")
    sub.add_parser("test")
    sub.add_parser("clear")
    a = ap.parse_args()
    {"show": cmd_show, "set": cmd_set, "test": cmd_test, "clear": cmd_clear}.get(
        a.cmd or "show", cmd_show)(a)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""用本地 mock OpenAI 兼容服务验证 pipeline 的 API LLM 后端。"""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        assert "messages" in body
        content = json.dumps({"disaster_type": "coral_bleaching", "detected": True,
                              "severity": "alert1", "source": "TIR"})
        out = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", 8098), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

recs = json.load(open(r"C:\vlmdisaster\LLaMA-Factory\data\train_coral_dhw_json_val.json", encoding="utf-8"))
r = [x for x in recs if os.path.exists(x.get("image", ""))][0]
cfg = {"element": "coral_dhw",
       "numeric": {"extractor": "none"}, "visual": {"extractor": "none"},
       "semantic": {"extractor": "none"},
       "fusion": {"mode": ["text", "image"]},
       "llm": {"backend": "api", "base_url": "http://127.0.0.1:8098/v1",
               "api_key": "x", "model": "mock"}}
p = ElementPipeline(cfg)
out = p.run(r["image"], "unused", "coral_dhw", {})
pred = out["prediction"].get("prediction") or out["prediction"]
print("api llm prediction", pred)
assert pred.get("severity") == "alert1"
print("llm api backend OK")
srv.shutdown()

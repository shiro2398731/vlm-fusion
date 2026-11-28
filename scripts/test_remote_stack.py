# -*- coding: utf-8 -*-
"""验证远程语义 + 远程判读 LLM 的组合模式（本地 mock）。"""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pipeline import ElementPipeline


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        json.loads(self.rfile.read(n) or b"{}")
        if self.path.startswith("/semantic"):
            out = json.dumps({"scores": {"none": 0.05, "alert1": 0.9, "alert2": 0.05}}).encode()
        else:
            content = json.dumps({"disaster_type": "coral_bleaching", "detected": True,
                                  "severity": "watch", "source": "TIR"})
            out = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", 8097), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

recs = json.load(open(r"C:\vlmdisaster\LLaMA-Factory\data\train_coral_dhw_json_val.json", encoding="utf-8"))
r = [x for x in recs if os.path.exists(x.get("image", ""))][0]
cfg = {"element": "coral_dhw",
       "numeric": {"extractor": "none"},
       "visual": {"extractor": "none"},
       "semantic": {"extractor": "clip_api",
                    "base_url": "http://127.0.0.1:8097/semantic",
                    "model": "mock-semantic"},
       "fusion": {"mode": ["decision"], "tau": 0.6, "source": "semantic"},
       "llm": {"backend": "api", "base_url": "http://127.0.0.1:8097/v1",
               "api_key": "x", "model": "mock-llm"}}
p = ElementPipeline(cfg)
out = p.run(r["image"], "unused", "coral_dhw", {})
pred = out["prediction"]
print("remote stack:", {k: pred.get(k) for k in ("base_severity", "severity", "severity_source")})
assert pred.get("severity") == "alert1" and pred.get("severity_source") == "probe"
print("remote semantic + remote LLM OK")
srv.shutdown()

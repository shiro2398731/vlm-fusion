# -*- coding: utf-8 -*-
"""用本地 mock HTTP 服务验证 clip_api 契约。"""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from features.base import FeatureBundle
from features.extractors import ClipAPIExtractor


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        assert "image_b64" in body
        out = json.dumps({"scores": {"none": 0.1, "alert1": 0.9}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers()
        self.wfile.write(out)
    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", 8099), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

ext = ClipAPIExtractor(base_url="http://127.0.0.1:8099/clip", model="mock")
b = FeatureBundle(element="coral_dhw", image_path=r"C:\vlmdisaster\datasets\01_optical_vnir\snow_fsc\images\A2024002.0736_qingzang_794_0_fsc.png")
b = ext.extract(b, labels=["none", "alert1"])
print("scores", b.semantic_scores)
assert b.semantic_scores and abs(b.semantic_scores["alert1"] - 0.9) < 1e-6
print("clip_api contract OK")
srv.shutdown()

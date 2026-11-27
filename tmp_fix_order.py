# -*- coding: utf-8 -*-
"""把 ClipProbeExtractor 移到 ClipExtractor 之后。"""
import io
p = r"C:\vlm-fusion\features\extractors.py"
s = io.open(p, encoding="utf-8").read()
start = s.find("class ClipProbeExtractor")
end = s.find("class DinoV2Extractor")
if start < 0 or end < 0:
    print("MISS"); raise SystemExit(1)
block = s[start:end]
s = s[:start] + s[end:]
anchor = "REGISTRY = {"
i = s.find(anchor)
s = s[:i] + block + "\n" + s[i:]
io.open(p, "w", encoding="utf-8").write(s)
print("moved", len(block))

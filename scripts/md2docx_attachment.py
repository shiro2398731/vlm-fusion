# -*- coding: utf-8 -*-
"""把附件 markdown 转成 docx。"""
import re, sys
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC = r"C:\vlm-fusion\docs\附件_三路特征融合架构.md"
DST = r"C:\vlm-fusion\docs\附件_三路特征融合架构.docx"

lines = open(SRC, encoding="utf-8").read().splitlines()
doc = Document()
style = doc.styles["Normal"]
style.font.name = "等线"
style.font.size = Pt(10.5)

i = 0
while i < len(lines):
    line = lines[i].rstrip()
    if not line.strip():
        i += 1
        continue
    if line.startswith("#"):
        lvl = len(line) - len(line.lstrip("#"))
        doc.add_heading(line[lvl:].strip(), level=min(lvl, 4))
        i += 1
        continue
    if line.startswith("```"):
        i += 1
        buf = []
        while i < len(lines) and not lines[i].startswith("```"):
            buf.append(lines[i]); i += 1
        i += 1
        for b in buf:
            p = doc.add_paragraph(b)
            p.paragraph_format.left_indent = Pt(12)
            for r in p.runs:
                r.font.name = "Consolas"; r.font.size = Pt(9)
        continue
    if line.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                rows.append(cells)
            i += 1
        if rows:
            cols = max(len(r) for r in rows)
            t = doc.add_table(rows=len(rows), cols=cols)
            t.style = "Table Grid"
            for ri, row in enumerate(rows):
                for ci in range(cols):
                    t.cell(ri, ci).text = row[ci] if ci < len(row) else ""
        continue
    if line.startswith(("- ", "* ")):
        doc.add_paragraph(line[2:].strip(), style="List Bullet")
        i += 1
        continue
    if re.match(r"^\d+\.\s", line):
        doc.add_paragraph(re.sub(r"^\d+\.\s", "", line), style="List Number")
        i += 1
        continue
    doc.add_paragraph(line)
    i += 1

doc.save(DST)
print("saved", DST)

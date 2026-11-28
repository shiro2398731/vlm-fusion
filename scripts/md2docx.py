# -*- coding: utf-8 -*-
"""通用 markdown → docx（标题/表格/代码块/列表）。"""
import re, sys
from docx import Document
from docx.shared import Pt


def convert(src, dst, font_size=10.5):
    lines = open(src, encoding="utf-8").read().splitlines()
    doc = Document()
    doc.styles["Normal"].font.name = "等线"
    doc.styles["Normal"].font.size = Pt(font_size)
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1; continue
        if line.startswith("#"):
            lvl = len(line) - len(line.lstrip("#"))
            doc.add_heading(line[lvl:].strip(), level=min(lvl, 4)); i += 1; continue
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
                t = doc.add_table(rows=len(rows), cols=cols); t.style = "Table Grid"
                for ri, row in enumerate(rows):
                    for ci in range(cols):
                        t.cell(ri, ci).text = row[ci] if ci < len(row) else ""
            continue
        if line.startswith(("- ", "* ")):
            doc.add_paragraph(line[2:].strip(), style="List Bullet"); i += 1; continue
        if re.match(r"^\d+\.\s", line):
            doc.add_paragraph(re.sub(r"^\d+\.\s", "", line), style="List Number"); i += 1; continue
        doc.add_paragraph(line); i += 1
    doc.save(dst)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
    print("saved", sys.argv[2])

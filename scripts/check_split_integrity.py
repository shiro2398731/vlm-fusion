# -*- coding: utf-8 -*-
"""检查每个元素训练集与测试集（val/hold）的图像与日期重叠，输出合法切分表。"""
import glob, io, json, os, re, sys

LF = r"C:\vlmdisaster\LLaMA-Factory\data"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

els = sorted(json.load(open(os.path.join(LF, "..", "..", "处理方案", "元素能力_20261125.json"),
                            encoding="utf-8")).keys())

def load(name):
    p = os.path.join(LF, name)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None

def imgs(recs):
    return {os.path.basename(str(r.get("image") or "")) for r in recs if r.get("image")}

def dates(recs):
    out = set()
    for r in recs:
        s = str(r.get("stem") or r.get("stamp") or r.get("date") or os.path.basename(str(r.get("image") or "")))
        m = re.search(r"A?(\d{4})[m.-]?(\d{2})[m.-]?(\d{2})", s)
        if m:
            out.add(m.group(1) + m.group(2) + m.group(3))
            continue
        m = re.search(r"(\d{4})(\d{3})", s)
        if m:
            out.add(m.group(1) + m.group(2))
    return out

rows = []
for el in els:
    cands = sorted(os.path.basename(p) for p in glob.glob(os.path.join(LF, "train_%s_*json*" % el)))
    train_name = "train_%s_json.json" % el
    tr = load(train_name)
    if tr is None:
        # 尝试 hold 与 json 的不同组合
        for c in cands:
            if c.endswith("_json.json"):
                train_name = c; tr = load(c); break
    if tr is None:
        rows.append((el, "?", 0, "-", "-", "无训练文件"))
        continue
    tr_i, tr_d = imgs(tr), dates(tr)
    tests = []
    for c in cands:
        if c == train_name:
            continue
        if not (c.endswith("_val.json") or "_val." in c or "hold" in c):
            continue
        rs = load(c)
        if not rs:
            continue
        ti, td = imgs(rs), dates(rs)
        io_ = len(tr_i & ti)
        do = len(tr_d & td)
        tests.append((c, len(rs), io_, do, len(td)))
    if not tests:
        rows.append((el, train_name, len(tr), "-", "-", "无测试文件"))
        continue
    # 选重叠最少、样本最多的测试集作为判定依据
    best = min(tests, key=lambda t: (t[2], -t[1]))
    verdict = "干净" if best[2] == 0 and best[3] == 0 else (
        "样本级干净但同日期" if best[2] == 0 else "污染(图像重叠)")
    rows.append((el, train_name, len(tr), best[0], "%d/%d 图重叠, %d 日期重叠" % (best[2], best[1], best[3]), verdict))

print("%-22s %-34s %6s %-38s %-30s %s" % ("element", "train", "n_train", "best test", "overlap", "verdict"))
out = []
for r in rows:
    print("%-22s %-34s %6s %-38s %-30s %s" % r)
    out.append({"element": r[0], "train": r[1], "n_train": r[2], "test": r[3], "overlap": r[4], "verdict": r[5]})

json.dump(out, open(r"C:\vlmdisaster\处理方案\切分合法性.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved 切分合法性.json")

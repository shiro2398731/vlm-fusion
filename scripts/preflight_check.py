# -*- coding: utf-8 -*-
r"""
训练前体检：一个元素在开训之前必须过的 6 道检查。

为什么需要：项目踩过的坑几乎都能在这里提前发现——
  · 正负比例失衡（35~75% 外容易塌缩）
  · 数值字段不方差/全 0（模型没法读，必然塌）
  · 训练/验证集图片重叠（指标虚高）
  · 验证集全正类 / 标签分布和训练集不一致（数字无意义）
  · 提问枚举与答案不一致（模型答非所问）
  · 图片缺失

用法：
  .\venv\Scripts\python.exe scripts\preflight_check.py --element sea_ice_amsr2
  .\venv\Scripts\python.exe scripts\preflight_check.py --element wind_gale --strict

退出码：0 = 通过（可带警告）；1 = 有硬问题，不建议开训
"""
import argparse
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = r"C:\vlmdisaster"
LF = os.path.join(ROOT, "LLaMA-Factory", "data")
INFO = os.path.join(LF, "dataset_info.json")

hard, soft = [], []


def load_json(name):
    p = os.path.join(LF, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def img_of(rec):
    v = rec.get("image") or rec.get("images")
    if isinstance(v, list):
        v = v[0] if v else None
    return v


def gold_of(rec):
    for t in rec.get("conversations") or []:
        if t.get("from") == "gpt":
            try:
                return json.loads(t.get("value", ""))
            except Exception:
                return None
    return None


def prompt_of(rec):
    for t in rec.get("conversations") or []:
        if t.get("from") == "human":
            return t.get("value", "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--element", required=True)
    ap.add_argument("--strict", action="store_true", help="把警告也算失败")
    a = ap.parse_args()
    el = a.element
    info = json.load(open(INFO, encoding="utf-8"))
    tk, vk = "disaster_%s_json" % el, "disaster_%s_json_val" % el

    print("=" * 70)
    print("  训练前体检：%s" % el)
    print("=" * 70)

    # 1) 注册与文件
    if tk not in info or vk not in info:
        print("  ✗ dataset_info 里缺少 %s / %s" % (tk, vk))
        return 1
    train = load_json(info[tk]["file_name"])
    val = load_json(info[vk]["file_name"])
    if not train or not val:
        print("  ✗ 训练或验证 JSON 读不到")
        return 1
    print("\n[1] 数据文件")
    print("  train %d 条 / val %d 条" % (len(train), len(val)))

    # 2) 图片存在
    miss = 0
    for rec in train + val:
        p = img_of(rec)
        if not p or not os.path.exists(p):
            miss += 1
    print("\n[2] 图片")
    print("  缺失 %d 张" % miss)
    if miss:
        hard.append("图片缺失 %d 张" % miss)

    # 3) 标签分布
    def dist(recs):
        c = collections.Counter()
        det = 0
        for r in recs:
            g = gold_of(r) or {}
            c[str(g.get("severity"))] += 1
            if g.get("detected") is True:
                det += 1
        return c, det

    ct, dt = dist(train)
    cv, dv = dist(val)
    print("\n[3] 标签分布")
    print("  train severity:", dict(ct), " 正类 %d/%d = %.0f%%" % (dt, len(train), 100.0 * dt / max(1, len(train))))
    print("  val   severity:", dict(cv), " 正类 %d/%d = %.0f%%" % (dv, len(val), 100.0 * dv / max(1, len(val))))
    pr = dt / max(1, len(train))
    if not (0.35 <= pr <= 0.75):
        soft.append("训练集正类比例 %.0f%% 不在 35~75%% 区间" % (pr * 100))
    if dv == 0 or dv == len(val):
        hard.append("验证集标签全同类（全正或全负），指标无意义")
    if set(cv) != set(ct):
        soft.append("验证集与训练集的 severity 类别不一致")

    # 4) 数值字段
    print("\n[4] 数值字段（防塌缩的关键）")
    keys = set()
    for r in train:
        g = gold_of(r) or {}
        for k, v in g.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    if not keys:
        soft.append("答案里没有任何数值字段（模型可能走多数类）")
        print("  无")
    for k in sorted(keys):
        vals = []
        for r in train:
            g = gold_of(r) or {}
            if k in g:
                try:
                    vals.append(float(g[k]))
                except (TypeError, ValueError):
                    pass
        if not vals:
            continue
        nz = sum(1 for x in vals if x != 0) / len(vals)
        uniq = len(set(vals))
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        cv_ = (var ** 0.5) / abs(mean) if abs(mean) > 1e-9 else 0.0
        flag = ""
        if nz < 0.3:
            flag = " ⚠ 非零<30%"
            soft.append("字段 %s 非零仅 %.0f%%" % (k, nz * 100))
        if uniq <= 10:
            flag += " ⚠ 唯一值<10"
            soft.append("字段 %s 唯一值仅 %d" % (k, uniq))
        if cv_ < 0.1:
            flag += " ⚠ CV<0.1"
            soft.append("字段 %s 变异系数 %.2f 过小" % (k, cv_))
        print("  %-18s 非零 %3.0f%%  唯一值 %4d  CV %.2f%s" % (k, nz * 100, uniq, cv_, flag))

    # 5) 训练/验证集重叠
    tr = {os.path.normcase(os.path.abspath(img_of(r))) for r in train if img_of(r)}
    va = {os.path.normcase(os.path.abspath(img_of(r))) for r in val if img_of(r)}
    inter = tr & va
    print("\n[5] 训练/验证集重叠")
    print("  %d 张" % len(inter))
    if inter:
        hard.append("训练/验证集重叠 %d 张" % len(inter))

    # 6) 提问枚举与答案一致性
    print("\n[6] 提问枚举 vs 答案")
    sev_vals = set(ct) | set(cv)
    sample_prompt = prompt_of(train[0]).lower()
    missing = [s for s in sev_vals if s and s != "None" and s.lower() not in sample_prompt]
    if missing:
        soft.append("提问里没出现答案中的 severity: %s" % ", ".join(missing))
        print("  ⚠ 答案里有但提问没写：%s" % ", ".join(missing))
    else:
        print("  ✓ 答案 severity 都能在提问里找到")

    print("\n" + "=" * 70)
    if hard:
        print("  ✗ 硬问题 %d 个：%s" % (len(hard), "；".join(hard)))
    if soft:
        print("  ⚠ 警告 %d 个：%s" % (len(soft), "；".join(soft)))
    if not hard and not soft:
        print("  ✓ 全部通过，可以开训")
    elif not hard:
        print("  可开训（但请保留上面的警告，论文里要说明）")
    return 1 if (hard or (a.strict and soft)) else 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""用可配置 LLM（本地/API）做判读一致性评估。

输入：JSON 列表，每条含 truth 与 pred（判读 JSON 或等级词）。
输出：逐条 match 与汇总准确率。
"""
import argparse, json, os, sys
sys.path.insert(0, r"C:\vlm-fusion")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from llmeval import LLMClient

PROMPT = """你是遥感灾害判读的评审。比较标准答案与模型判读，判断等级是否一致。
标准答案：{truth}
模型判读：{pred}
只输出一个 JSON：{{"match": true/false, "reason": "一句话"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=r"C:\vlm-fusion\configs\llm_eval.example.yaml")
    ap.add_argument("--input", required=True, help="JSON 列表，元素含 truth/pred")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = LLMClient.from_config(args.config)
    print("backend=%s model=%s" % (client.backend, client.cfg.get("model", "")))
    if args.dry_run:
        return

    rows = json.load(open(args.input, encoding="utf-8"))
    ok = 0
    out = []
    for r in rows:
        txt = client.chat(PROMPT.format(truth=r.get("truth"), pred=r.get("pred")),
                          temperature=0.0, max_tokens=200)
        try:
            j = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
        except Exception:
            j = {"match": False, "reason": txt[:100]}
        ok += bool(j.get("match"))
        out.append({"truth": r.get("truth"), "pred": r.get("pred"), **j})
    print("match %d/%d" % (ok, len(rows)))
    print(json.dumps(out[:3], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

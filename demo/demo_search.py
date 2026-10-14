"""
离线检索演示（零重依赖）
========================
用途：在没有安装 sentence-transformers / chromadb 的机器上，
      也能用**真实语料**跑出检索结果，用于生成演示截图或快速验证。

原理：手写 BM25（字符 2-gram + 3-gram，对中文无需分词）。
      它不是线上用的向量检索，只用于「离线演示 / 无依赖冒烟测试」。

线上正式链路见 src/retriever.py（向量检索）。

用法：
    python demo/demo_search.py                          # 跑内置示例问题
    python demo/demo_search.py "武汉哪个公园适合带孩子"    # 自定义问题
    python demo/demo_search.py "黄鹤楼值不值得去" --city 武汉
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "comments.jsonl"

# 中文停用词（简短表，够用即可）
STOP = set("的了吧呢吗啊呀哦嗯是在有和与及或就都也很不我你他她它们这那什么怎么哪个哪里")


def _grams(text: str):
    """把中文句子切成字符 2-gram / 3-gram（无需分词库）。"""
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    out = []
    for n in (2, 3):
        out += [s[i:i + n] for i in range(len(s) - n + 1)]
    return [g for g in out if g not in STOP]


class BM25:
    """极简 BM25。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.tfs = [Counter(_grams(d["text"])) for d in docs]
        self.lens = [sum(tf.values()) or 1 for tf in self.tfs]
        self.avg = sum(self.lens) / len(self.lens)

        df = defaultdict(int)
        for tf in self.tfs:
            for g in tf:
                df[g] += 1
        n = len(docs)
        self.idf = {g: math.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}

    def search(self, query, top_k=5, city=None):
        q = Counter(_grams(query))
        scored = []
        for i, tf in enumerate(self.tfs):
            if city and self.docs[i]["city"] != city:
                continue
            score = 0.0
            for g, qn in q.items():
                f = tf.get(g, 0)
                if not f:
                    continue
                idf = self.idf.get(g, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                score += idf * (f * (self.k1 + 1)) / denom * qn
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)

        hits = []
        for score, i in scored[:top_k]:
            d = dict(self.docs[i])
            d["score"] = round(score, 3)
            hits.append(d)
        return hits


def load():
    docs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default="武汉哪个公园适合带孩子")
    ap.add_argument("--city", default=None, help="限定城市")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    docs = load()
    print(f"载入语料：{len(docs)} 条评论 | 城市 {len({d['city'] for d in docs})} 个\n")

    bm = BM25(docs)
    hits = bm.search(args.question, top_k=args.top, city=args.city)

    print("=" * 62)
    print(f'问题：{args.question}' + (f'（限定城市：{args.city}）' if args.city else ""))
    print("=" * 62)
    if not hits:
        print("（没有命中任何评论）")
    for i, h in enumerate(hits, 1):
        text = h["text"]
        if len(text) > 110:
            text = text[:110] + "…"
        print(f"\n[{i}] {h['spot']}  ·  {h['city']}   score={h['score']:.3f}")
        print(f"    {text}")
    print()


if __name__ == "__main__":
    sys.exit(main())

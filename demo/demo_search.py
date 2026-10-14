"""
离线关键词检索演示（零重依赖）
==============================
用途：在没有安装 sentence-transformers / chromadb 的机器上，
      也能用**真实语料**跑出检索结果，用于生成演示截图或快速冒烟测试。

BM25 实现已移到 src/bm25.py（线上混合检索与评测共用同一份），这里只做命令行包装。

用法：
    python demo/demo_search.py                          # 跑内置示例问题
    python demo/demo_search.py "武汉哪个公园适合带孩子"    # 自定义问题
    python demo/demo_search.py "黄鹤楼值不值得去" --city 武汉
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "comments.jsonl"
sys.path.insert(0, str(ROOT / "src"))

from bm25 import BM25          # noqa: E402


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

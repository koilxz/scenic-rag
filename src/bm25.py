"""
BM25 关键词检索
==============
轻量实现，不依赖 jieba 等分词库 —— 用**字符 2-gram / 3-gram** 代替分词，
对中文短句足够稳，且省掉一个依赖。

它是混合检索（src/hybrid.py）的两路召回之一，也是评测里的关键词基线。

用法：
    from bm25 import BM25
    bm = BM25(docs)                     # docs: [{"text":…, "spot":…, "city":…}, …]
    bm.search("武汉哪里适合带孩子", top_k=5, city="武汉")
"""
import math
import re
from collections import Counter, defaultdict

# 中文停用词（简短表，够用即可）
STOP = set("的了吧呢吗啊呀哦嗯是在有和与及或就都也很不我你他她它们这那什么怎么哪个哪里")


def _grams(text: str):
    """把中文句子切成字符 2-gram / 3-gram（无需分词库）。"""
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(text))
    out = []
    for n in (2, 3):
        out += [s[i:i + n] for i in range(len(s) - n + 1)]
    return [g for g in out if g not in STOP]


class BM25:
    """极简 BM25（k1=1.5, b=0.75 为常用默认值）。"""

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

    def search(self, query: str, top_k: int = 5, city: str = None):
        """
        返回 top_k 条：
        [{"text":…, "spot":…, "city":…, "score":…}, …]
        score 为 BM25 原始分（不同查询之间不可直接比较）。
        """
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

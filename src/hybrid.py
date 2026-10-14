"""
混合检索（向量 + 关键词 + 景点名精确匹配）
=========================================
三路信号融合，再按「每景点最多 N 条」做多样性重排。

为什么需要它（都是评测跑出来的结论，见 eval/results.md，不是拍脑袋）：
  1. 只用向量：MiniLM 是小模型，中文细粒度语义区分度有限，景点级召回偏低。
  2. 只用 BM25：用户「换个说法问」（纳凉 vs 避暑）时直接归零（Hit@5 = 0%）。
  3. ⭐ 只看词面还会被「他景点的评论」抢词：问「西安兵马俑值得去吗」，
     排在最前的却是「兵马俑+华清宫一日游」这种**跟团游记**（它们确实写了"兵马俑"），
     而兵马俑自己的评论反而没提到"兵马俑"三个字 ⇒ 纯 BM25 全落空。
  4. ⭐ 同一景点的多条评论会挤满 top-k，景点级覆盖率被浪费。

对应三个处理：
  · RRF 融合    ：对每个文档取它在两路里的排名，得分 = Σ 1/(RRF_K + rank)；
                  只用排名不用分数，省掉「余弦相似度 vs BM25 分」的量纲对齐。
  · 景点名加权  ：问题与景点名的最长公共子串 ≥ 3 字（且不等于城市名）时，
                  把该景点的评论直接提进候选池并加一个高分 —— 相当于轻量实体链接。
  · 多样性重排  ：同一景点最多留 MAX_PER_SPOT 条，让 top-k 覆盖更多景点。

用法：
    from hybrid import HybridRetriever
    r = HybridRetriever()
    hits = r.search("武汉哪里适合带孩子", top_k=5, city="武汉")
"""
import json

from bm25 import BM25
from config import (DATA_FILE, RRF_K, HYBRID_POOL, NAME_BOOST,
                    NAME_MATCH_MIN, MAX_PER_SPOT, KNOWN_CITIES)


def _key(hit):
    """同一景点、同一条评论可能被多路召回，用这个 key 合并。"""
    return (hit.get("spot", ""), hit.get("text", ""))


def _lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（滚动数组，O(len(a)×len(b))）。

    只用来判断「问题里是不是提到了某个景点名」，字符串都很短，够快。
    """
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


class HybridRetriever:
    #: 分数含义：RRF 融合分，**不是**余弦相似度，界面/日志里不能标成"相似度"
    score_label = "融合分"

    def __init__(self, docs=None, use_vector=True):
        if docs is None:
            docs = []
            with open(DATA_FILE, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line))
        self.docs = docs
        self.spots = sorted({d["spot"] for d in docs})
        self.bm25 = BM25(docs)

        self.vector = None
        self._vec_error = None
        if use_vector:
            try:
                from retriever import Retriever
                self.vector = Retriever()
            except Exception as e:                      # 缺依赖 / 缺索引 → 退化为纯 BM25
                self._vec_error = f"{type(e).__name__}: {e}"

    # ---------- 对外接口（与 Retriever 一致，可直接替换） ----------
    def cities(self):
        if self.vector is not None:
            return self.vector.cities()
        return sorted({d["city"] for d in self.docs if d.get("city")})

    def info(self):
        """当前生效的召回通道，供界面显示。"""
        if self.vector is None:
            return f"BM25（向量不可用：{self._vec_error}）"
        return "BM25 + 向量 + 景点名匹配（RRF 融合）"

    # ---------- 景点名精确匹配 ----------
    def _matched_spots(self, question: str):
        """问题里点名了哪些景点。

        ⚠️ 用最长公共子串而不是「包含」，因为景点名后面常带一句描述
        （如「成都动物园看看金丝猴馆和熊猫馆」），全名不可能出现在问题里。

        ⚠️ 必须排除「匹配到的就是城市名」的情况：否则问「西安兵马俑值得去吗」
        会命中「西安城墙」（公共子串 2 字「西安」）。这里用 NAME_MATCH_MIN=3 挡掉，
        同时显式排除城市名，双保险。
        """
        out = set()
        for s in self.spots:
            n = _lcs_len(question, s)
            if n >= NAME_MATCH_MIN and s not in KNOWN_CITIES:
                out.add(s)
        return out

    # ---------- 主流程 ----------
    def search(self, question: str, top_k: int = 5, city: str = None):
        """
        返回 top_k 条：
        [{"text":…, "spot":…, "city":…, "score":…, "from": "…"}, …]
        score 为融合分（越大越靠前，不同查询间不可直接比较）。
        """
        pool = max(top_k * 4, HYBRID_POOL)
        ranks = {}          # key -> {"bm25": rank, "vec": rank}
        store = {}          # key -> 原始 hit

        def add(hit, src, r):
            k = _key(hit)
            ranks.setdefault(k, {})[src] = r
            store.setdefault(k, hit)

        for r, h in enumerate(self.bm25.search(question, top_k=pool, city=city), 1):
            add(h, "bm25", r)

        if self.vector is not None:
            for r, h in enumerate(self.vector.search(question, top_k=pool, city=city), 1):
                add(h, "vec", r)

        # 景点名命中 → 把该景点的评论直接提进候选池
        named = self._matched_spots(question)
        if named:
            for d in self.docs:
                if d["spot"] in named and (not city or d["city"] == city):
                    add(d, "name", 1_000_000)      # 伪排名：不影响其他两路，只保证进池

        fused = []
        for k, rs in ranks.items():
            real = {s: r for s, r in rs.items() if s != "name"}
            score = sum(1.0 / (RRF_K + r) for r in real.values())
            if k[0] in named:
                score += NAME_BOOST
            fused.append((score, k))
        # 稳定排序：先按融合分降序，再按 key 兜底，保证多次运行结果一致
        fused.sort(key=lambda t: (-t[0], t[1]))

        # 多样性重排：同一景点最多 MAX_PER_SPOT 条。
        # 例外：问题里**点名**的景点不限条数 —— 问「黄鹤楼值不值得去」时，
        # 用户要的就是多几条黄鹤楼的评论，而不是硬凑 5 个不同景点。
        hits, used = [], {}
        for score, k in fused:
            spot = k[0]
            cap = top_k if spot in named else MAX_PER_SPOT
            if used.get(spot, 0) >= cap:
                continue
            used[spot] = used.get(spot, 0) + 1
            h = dict(store[k])
            h["score"] = round(score, 5)
            h["from"] = "+".join(sorted(s for s in ranks[k] if s != "name")) or "name"
            hits.append(h)
            if len(hits) >= top_k:
                break
        return hits

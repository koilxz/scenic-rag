"""
检索模块
========
给定用户问题，从向量库里找语义最相关的评论。

用法：
    from retriever import Retriever
    r = Retriever()
    hits = r.search("武汉哪个公园适合带孩子", top_k=5)
"""
from sentence_transformers import SentenceTransformer
import chromadb

from config import CHROMA_DIR, COLLECTION, EMBED_MODEL, TOP_K


class Retriever:
    def __init__(self, embed_model: str = EMBED_MODEL):
        self.model = SentenceTransformer(embed_model)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = client.get_collection(COLLECTION)

    def search(self, question: str, top_k: int = TOP_K, city: str = None):
        """
        question: 用户问题
        top_k:    返回条数
        city:     只在该城市内检索（None = 全库检索）

        返回：
        [
          {"text": 评论原文, "spot": 景点名, "city": 城市, "score": 相似度},
          ...
        ]
        """
        q_vec = self.model.encode([question], normalize_embeddings=True).tolist()

        kwargs = {"query_embeddings": q_vec, "n_results": top_k}
        if city:
            # 元数据过滤：只在该城市的评论里检索
            kwargs["where"] = {"city": city}

        res = self.col.query(**kwargs)

        hits = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({
                "text": doc,
                "spot": meta.get("spot", ""),
                "city": meta.get("city", ""),
                "score": round(1 - dist, 4),   # 余弦距离 → 相似度
            })
        return hits

    def cities(self):
        """返回库里有哪些城市（供界面下拉选择）"""
        metas = self.col.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("city", "") for m in metas if m.get("city")})


if __name__ == "__main__":
    r = Retriever()
    for h in r.search("武汉哪个公园适合带孩子"):
        print(f"[{h['score']}] [{h['city']}] {h['spot']}: {h['text'][:60]}")

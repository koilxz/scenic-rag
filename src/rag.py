"""
RAG 主链路
==========
检索 → 组装 prompt → 调用 LLM 生成答案（附原文出处）

用法：
    python src/rag.py "武汉哪个公园适合带孩子"
    python src/rag.py                      # 交互模式
"""
import argparse
import sys

from retriever import Retriever
from llm import get_llm
from config import TOP_K

SYSTEM_PROMPT = """你是一名旅游顾问。你的回答必须**严格基于**下面提供的游客评论。

规则：
1. 只根据提供的评论内容回答，不要添加评论里没有的信息。
2. 如果评论中没有相关信息，直接说"这些评论中没有提到相关信息"。
3. 回答要具体：提到能支撑结论的关键评论内容。
4. 最后单独用一行列出你参考了哪些景点。
5. 用简洁的中文回答，不要编造数字或事实。"""

# 库里出现的城市名。
# ⚠️ 曾经这里写死了一份列表，app.py 又写了一份，容易与实际数据不同步。
#    现在改为「优先从向量库动态读取」，读取失败才回退到这份兜底名单。
_FALLBACK_CITIES = ["北京", "上海", "武汉", "成都", "西安", "重庆", "青岛", "杭州", "济南", "南京"]

_KNOWN_CITIES_CACHE = None


def known_cities(retriever=None):
    """返回库里实际存在的城市列表（带缓存）。

    优先问向量库要，避免「代码里写的城市」和「库里真实的城市」对不上。
    """
    global _KNOWN_CITIES_CACHE
    if _KNOWN_CITIES_CACHE is not None:
        return _KNOWN_CITIES_CACHE

    cities = None
    if retriever is not None:
        try:
            cities = retriever.cities()
        except Exception:
            cities = None
    _KNOWN_CITIES_CACHE = cities or list(_FALLBACK_CITIES)
    return _KNOWN_CITIES_CACHE


def detect_city(question: str, cities=None):
    """从问题里识别城市名；识别不到返回 None（全库检索）。

    ⚠️ 按城市名长度「从长到短」匹配：避免短名是长名前缀时误判
    （目前 10 个城市都是 2 字，但将来若加入「呼和浩特」这类 3 字城市，
     必须保证「最具体优先」，否则逻辑上不严谨）。
    """
    cands = cities or known_cities()
    for c in sorted(cands, key=len, reverse=True):
        if c in question:
            return c
    return None


def build_prompt(question: str, hits: list) -> str:
    """把检索到的评论拼成提示词"""
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"{i}. [{h['city']}·{h['spot']}] {h['text']}")
    context = "\n".join(blocks)
    return f"""以下是与问题相关的游客评论：

{context}

---
用户的问题：{question}

请基于以上评论回答。"""


class RagPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = get_llm()

    def ask(self, question: str, top_k: int = TOP_K, city: str = None, verbose: bool = False):
        """
        city: 指定城市则只在该城市检索；None 则自动从问题里识别。
        """
        if city is None:
            city = detect_city(question, self.retriever.cities())

        # ① 检索
        hits = self.retriever.search(question, top_k=top_k, city=city)

        # ② 组装 prompt
        user_prompt = build_prompt(question, hits)

        # ③ 生成
        answer = self.llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        if verbose:
            print("=" * 60)
            print(f"检索范围：{city if city else '全库（10城）'}")
            print("检索到的评论：")
            for i, h in enumerate(hits, 1):
                print(f"  {i}. [{h['score']}] [{h['city']}] {h['spot']}: {h['text'][:50]}...")
            print("=" * 60)

        return answer, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="要问的问题")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K)
    parser.add_argument("-c", "--city", default=None,
                        help="限定城市（默认自动从问题里识别）")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示检索详情")
    args = parser.parse_args()

    print("正在初始化 RAG 系统……")
    rag = RagPipeline()
    print(f"使用模型后端：{rag.llm.name}")

    if args.question:
        answer, hits = rag.ask(args.question, args.top_k, args.city, args.verbose)
        print("\n【回答】")
        print(answer)
        print("\n【参考来源】")
        for h in hits:
            print(f"  · {h['city']} - {h['spot']}（相似度 {h['score']}）")
    else:
        # 交互模式
        print("进入交互模式（输入 q 退出）\n")
        while True:
            try:
                q = input("你问：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("q", "quit", "exit") or not q:
                break
            answer, _ = rag.ask(q, args.top_k, args.city)
            print(f"\n回答：{answer}\n")


if __name__ == "__main__":
    main()

"""
检索效果评估
============
量化「关键词检索」与「向量检索」在 42 条查询上的表现，给出可复现的数字。

对比系统
--------
  · BM25        —— 关键词检索（demo/demo_search.py），作为基线
  · 向量检索     —— 项目实际使用的链路（src/retriever.py：MiniLM + ChromaDB），按问题里
                   识别到的城市做元数据过滤
  · 向量·全库    —— 同上但不做城市过滤，用来观察「城市过滤」这一步值多少

真值（gold）怎么来的
-------------------
全部写在 eval/queries.jsonl 里，可复现：
  · T1「精确提及」：直接指定 gold 景点（问题里出现了景点名）
  · T2「城市+主题」/ T3「跨城主题」：给出关键词与阈值 min_hits，
    脚本扫描语料，把「有 ≥ min_hits 条评论命中关键词」的景点算作 gold。
⚠️ 这是**规则标注**（弱监督），不是人工逐条判读；它衡量的是
   「能否把含有相关信息的景点捞到前面」，局限性已写入 README。

指标（景点级）
-------------
  Hit@k     top-k 里是否至少捞到 1 个 gold 景点
  Recall@k  top-k 覆盖了 gold 景点集合的多大比例
  MRR       第一个 gold 景点的排名倒数（0 = 完全没捞到）

用法
----
    python eval/run_eval.py
    python eval/run_eval.py --top 5
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "comments.jsonl"
QUERIES = ROOT / "eval" / "queries.jsonl"
OUT_DIR = ROOT / "results" / "eval"
OUT_MD = ROOT / "eval" / "results.md"

# 让 src/ 与 demo/ 里的模块可以被 import
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))

MAX_K = 10          # 一次检索的最大条数，指标在 1..MAX_K 上切片算
KS = [1, 3, 5]


# ---------------- 数据 ----------------
def load_docs():
    docs = []
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def load_queries():
    qs = []
    with QUERIES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs


def build_gold(queries, docs):
    """把每条查询解析成 gold 景点集合，并做防呆校验（gold 不能为空）。"""
    by_city = defaultdict(list)
    for d in docs:
        by_city[d["city"]].append(d)

    out = []
    for q in queries:
        if "spots" in q:                       # T1：显式给定
            gold = set(q["spots"])
        else:                                  # T2/T3：规则标注
            pool = by_city.get(q["city"], []) if q.get("city") else docs
            cnt = Counter()
            for d in pool:
                if any(k in d["text"] for k in q["keywords"]):
                    cnt[d["spot"]] += 1
            gold = {s for s, n in cnt.items() if n >= q["min_hits"]}
        if not gold:
            raise ValueError(f"{q['id']} 的 gold 为空，请调整关键词/阈值：{q['question']}")
        out.append({**q, "gold": gold})
    return out


# ---------------- 检索系统 ----------------
class VectorSystem:
    """项目实际链路：MiniLM 向量 + ChromaDB（可选按城市过滤）。"""

    def __init__(self):
        from retriever import Retriever          # noqa: WPS433 (延迟 import，缺依赖时能优雅报错)
        self.r = Retriever()

    def search(self, question, k, city=None):
        return self.r.search(question, top_k=k, city=city)


class BM25System:
    """关键词基线。"""

    def __init__(self, docs):
        from bm25 import BM25
        self.bm = BM25(docs)

    def search(self, question, k, city=None):
        return self.bm.search(question, top_k=k, city=city)


class HybridSystem:
    """线上实际使用的链路：BM25 + 向量，RRF 融合。"""

    def __init__(self):
        from hybrid import HybridRetriever
        self.r = HybridRetriever()

    def search(self, question, k, city=None):
        return self.r.search(question, top_k=k, city=city)


# ---------------- 指标 ----------------
def evaluate(system, queries, use_city_filter=True, max_k=MAX_K):
    """返回 {query_id: {metric: value}} 与 top-k 明细。"""
    per_query = {}
    for q in queries:
        city = q.get("city") if use_city_filter else None
        hits = system.search(q["question"], max_k, city=city)
        ranks = [i + 1 for i, h in enumerate(hits) if h["spot"] in q["gold"]]

        row = {}
        for k in KS:
            top = hits[:k]
            got = {h["spot"] for h in top} & q["gold"]
            row[f"hit@{k}"] = 1.0 if got else 0.0
            row[f"recall@{k}"] = len(got) / len(q["gold"])
        row["mrr"] = (1.0 / ranks[0]) if ranks else 0.0
        row["_hits"] = [(h["spot"], round(h.get("score", 0), 4)) for h in hits[:5]]
        per_query[q["id"]] = row
    return per_query


def agg(per_query, ids):
    keys = [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr"]
    n = len(ids)
    return {k: sum(per_query[i][k] for i in ids) / n for k in keys}


# ---------------- 输出 ----------------
def fmt_table(systems, queries):
    """systems: [(name, per_query_dict)]。返回 markdown 表格行列表。"""
    tiers = ["T1", "T2", "T3", "T4", "ALL"]
    by_tier = {}
    for t in ["T1", "T2", "T3", "T4"]:
        by_tier[t] = [q["id"] for q in queries if q["tier"] == t]
    by_tier["ALL"] = [q["id"] for q in queries]

    lines = []
    cols = [f"Hit@1", f"Hit@3", f"Hit@5", f"Recall@5", "MRR"]
    lines.append("| 分组 | 系统 | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "---|" * len(cols))
    for t in tiers:
        ids = by_tier[t]
        for name, pq in systems:
            a = agg(pq, ids)
            lines.append(
                f"| {t} | {name} | "
                f"{a['hit@1']*100:.1f}% | {a['hit@3']*100:.1f}% | {a['hit@5']*100:.1f}% | "
                f"{a['recall@5']*100:.1f}% | {a['mrr']:.3f} |"
            )
    return lines, by_tier


def plot(systems, by_tier, top_k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

    names = [n for n, _ in systems]
    colors = ["#8D99AE", "#2E4057", "#2A9D8F", "#E8871E"]
    outline = "#B4560F"          # 给线上链路描边，一眼看出哪根是实际用的

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- 左：总体指标 ----
    metrics = ["Hit@1", "Hit@5", "Recall@5", "MRR"]
    keys = ["hit@1", "hit@5", "recall@5", "mrr"]
    ids = by_tier["ALL"]
    x = np.arange(len(metrics))
    w = 0.8 / len(names)
    for i, (name, pq) in enumerate(systems):
        a = agg(pq, ids)
        vals = [a[k] * 100 for k in keys]        # MRR 也放成百分数便于同轴比较
        bars = ax1.bar(x + (i - (len(names) - 1) / 2) * w, vals, w,
                       label=name, color=colors[i % len(colors)],
                       edgecolor=outline if name.startswith("混合") else "none",
                       linewidth=1.6)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11)
    ax1.set_ylabel("百分比（%）", fontsize=11)
    ax1.set_ylim(0, 110)
    ax1.set_title(f"总体（{len(ids)} 条查询）", fontsize=12.5, fontweight="bold")
    ax1.legend(fontsize=9, frameon=False, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.25)

    # ---- 右：分档 Hit@5（T4 是「换个说法问」的硬骨头）----
    tier_labels = ["T1 精确提及", "T2 城市+主题", "T3 跨城主题", "T4 语义改写"]
    tkeys = ["T1", "T2", "T3", "T4"]
    x2 = np.arange(len(tkeys))
    for i, (name, pq) in enumerate(systems):
        vals = [agg(pq, by_tier[t])["hit@5"] * 100 for t in tkeys]
        bars = ax2.bar(x2 + (i - (len(names) - 1) / 2) * w, vals, w,
                       color=colors[i % len(colors)], label=name,
                       edgecolor=outline if name.startswith("混合") else "none",
                       linewidth=1.6)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(tier_labels, fontsize=10)
    ax2.set_ylabel("Hit@5（%）", fontsize=11)
    ax2.set_ylim(0, 118)
    ax2.set_title("分档 Hit@5：越往右越考验「语义」", fontsize=12.5, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("检索效果对比 · 景点级（命中判定：检索到的评论是否来自正确景点）",
                 fontsize=13, fontweight="bold", y=1.02)
    # 注脚数字直接从结果里取，避免写死后与图表脱节
    t4 = {n: agg(pq, by_tier["T4"])["hit@5"] * 100 for n, pq in systems}
    fig.text(0.99, -0.07,
             "注：T4 全用「换个说法」提问（如「纳凉」而非「避暑」）—— 关键词检索 Hit@5 为 "
             f"{t4.get('BM25（关键词基线）', 0):.0f}%，"
             f"向量 {t4.get('向量检索', 0):.0f}%，"
             f"混合 {t4.get('混合检索（线上）', 0):.0f}%。",
             ha="right", fontsize=9, color="#B4560F")


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "retrieval_metrics.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5, help="明细里展示的检索条数")
    args = ap.parse_args()

    docs = load_docs()
    queries = build_gold(load_queries(), docs)
    print(f"语料 {len(docs)} 条 / 景点 {len({d['spot'] for d in docs})} 个 / 查询 {len(queries)} 条")
    print(f"gold 规模：最小 {min(len(q['gold']) for q in queries)}，"
          f"最大 {max(len(q['gold']) for q in queries)}，"
          f"平均 {sum(len(q['gold']) for q in queries)/len(queries):.1f}\n")

    systems = []
    bm = BM25System(docs)
    systems.append(("BM25（关键词基线）", evaluate(bm, queries, use_city_filter=True)))
    try:
        vec = VectorSystem()
        systems.append(("向量检索", evaluate(vec, queries, use_city_filter=True)))
        systems.append(("向量·不过滤城市", evaluate(vec, queries, use_city_filter=False)))
    except Exception as e:                          # 缺依赖/索引时优雅降级
        print(f"⚠️ 向量检索不可用，跳过：{type(e).__name__}: {e}")

    try:
        hyb = HybridSystem()
        systems.append(("混合检索（线上）", evaluate(hyb, queries, use_city_filter=True)))
    except Exception as e:
        print(f"⚠️ 混合检索不可用，跳过：{type(e).__name__}: {e}")

    lines, by_tier = fmt_table(systems, queries)

    print("\n===== 结果 =====")
    for ln in lines:
        print(ln)

    png = plot(systems, by_tier, args.top)
    print(f"\n图表已保存：{png.relative_to(ROOT)}")

    # ---- 写 results.md ----
    md = ["# 检索效果评估结果", "",
          f"- 语料：**{len(docs)}** 条评论 / **{len({d['spot'] for d in docs})}** 个景点 / "
          f"**{len({d['city'] for d in docs})}** 个城市",
          f"- 查询：**{len(queries)}** 条（T1 精确提及 {len(by_tier['T1'])} · "
          f"T2 城市+主题 {len(by_tier['T2'])} · T3 跨城主题 {len(by_tier['T3'])} · "
          f"T4 语义改写 {len(by_tier['T4'])}）",
          "- 指标为**景点级**：命中判定 = 检索到的评论是否来自 gold 景点",
          "- 真值由 `eval/queries.jsonl` 的规则标注（可复现），非人工逐条标注",
          "", "## 总表", ""]
    md += [ln.replace("| 混合检索（线上） |", "| **混合检索（线上）** |") for ln in lines]
    md += ["", "## 每条查询明细（线上链路：混合检索）", "",
           "| 查询 | 城市 | gold 数 | Hit@1 | Hit@3 | Hit@5 | MRR | top-3 检索到的景点 |",
           "|---|---|---|---|---|---|---|---|"]
    vec_pq = (dict(systems).get("混合检索（线上）")
              or dict(systems).get("向量检索"))
    for q in queries:
        r = vec_pq[q["id"]]
        hits = "、".join(s for s, _ in r["_hits"][:3])
        md.append(
            f"| {q['question']} | {q.get('city') or '—'} | {len(q['gold'])} | "
            f"{r['hit@1']:.0f} | {r['hit@3']:.0f} | {r['hit@5']:.0f} | "
            f"{r['mrr']:.2f} | {hits} |"
        )
    md.append("")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"报告已保存：{OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())

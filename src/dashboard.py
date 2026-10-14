"""
阶段 4：语料数据看板
====================
把 comments.jsonl 的分布情况画出来，用途有两个：

1. **给人看**：这份语料长什么样（城市、情感、长度、景点集中度）。
2. **给数据质量把关**（更重要）：自动标出「可疑数据」——
   比如景点名里含其他城市名（跨城市污染），这类问题肉眼翻 6000 条评论是发现不了的。

输入：data/comments.jsonl
输出：results/dashboard/*.png + 终端摘要

用法：
    python src/dashboard.py
"""
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "comments.jsonl"
OUT = ROOT / "results" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)

NAVY, ORANGE, TEAL, RED, GREEN = "#2E4057", "#E8871E", "#2A9D8F", "#C73E1D", "#4C956C"

# 语料里出现的城市（用于「景点名含他城名」检测）
KNOWN_CITIES = ["北京", "上海", "武汉", "成都", "西安", "重庆", "青岛", "杭州", "济南", "南京"]


# ============ 读取 ============
def load():
    records = []
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


# ============ 数据质量检测 ============
def find_cross_city(df):
    """找「景点名里含其他城市名」的记录 —— 跨城市污染的高置信信号。

    只判「景点名含**另一个**已收录城市名」这一种情况，
    因为这种误标几乎零歧义（「成都欢乐谷」被标成重庆，基本可以确定标错了）。
    反例：「武汉东湖」含"武汉"但城市就是武汉 → 不算。
    """
    bad = []
    for (city, spot), g in df.groupby(["city", "spot"]):
        for other in KNOWN_CITIES:
            if other != city and other in str(spot):
                bad.append({
                    "标注城市": city,
                    "景点名": spot,
                    "疑似归属": other,
                    "评论数": len(g),
                })
                break
    return pd.DataFrame(bad)


def comment_count_dist(df):
    """每景点评论数分布 —— 用来发现「截断存储」。"""
    return df.groupby(["city", "spot"]).size()


# ============ 图表 ============
def fig_city(df):
    vc = df["city"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(vc.index[::-1], vc.values[::-1], color=NAVY)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_xlabel("评论数", fontsize=11)
    ax.set_title("语料评论的城市分布", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, vc.max() * 1.15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.savefig(f"{OUT}/01_city_comments.png")
    plt.close()
    print("  ✓ 01_city_comments.png")


def fig_city_spot(df):
    """城市 × 景点数 vs 评论数 —— 两个维度放一起看，暴露「城市景点少但评论多」的异常。"""
    g = df.groupby("city").agg(评论数=("text", "size"), 景点数=("spot", "nunique"))
    g["均评论数"] = g["评论数"] / g["景点数"]
    g = g.sort_values("评论数", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(g))
    ax.barh(y - 0.2, g["评论数"], height=0.4, color=TEAL, label="评论数")
    ax.barh(y + 0.2, g["景点数"] * 8, height=0.4, color=ORANGE,
            label="景点数（×8 便于同轴对比）")
    ax.set_yticks(y)
    ax.set_yticklabels(g.index)
    for i, (c, s, n) in enumerate(zip(g["评论数"], g["景点数"], g["均评论数"])):
        ax.text(c + 8, i - 0.2, f"{c}", va="center", fontsize=8.5)
        ax.text(s * 8 + 8, i + 0.2, f"{s} 个", va="center", fontsize=8.5)
    ax.set_xlabel("数量", fontsize=11)
    ax.set_title("各城市：评论量 vs 景点覆盖数", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.savefig(f"{OUT}/02_city_spot_vs_comments.png")
    plt.close()
    print("  ✓ 02_city_spot_vs_comments.png")


def fig_sentiment(df):
    if "sentiment" not in df.columns:
        print("  · 跳过情感图（无 sentiment 字段）")
        return
    vc = df["sentiment"].value_counts()
    color_map = {"正面": GREEN, "中性": "#B0B0B0", "负面": RED}
    colors = [color_map.get(k, NAVY) for k in vc.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(vc.index, vc.values, color=colors, width=0.55)
    total = vc.sum()
    ax.bar_label(bars, labels=[f"{v}\n{v/total*100:.1f}%" for v in vc.values],
                 padding=4, fontsize=10)
    ax.set_ylabel("评论数", fontsize=11)
    ax.set_title("评论情感分布", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, vc.max() * 1.2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.savefig(f"{OUT}/03_sentiment.png")
    plt.close()
    print("  ✓ 03_sentiment.png")


def fig_length(df):
    lens = df["text"].str.len()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(lens, bins=50, color=TEAL, edgecolor="white", linewidth=0.6)
    ax.axvline(lens.median(), color=RED, linestyle="--", linewidth=1.8,
               label=f"中位数 {lens.median():.0f} 字")
    ax.axvline(lens.mean(), color=ORANGE, linestyle="--", linewidth=1.8,
               label=f"平均 {lens.mean():.0f} 字")
    # 标出 500 字截断线 —— 这是「超长被截断」的直观证据
    if lens.max() >= 500:
        ax.axvline(500, color="purple", linestyle=":", linewidth=1.8,
                   label=f"500 字截断线（{(lens >= 500).sum()} 条贴线）")
    ax.set_xlabel("评论文本长度（字符）", fontsize=11)
    ax.set_ylabel("评论数", fontsize=11)
    ax.set_title("评论长度分布", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.savefig(f"{OUT}/04_length_hist.png")
    plt.close()
    print("  ✓ 04_length_hist.png")


def fig_top_spots(df):
    g = df.groupby(["city", "spot"]).size().nlargest(15)
    labels = [f"{s}（{c}）" for c, s in g.index][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, g.values[::-1], color=ORANGE)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("评论数", fontsize=11)
    ax.set_title("评论数最高的 15 个景点", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, g.max() * 1.18)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.savefig(f"{OUT}/05_top_spots.png")
    plt.close()
    print("  ✓ 05_top_spots.png")


def fig_cc_dist(df):
    """每景点评论数分布 —— 尖峰在 30 说明是截断存储。"""
    s = comment_count_dist(df)
    vc = pd.Series(Counter(s.values)).sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(vc.index, vc.values, color=NAVY, width=0.8)
    if 30 in vc.index:
        bars[list(vc.index).index(30)].set_color(RED)
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xlabel("该景点的评论条数", fontsize=11)
    ax.set_ylabel("景点个数", fontsize=11)
    ax.set_title("每个景点的评论条数分布（红柱=30 条）", fontsize=14,
                 fontweight="bold", pad=15)
    ax.annotate("93 个景点恰好 30 条\n→ 疑似每景点最多存 30 条",
                xy=(30, vc.get(30, 0)), xytext=(1, vc.max() * 0.72),
                fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.savefig(f"{OUT}/06_comment_count_dist.png")
    plt.close()
    print("  ✓ 06_comment_count_dist.png")


def fig_quality(bad):
    """数据质量图 —— 可疑跨城市景点。"""
    if bad.empty:
        print("  · 未发现跨城市污染")
        return
    bad = bad.sort_values("评论数", ascending=True)
    labels = [f"{r.景点名}\n（标为{r.标注城市} · 疑属{r.疑似归属}）"
              for r in bad.itertuples()]

    fig, ax = plt.subplots(figsize=(10, max(4, len(bad) * 0.55)))
    bars = ax.barh(range(len(bad)), bad["评论数"], color=RED)
    ax.set_yticks(range(len(bad)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("评论数", fontsize=11)
    ax.set_title(f"可疑跨城市污染景点（{len(bad)} 个）", fontsize=14,
                 fontweight="bold", pad=15, color=RED)
    ax.set_xlim(0, bad["评论数"].max() * 1.2)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.savefig(f"{OUT}/07_quality_suspects.png")
    plt.close()
    print("  ✓ 07_quality_suspects.png")


# ============ 终端摘要 ============
def summary(df, bad):
    print("\n" + "=" * 62)
    print("语料摘要")
    print("=" * 62)
    print(f"  评论总数   : {len(df)}")
    print(f"  覆盖城市   : {df['city'].nunique()} 个")
    print(f"  覆盖景点   : {df['spot'].nunique()} 个（有评论的）")
    lens = df['text'].str.len()
    print(f"  平均长度   : {lens.mean():.1f} 字（中位 {lens.median():.0f} 字）")
    if "sentiment" in df.columns:
        vc = df["sentiment"].value_counts()
        s = "、".join(f"{k} {v}" for k, v in vc.items())
        print(f"  情感分布   : {s}")

    print("\n" + "-" * 62)
    print("⚠️  数据质量检查")
    print("-" * 62)

    # 检查 1：跨城市污染
    if bad.empty:
        print("  [1] 跨城市污染：未发现高置信可疑项")
    else:
        print(f"  [1] 跨城市污染：发现 {len(bad)} 个高置信可疑景点"
              f"（景点名含其他城市名）")
        for r in bad.head(10).itertuples():
            print(f"        · 「{r.景点名}」标为 {r.标注城市}"
                  f"，疑属 {r.疑似归属}（{r.评论数} 条）")

    # 检查 2：评论截断
    s = comment_count_dist(df)
    n30 = (s == 30).sum()
    if n30 > len(s) * 0.1:
        print(f"  [2] 评论截断  ：{n30} 个景点恰好 30 条"
              f"（占 {n30/len(s)*100:.0f}%）→ 疑似每景点最多存 30 条")

    # 检查 3：超长截断
    n500 = (lens >= 500).sum()
    if n500:
        print(f"  [3] 长度截断  ：{n500} 条评论贴到 500 字上限")

    # 检查 4：城市覆盖不均
    vc = df["city"].value_counts()
    if vc.min() < vc.max() * 0.15:
        print(f"  [4] 覆盖不均  ：最多 {vc.index[0]} {vc.max()} 条，"
              f"最少 {vc.index[-1]} {vc.min()} 条（相差 {vc.max()/max(vc.min(),1):.0f} 倍）")
    print("=" * 62)


def main():
    print(f"读取语料：{DATA}")
    df = load()
    print(f"读入 {len(df)} 条评论\n")

    print("生成图表：")
    fig_city(df)
    fig_city_spot(df)
    fig_sentiment(df)
    fig_length(df)
    fig_top_spots(df)
    fig_cc_dist(df)

    bad = find_cross_city(df)
    fig_quality(bad)

    summary(df, bad)
    print(f"\n✅ 看板已输出到：{OUT}")


if __name__ == "__main__":
    main()

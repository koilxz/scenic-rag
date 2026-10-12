"""
阶段 0：从一号作品导出评论语料
=================================
读取 city-attraction-analysis 的清洗后数据，
把 all_comments 字段（| 分隔的多条评论）拆成独立记录，
清洗后导出为 comments.jsonl，供 RAG 建索引使用。

用法：
    python src/export_comments.py
"""
import json
import re
from pathlib import Path

import pandas as pd

# 一号作品的数据位置
SRC = Path(__file__).parent.parent.parent / "city-attraction-analysis" / "data" / "processed" / "scenic_spots_clean.xlsx"
OUT = Path(__file__).parent.parent / "data" / "comments.jsonl"

# 清洗参数
MIN_LEN = 5          # 最短有效评论长度（字符）
MAX_LEN = 500        # 超长截断（避免单条吃掉整个上下文窗口）


def is_meaningful(text: str) -> bool:
    """判断一条评论是否有意义（去除纯符号、纯数字、过短）"""
    t = text.strip()
    if len(t) < MIN_LEN:
        return False
    # 中文或英文字母数量占比过低 → 视为无意义
    alnum = len(re.findall(r"[\u4e00-\u9fa5a-zA-Z]", t))
    if alnum / max(len(t), 1) < 0.4:
        return False
    return True


def main():
    print(f"读取：{SRC}")
    df = pd.read_excel(SRC)
    print(f"景点总数：{len(df)}")

    records = []
    skipped_short = 0
    skipped_empty = 0

    for _, row in df.iterrows():
        raw = row.get("all_comments")
        if pd.isna(raw):
            skipped_empty += 1
            continue

        text = str(raw).strip()
        if not text or text == "nan":
            skipped_empty += 1
            continue

        # 按 | 拆分（同时兼容 || ）
        parts = re.split(r"\|+", text)
        for p in parts:
            c = p.strip()
            if not c:
                continue
            if not is_meaningful(c):
                skipped_short += 1
                continue
            if len(c) > MAX_LEN:
                c = c[:MAX_LEN]

            records.append({
                "spot": str(row.get("name", "")).strip(),
                "city": str(row.get("city", "")).strip(),
                "rating": float(row["rating"]) if pd.notna(row.get("rating")) else None,
                "sentiment": str(row.get("sentiment_label", "")).strip(),
                "text": c,
            })

    # 去重（同一景点内完全相同的评论）
    seen = set()
    deduped = []
    for r in records:
        key = (r["spot"], r["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 50)
    print(f"无评论景点：{skipped_empty}")
    print(f"过短/无意义评论剔除：{skipped_short}")
    print(f"去重前：{len(records)}  →  去重后：{len(deduped)}")
    print(f"✅ 导出完成：{OUT}")
    print(f"   共 {len(deduped)} 条评论，来自 {df['city'].nunique()} 个城市")


if __name__ == "__main__":
    main()

"""
建索引：评论 → 向量 → ChromaDB
=============================
读取 data/comments.jsonl，用多语言模型把每条评论编码成向量，
存入本地 ChromaDB 持久化目录，供检索使用。

用法：
    python src/build_index.py            # 全量重建
    python src/build_index.py --rebuild  # 删除旧库后重建
"""
import argparse
import json
import shutil
import time

import chromadb
from sentence_transformers import SentenceTransformer

from config import DATA_FILE, CHROMA_DIR, COLLECTION, EMBED_MODEL


def load_comments():
    """读取 jsonl 评论"""
    rows = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="删除旧向量库后重建")
    args = parser.parse_args()

    # 可选：清空旧库
    if args.rebuild and CHROMA_DIR.exists():
        print(f"删除旧向量库：{CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    # 1. 读数据
    rows = load_comments()
    print(f"读取评论：{len(rows)} 条")

    texts = [r["text"] for r in rows]
    metadatas = [{"spot": r["spot"], "city": r["city"]} for r in rows]
    ids = [f"c{i}" for i in range(len(rows))]

    # 2. 加载向量模型（首次会下载 ~118MB）
    print(f"加载向量模型：{EMBED_MODEL}")
    print("（首次运行需下载模型，请耐心等待……）")
    model = SentenceTransformer(EMBED_MODEL)

    # 3. 向量化
    t0 = time.time()
    print("正在向量化……")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # 归一化后可用余弦相似度
    )
    print(f"向量化完成，耗时 {time.time() - t0:.1f} 秒")

    # 4. 写入 ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # 用余弦相似度
    )

    # 分批写入（避免单次过大）
    BATCH = 1000
    for i in range(0, len(ids), BATCH):
        col.add(
            ids=ids[i:i + BATCH],
            documents=texts[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH].tolist(),
            metadatas=metadatas[i:i + BATCH],
        )
        print(f"  已写入 {min(i + BATCH, len(ids))}/{len(ids)}")

    # ⚠️ 关键：ChromaDB 1.x（Rust 内核）的 HNSW 索引是**惰性构建 + 异步落盘**的。
    #    只写入、不查询的话，索引二进制文件（data_level0.bin 等）不会生成，
    #    之后新进程打开会报 "Error loading hnsw index"。
    #    因此这里主动触发一次查询，强制构建索引，再等待其落盘。
    print("正在构建 HNSW 索引（触发查询 + 等待落盘）……")
    col.query(query_embeddings=[embeddings[0].tolist()], n_results=1)
    time.sleep(8)   # 给异步 compactor 时间把索引写入磁盘

    # 落盘校验
    idx_files = list(CHROMA_DIR.rglob("data_level0.bin"))
    if idx_files:
        print(f"✅ HNSW 索引已落盘：{idx_files[0].relative_to(CHROMA_DIR)}"
              f"（{idx_files[0].stat().st_size / 1e6:.1f} MB）")
    else:
        print("⚠️ 未检测到 HNSW 索引文件，请重新运行本脚本")

    print(f"✅ 建索引完成：{col.count()} 条向量已入库")
    print(f"   向量库位置：{CHROMA_DIR}")


if __name__ == "__main__":
    main()

"""
统一配置
========
所有可调参数集中在这里，改配置不用翻代码。
"""
import os
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data" / "comments.jsonl"


def _safe_chroma_dir() -> Path:
    """
    向量库持久化目录。

    ⚠️ 重要：ChromaDB 1.x 的 Rust 内核**在非 ASCII（中文）路径下无法写入 HNSW 索引文件**，
       表现为新进程打开时抛 `Error loading hnsw index`。
       因此这里检测：若项目路径含非 ASCII 字符，则自动把向量库放到一个 ASCII 安全目录。

    可用环境变量 CHROMA_DIR 手动覆盖。
    """
    override = os.getenv("CHROMA_DIR")
    if override:
        return Path(override)

    try:
        str(ROOT).encode("ascii")
        return ROOT / "chroma_db"          # 项目路径是 ASCII，直接用项目内目录
    except UnicodeEncodeError:
        # 项目路径含中文 → 落到用户目录下的 ASCII 路径
        fallback = Path.home() / ".scenic_rag" / "chroma_db"
        return fallback


CHROMA_DIR = _safe_chroma_dir()          # 向量库持久化目录

# ---------- Embedding ----------
# 多语言小模型：中文支持好、仅 ~118MB、CPU 可跑
#
# ⚠️ 网络说明：huggingface.co 在国内直连不通。
#   本项目把模型文件手动下载到 models/minilm/ 本地目录，从本地加载，
#   避免 HF 下载器在镜像环境下的兼容问题（会写入 0 字节空文件）。
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    str(ROOT / "models" / "minilm") if (ROOT / "models" / "minilm").exists()
    else EMBED_MODEL_NAME,
)

# ---------- 向量库 ----------
COLLECTION = "scenic_comments"

# ---------- 检索 ----------
TOP_K = 5            # 检索返回的评论条数

# ---------- LLM ----------
# 通过环境变量 LLM_PROVIDER 切换：ollama（默认）/ cloud
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# 本地 Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

# 云端 API（OpenAI 兼容接口）
CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "https://api.deepseek.com")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "deepseek-chat")
CLOUD_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("CLOUD_API_KEY")

# ---------- 生成参数 ----------
TEMPERATURE = 0.3    # 低温度，减少发挥、更贴近原文
MAX_TOKENS = 800

# 景点评论 RAG 问答系统

> 用 **6316 条真实景点评论**搭建的检索增强生成（RAG）问答系统。
> 问它"武汉哪个公园适合带孩子""黄鹤楼值不值得去"，它会**基于真实游客评论**回答，并给出原文出处。

---

## 问题

数据来源：去哪儿网 10 个热门旅游城市（北京、上海、武汉、成都、西安、重庆、青岛、杭州、济南、南京）的景点评论。

游客在做决策时真正想知道的是：

1. **"这个景点到底值不值得去？"**
2. **"带小孩适合去哪？"**
3. **"门票贵不贵、有没有坑？"**

传统的做法是翻几十条评论自己看。这个项目用 **RAG** 让机器来读，直接给答案。

---

## 它是什么（RAG 一句话解释）

```
用户提问 → ① 向量化 → ② 检索最相关的评论 → ③ 连同问题喂给大模型 → ④ 基于评论生成答案（附出处）
```

**为什么要 RAG，不让大模型直接答？**
- 大模型**不知道**这些评论的内容（训练数据里没有）
- 大模型会**瞎编**（幻觉）——RAG 用"只根据检索到的评论回答"来约束它
- 让答案**可追溯**：每条结论都能点回原文

---

## 架构

```
┌──────────────────────────────────────┐
│  Streamlit 网页界面                   │
└───────────────┬──────────────────────┘
                ↓
┌──────────────────────────────────────┐
│  RAG 核心                             │
│  Embedding: 本地 MiniLM 多语言模型     │
│  向量库:    ChromaDB（本地持久化）     │
│  LLM:       Ollama 本地 / 云端 API     │
└───────────────┬──────────────────────┘
                ↓
┌──────────────────────────────────────┐
│  6316 条评论 → 分块 → 向量 → 入库      │
└──────────────────────────────────────┘
```

**设计要点：可插拔 LLM**
- 默认：本地 `Ollama` + `qwen2.5:1.5b`（集显笔记本可跑，零成本）
- 可选：云端 API（配了 `DEEPSEEK_API_KEY` 即自动切换）

---

## 快速开始

> ⚠️ **国内网络必读**：`huggingface.co` 与 `github.com` 直连不通，本文档所有下载步骤都已改用国内镜像。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备向量模型（手动下载，绕过 HF 下载器）

HF 官方下载器在镜像环境下会写出 0 字节空文件，因此**用 curl 逐个文件拉取**：

```bash
mkdir -p models/minilm && cd models/minilm
BASE="https://hf-mirror.com/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main"
for f in config.json config_sentence_transformers.json modules.json \
         sentence_bert_config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json model.safetensors; do
  curl -L -C - -o "$f" "$BASE/$f"
done
mkdir -p 1_Pooling
curl -L -o 1_Pooling/config.json "$BASE/1_Pooling/config.json"
```

> 合计约 460MB（其中 `model.safetensors` 449MB）。程序会**自动优先从本地 `models/minilm/` 加载**。

### 3. 建索引（把评论向量化入库）

```bash
python src/build_index.py
```

> ⚠️ **ChromaDB 中文路径限制**：ChromaDB 1.x 的 Rust 内核**无法在含中文的路径下写入 HNSW 索引**。
> 本项目已自动处理——若项目路径含中文，向量库会写到 `~/.scenic_rag/chroma_db`（可用环境变量 `CHROMA_DIR` 覆盖）。

### 4. 命令行问答

```bash
python src/rag.py "武汉哪个公园适合带孩子"
python src/rag.py "黄鹤楼值不值得去" -v        # -v 显示检索详情
python src/rag.py "有什么适合亲子的" -c 杭州   # -c 指定城市
```

### 5. 大模型后端（可选）

**默认离线模式（MockLLM）**：不依赖任何模型服务，把检索到的评论原文整理输出。
用于验证检索效果、跑通链路。

**接入真模型（二选一）**：

```bash
# 方案 A：本地 Ollama（零成本，集显可跑）
ollama pull qwen2.5:1.5b
# 默认即为 ollama 后端，无需配置

# 方案 B：云端 API（效果最好）
set LLM_PROVIDER=cloud
set DEEPSEEK_API_KEY=你的key
```

后端按 `ollama → cloud → mock` 顺序自动选择，**任一不可用会自动回退**，不会报错中断。

### 6. 网页界面

```bash
streamlit run app.py
```

---

## 项目结构

```
scenic-rag/
├── data/
│   └── comments.jsonl        # 6316 条评论（从一号作品导出）
├── src/
│   ├── export_comments.py    # 导出评论数据
│   ├── build_index.py        # 建向量索引
│   ├── retriever.py          # 检索模块
│   ├── llm.py                # 可插拔 LLM（Ollama / 云端）
│   ├── rag.py                # RAG 主链路
│   └── config.py             # 配置
├── app.py                    # Streamlit 界面
└── eval/
    └── evaluate.py           # 检索效果评估
```

---

## 数据说明

| 项目 | 说明 |
|---|---|
| 原始评论 | 8227 条（含重复、过短） |
| **清洗后** | **6316 条**（去重 + 剔除 <5 字及纯符号） |
| 城市覆盖 | 10 城（武汉 889、重庆 821、杭州 809、北京 776…） |
| 总字符 | 约 47 万 |
| 平均长度 | 75 字 |

评论数据来自[一号作品](https://github.com/koilxz/city-attraction-analysis)的清洗结果。

---

## 功能

- 🔍 **语义检索**：基于向量相似度找相关评论，而非关键词匹配
- 🏙️ **城市过滤**：问"武汉…"只检索武汉的评论（从问题自动识别，也可手动指定）
- 📎 **答案可溯源**：每条回答都附带检索到的原始评论出处
- 🔌 **可插拔模型层**：本地 Ollama / 云端 API / 离线 Mock 三种后端，自动回退
- 🖥️ **网页界面**：Streamlit，支持城市选择、示例问题、来源展开

## 局限性（诚实说明）

1. **评论文本长度上限 500 字**：超长评论被截断，可能丢失后半段信息。
2. **本地 1.5B 模型能力有限**：默认模式的答案质量不如云端大模型，这是"能在集显笔记本上跑"的代价。
3. **离线（Mock）模式不生成**：它只整理检索到的原文，不做归纳润色 —— 仅用于验证检索环节。
4. **检索基于语义相似度**：对"多少钱"这类需要精确数字的问题，可能不如关键词检索准。
5. **源数据仍有残留归属错误**：一号作品的地标映射表只覆盖了 40+ 个知名地标，
   像「滕王阁」（南昌）、「平江石牛寨」（岳阳）这类未覆盖的景点仍被标错城市。
   这是检索结果里出现"答非所问"评论的一个来源。
6. **未做检索效果定量评估**（仅定性测试），后续可补召回率指标。

---

## 技术栈

| 环节 | 工具 |
|---|---|
| 向量化 | `sentence-transformers`（paraphrase-multilingual-MiniLM-L12-v2） |
| 向量库 | `ChromaDB` |
| LLM | `Ollama`（本地）/ OpenAI 兼容 API（云端） |
| 界面 | `Streamlit` |
| 数据处理 | `pandas` |

---

## 相关项目

- [城市景点数据分析（一号作品）](https://github.com/koilxz/city-attraction-analysis) — 本项目的评论数据来源

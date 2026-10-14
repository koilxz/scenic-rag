# 景点评论 RAG 问答系统

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-1C3C3C?logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-向量库-FF6B6B)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> 用 **6316 条真实景点评论**搭建的检索增强生成（RAG）问答系统。
> 问它"武汉哪个公园适合带孩子""黄鹤楼值不值得去"，它会**基于真实游客评论**回答，并给出原文出处。

**技术栈**：Python · LangChain · ChromaDB · sentence-transformers · Streamlit · Ollama / DeepSeek
**核心能力**：语义检索 + 城市限定过滤 + 出处可溯 + 数据质量自检

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

后端按 `ollama → cloud → mock` 顺序自动选择，**任一不可用会自动回退**，不会报错中断。

**方案 A：本地 Ollama（推荐，零成本、离线）** ✅ 已验证可用

```bash
ollama pull qwen2.5:1.5b     # 986MB
```

> ⚠️ **国内安装 Ollama 主程序的坑**：官网 `ollama.com/download` 会 307 跳转到 `github.com`（不通）。
> 命令行镜像（ghfast.top / ghproxy.net）实测只有 50–188KB/s，1.5GB 要 2–8 小时，**不实用**。
> ✅ **可行做法**：浏览器把梯子切到「系统代理模式」，直接访问 https://ollama.com/download/windows 下载。
> 装好后 `ollama pull` 走 `registry.ollama.ai`，**该域名国内可直连**，速度正常。

**方案 B：云端 API（效果最好）**

```bash
set LLM_PROVIDER=cloud
set DEEPSEEK_API_KEY=你的key
```

**方案 C：离线 Mock**：不依赖模型服务，把检索到的评论原文整理输出，用于验证检索效果。

### 6. 网页界面

```bash
streamlit run app.py
```

### 7. 数据看板（了解语料长什么样）

```bash
python src/dashboard.py
```

输出到 `results/dashboard/`：城市分布、城市覆盖 vs 评论量、情感分布、长度分布、
评论数 Top15、每景点评论数分布，以及一张 **`07_quality_suspects.png` 数据质量图**。

> ⭐ 这个脚本不只是"画图"，它内置了一道**数据质量自检**：
> 自动找出「景点名里含其他城市名」的记录 —— 也就是**跨城市污染**。
> 首次运行即抓出 **16 个可疑景点、330 条评论**（详见下节局限性第 5 条）。

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
└── data/                     # 语料（见上）
```

> 📌 `eval/` 与 `docs/` 目录**已预留但暂无实现** → 检索效果定量评估尚未做，见「局限性」第 6 条。

---

## 数据说明

| 项目 | 数值 |
|---|---|
| 源景点 | 455 个（来自一号作品清洗结果） |
| 有评论的景点 | 340 个（115 个景点无评论） |
| 评论原始条数 | 8219 条 |
| **清洗后** | **6316 条**（去重 6316 / 剔除过短·无意义 8 条） |
| 城市覆盖 | 10 城（武汉 889、重庆 821、杭州 809、北京 776、成都 702、青岛 681、西安 620、上海 529、济南 438、南京 51） |
| 平均长度 | 约 75 字 |

> ⚠️ **上面这组城市数字不能直接用来做"城市热度对比"** ——
> 因为存在跨城市污染，比如 **杭州的 809 条里有 128 条其实属于上海**。
> 详见「局限性」第 5 条。

> 以上数字由 `python src/export_comments.py` 重跑验证，**完全可复现**。

评论数据来自[一号作品](https://github.com/koilxz/city-attraction-analysis)的清洗结果。

---

## 功能

- 🔍 **语义检索**：基于向量相似度找相关评论，而非关键词匹配
- 🏙️ **城市过滤**：问"武汉…"只在该城市评论中检索（自动识别 / 手动指定）
- 📎 **答案可溯源**：每条回答都附带检索到的原始评论出处与相似度
- 🔌 **可插拔模型层**：本地 Ollama / 云端 API / 离线 Mock，自动回退
- 🖥️ **网页界面**：Streamlit，含后端状态徽标、城市选择、示例问题

## 实测效果

```
$ python src/rag.py "武汉哪个公园适合带孩子"

【回答】
根据提供的游客评论，武汉的黄鹤楼和武汉玛雅海滩水公园适合带孩子。黄鹤楼作为
武汉的重要历史建筑，有导游讲解，可以让孩子了解历史知识；玛雅海滩水公园以
水上娱乐为主，亲子游玩体验好……建议根据孩子的兴趣选择。

【参考来源】
  · 武汉 - 黄鹤楼（相似度 0.6524）
  · 武汉 - 武汉玛雅海滩水公园（相似度 0.6280）
  · 武汉 - 海合安武汉极地海洋公园（相似度 0.6163）
  · 武汉 - 解放公园（相似度 0.6041）
  · 武汉 - 武汉东湖磨山景区（相似度 0.5977）
```

> ⚠️ **上面这段是"重跑后"的输出**。此前版本曾出现「武汉的滕王阁」这类回答 ——
> 滕王阁在南昌，这是**源数据城市标注错误 + LLM 盲信检索结果**共同导致的幻觉，
> 同时也是本项目最值得讲的一个坑，详见「局限性」第 5 条。

## 局限性（诚实说明）

1. **评论文本长度上限 500 字**：超长评论被截断，可能丢失后半段信息。
2. **本地 1.5B 模型能力有限**：默认模式的答案质量不如云端大模型，这是"能在集显笔记本上跑"的代价。
3. **离线（Mock）模式不生成**：它只整理检索到的原文，不做归纳润色 —— 仅用于验证检索环节。
4. **检索基于语义相似度**：对"多少钱"这类需要精确数字的问题，可能不如关键词检索准。
5. **⚠️ 源数据存在跨城市污染（已量化，未修复）**：
   爬虫抓的是去哪儿网**城市列表页**，页面底部「猜你喜欢 / 周边推荐」会把
   **邻市热门景点**带进来，并打上当前城市标签。
   `dashboard.py` 的自动检测（规则：景点名含其他城市名）**首次运行即抓出
   16 个可疑景点、330 条评论**，且呈**成对双向传播**：

   | 标注城市 | 混入 | 景点数 | 评论数 |
   |---|---|---|---|
   | **杭州** | **上海** | 5 | **128** |
   | 重庆 | 成都 | 3 | 69 |
   | 济南 | 青岛 + 北京 | 5 | 64 |
   | 上海 | 南京 + 杭州 | 2 | 40 |
   | 成都 | 重庆 | 1 | 29 |

   典型错误：`上海迪士尼度假区` 被标为**杭州**、`重庆欢乐谷` 被标为**成都**、
   `北京野生动物园` 被标为**济南**。

   > 这 16 个只是**下限**（只覆盖"名字带城市名"的）。
   > `九寨沟`（标为成都）、`庐山`（标为武汉）、`张家界`（标为重庆）这类
   > **名字带异地地标但不带城市名**的检测不到，估计还有 ~70 条。

   已产出诊断报告，修复方向倾向**用 `address` / 经纬度字段反查城市**
   （源数据含这两个字段，比关键词匹配可靠一个量级）。**目前数据保持现状未改动。**
6. **未做检索效果定量评估**（仅定性测试）：`eval/` 目录已预留，后续可补召回率 / MRR 指标。
   注意 `dashboard.py` 做的是**语料层统计**，与检索效果评估不是一回事。

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

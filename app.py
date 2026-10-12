"""
景点评论 RAG 问答系统 · Streamlit 界面
=====================================
用法：
    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="景点评论问答 · RAG",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ 样式 ============
st.markdown("""
<style>
    /* 全局 */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

    /* 顶部标题区 */
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 50%, #06b6d4 100%);
        border-radius: 16px;
        padding: 26px 30px;
        color: #ffffff;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(15,118,110,.22);
    }
    .hero h1 { margin: 0 0 6px 0; font-size: 1.65rem; font-weight: 700; color:#fff; }
    .hero p  { margin: 0; opacity: .92; font-size: .92rem; color:#fff; }
    .hero .tags { margin-top: 14px; }
    .hero .tag {
        display: inline-block; background: rgba(255,255,255,.2);
        border: 1px solid rgba(255,255,255,.35);
        border-radius: 999px; padding: 3px 12px;
        font-size: .78rem; margin-right: 8px;
    }

    /* 回答卡片 */
    .answer-card {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-left: 5px solid #14b8a6;
        border-radius: 12px;
        padding: 20px 22px;
        margin: 10px 0 18px 0;
    }
    .answer-card .label {
        font-size: .78rem; font-weight: 700; letter-spacing: .06em;
        color: #0f766e; text-transform: uppercase; margin-bottom: 10px;
    }
    .answer-body { white-space: pre-wrap; line-height: 1.85; color: #134e4a; font-size: .95rem; }

    /* 离线提示 */
    .offline-note {
        background: #fffbeb; border: 1px solid #fde68a;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 14px;
        font-size: .85rem; color: #92400e;
    }

    /* 来源卡片 */
    .src-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 9px;
        transition: box-shadow .15s;
    }
    .src-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,.07); }
    .src-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .src-spot { font-weight: 600; color: #0f172a; font-size: .92rem; }
    .src-city {
        display:inline-block; background:#ccfbf1; color:#0f766e;
        border-radius: 6px; padding: 1px 8px; font-size: .74rem; margin-left: 8px;
    }
    .src-score { font-size: .76rem; color: #64748b; font-variant-numeric: tabular-nums; }
    .src-text { color: #475569; font-size: .87rem; line-height: 1.7; }

    /* 状态徽标 */
    .badge { display:inline-block; border-radius:999px; padding:3px 11px; font-size:.78rem; font-weight:600; }
    .badge-ok   { background:#dcfce7; color:#166534; }
    .badge-warn { background:#fef3c7; color:#92400e; }
    .badge-off  { background:#fee2e2; color:#991b1b; }

    /* 按钮 */
    .stButton > button { border-radius: 9px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ============ RAG 加载 ============
@st.cache_resource(show_spinner=False)
def load_rag():
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from rag import RagPipeline
    return RagPipeline()


# ============ 顶部 ============
st.markdown("""
<div class="hero">
  <h1>🏞️ 景点评论问答</h1>
  <p>基于 <b>6316 条</b>真实游客评论的检索增强问答 —— 答案全部来自真实评论，并标注出处。</p>
  <div class="tags">
    <span class="tag">RAG 检索增强</span>
    <span class="tag">ChromaDB 向量库</span>
    <span class="tag">6,316 条真实语料</span>
    <span class="tag">10 个城市</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============ 初始化 ============
with st.spinner("正在加载检索模型与向量库……"):
    try:
        rag = load_rag()
    except Exception as e:
        st.error(f"❌ 初始化失败：{e}")
        st.info("请先运行 `python src/build_index.py` 建立索引。")
        st.stop()

BACKENDS = {
    "ollama": ("本地 Ollama", "badge-ok"),
    "cloud": ("云端 API", "badge-ok"),
    "mock": ("离线模式（未接大模型）", "badge-warn"),
}
bname, bcls = BACKENDS.get(rag.llm.name, (rag.llm.name, "badge-off"))

# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("### ⚙️ 设置")

    st.markdown("**模型后端**")
    st.markdown(f'<span class="badge {bcls}">{bname}</span>', unsafe_allow_html=True)
    if rag.llm.name == "mock":
        st.caption("检索正常工作；接入 Ollama 或云端 API 后，回答将由大模型归纳生成。")

    st.divider()

    city = st.selectbox(
        "限定城市",
        ["自动识别", "北京", "上海", "武汉", "成都", "西安",
         "重庆", "青岛", "杭州", "济南", "南京"],
        help="选择城市后只在该城市的评论中检索",
    )
    top_k = st.slider("检索评论条数", 3, 10, 5,
                      help="检索出多少条相关评论交给大模型作参考")

    st.divider()
    st.markdown("### 💡 示例问题")
    for e in [
        "武汉哪个公园适合带孩子？",
        "黄鹤楼值不值得去？",
        "有哪些免费又好玩的景点？",
        "青岛有什么吐槽比较多的景点？",
    ]:
        if st.button(e, use_container_width=True, key=f"ex_{e}"):
            st.session_state.pending = e


# ============ 主区：提问 ============
col_q, col_b = st.columns([5, 1])
with col_q:
    question = st.text_input(
        "你的问题",
        value=st.session_state.get("pending", ""),
        placeholder="例如：武汉哪个公园适合带孩子？",
        label_visibility="collapsed",
    )
with col_b:
    ask = st.button("🔍 提问", type="primary", use_container_width=True)

if ask:
    if not question.strip():
        st.warning("请先输入问题")
    else:
        with st.spinner("检索评论中……"):
            answer, hits = rag.ask(
                question.strip(),
                top_k=top_k,
                city=None if city == "自动识别" else city,
            )
        st.session_state.last = (question.strip(), answer, hits)

# ============ 展示结果 ============
if "last" in st.session_state:
    q, answer, hits = st.session_state.last

    if rag.llm.name == "mock":
        st.markdown(
            '<div class="offline-note">⚠️ <b>离线模式</b>：当前未接入大模型，'
            '下面展示的是<b>检索到的评论原文</b>。接入 Ollama / 云端 API 后，'
            '此处会变成大模型写的归纳性回答。</div>',
            unsafe_allow_html=True,
        )

    st.markdown(f"""
    <div class="answer-card">
      <div class="label">回答 · {bname}</div>
      <div class="answer-body">{answer}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"#### 📚 参考来源（{len(hits)} 条）")
    for i, h in enumerate(hits, 1):
        st.markdown(f"""
        <div class="src-card">
          <div class="src-head">
            <div>
              <span class="src-spot">{i}. {h['spot']}</span>
              <span class="src-city">{h['city']}</span>
            </div>
            <div class="src-score">相似度 {h['score']:.3f}</div>
          </div>
          <div class="src-text">{h['text']}</div>
        </div>
        """, unsafe_allow_html=True)

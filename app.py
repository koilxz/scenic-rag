"""
Streamlit 网页界面
==================
用法：
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="景点评论问答", page_icon="🏞️", layout="centered")

st.title("🏞️ 景点评论问答系统")
st.caption("基于 6316 条真实游客评论的 RAG 问答 · 答案均来自真实评论并标注出处")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")

    city = st.selectbox(
        "限定城市",
        ["自动识别", "北京", "上海", "武汉", "成都", "西安", "重庆", "青岛", "杭州", "济南", "南京"],
        index=0,
        help="选择城市后只在该城市的评论里检索；选「自动识别」则从问题里判断",
    )
    top_k = st.slider("检索评论条数", min_value=3, max_value=10, value=5)
    show_source = st.checkbox("显示检索到的原始评论", value=True)

    st.divider()
    st.markdown("**示例问题**")
    examples = [
        "武汉哪个公园适合带孩子？",
        "黄鹤楼值不值得去？",
        "哪个景点吐槽比较多？",
        "有哪些免费又好玩的景点？",
    ]
    for e in examples:
        if st.button(e, use_container_width=True):
            st.session_state.pending = e


@st.cache_resource(show_spinner="正在加载 RAG 系统（首次较慢）……")
def load_rag():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from rag import RagPipeline
    return RagPipeline()


try:
    rag = load_rag()
    st.success(f"✅ 系统就绪 · 模型后端：`{rag.llm.name}`")
except Exception as e:
    st.error(f"❌ 初始化失败：{e}")
    st.info("请先运行 `python src/build_index.py` 建索引，并确保 Ollama 已启动或已配置云端 API。")
    st.stop()

# 输入框
question = st.text_input("输入你的问题", value=st.session_state.get("pending", ""),
                         placeholder="例如：武汉哪个公园适合带孩子？")

col1, col2 = st.columns([1, 5])
with col1:
    ask_btn = st.button("提问", type="primary", use_container_width=True)

if ask_btn and question.strip():
    with st.spinner("检索中……"):
        answer, hits = rag.ask(
            question.strip(),
            top_k=top_k,
            city=None if city == "自动识别" else city,
        )

    st.markdown("### 💬 回答")
    st.markdown(answer)

    if show_source:
        st.markdown("### 📚 参考的原始评论")
        for i, h in enumerate(hits, 1):
            with st.expander(f"{i}. [{h['city']}] {h['spot']} · 相似度 {h['score']}"):
                st.write(h["text"])
elif ask_btn:
    st.warning("请先输入问题")

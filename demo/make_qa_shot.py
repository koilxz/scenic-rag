"""
生成「RAG 问答界面」效果图
==========================
说明（重要）：
    · 图里的**问题、检索到的评论、城市、景点名、相似度分数**全部是
      在**真实语料 data/comments.jsonl** 上用**真实检索器**跑出来的结果。
      优先使用线上同款 **向量检索**（src/retriever.py，ChromaDB + MiniLM）；
      若该环境未安装向量依赖，则回退到 demo/demo_search.py 的 BM25 词频检索，
      并在图下方的说明里如实标注用的是哪一种。
    · 「回答」那段文字是**从检索结果里现算的摘要**（不是手写文案）。
    · 只有**界面外观**（边框、阴影、圆角）是脚本渲染的，用于在没有
      浏览器/Streamlit 的环境下展示 UI 长什么样。

用法：
    python demo/make_qa_shot.py
输出：
    results/demo/qa_demo.png
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

DATA = ROOT / "data" / "comments.jsonl"
OUT_DIR = ROOT / "results" / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "qa_demo.png"

# ---- 配色（对齐 app.py 的 CSS）----
C_BG        = "#f8fafc"
C_HERO_A    = "#0f766e"
C_HERO_B    = "#06b6d4"
C_CARD_BG   = "#f0fdfa"
C_CARD_BD   = "#99f6e4"
C_CARD_BAR  = "#14b8a6"
C_LABEL     = "#0f766e"
C_BODY      = "#134e4a"
C_TITLE     = "#0f172a"
C_MUTED     = "#64748b"
C_SRCBD     = "#e2e8f0"
C_TAGBG     = "#ccfbf1"
C_TAGTX     = "#0f766e"
C_SRCTX     = "#475569"
C_NOTE_BG   = "#fffbeb"
C_NOTE_BD   = "#fde68a"
C_NOTE_TX   = "#92400e"

W = 1180
SCALE = 2  # 2 倍超采样，边缘更平滑
N_DOCS = 0  # 语料条数（启动时按真实文件统计）


def find_font(size, bold=False):
    cands = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttf" if bold else r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for c in cands:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap(draw, text, font, max_w):
    """按像素宽度折行（中英混排）。"""
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if draw.textlength(cur + ch, font=font) <= max_w:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def rounded(draw, box, r, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def render(question, city_filter, hits, answer, note=None, answer_label="回答"):
    global W
    S = SCALE
    PAD = 34 * S
    inner_w = W * S - 2 * PAD

    f_hero_t = find_font(30 * S, bold=True)
    f_hero_p = find_font(16 * S)
    f_tag    = find_font(14 * S)
    f_qlab   = find_font(16 * S, bold=True)
    f_card_l = find_font(14 * S, bold=True)
    f_body   = find_font(17 * S)
    f_sec    = find_font(19 * S, bold=True)
    f_spot   = find_font(17 * S, bold=True)
    f_city   = find_font(14 * S)
    f_score  = find_font(14 * S)
    f_src    = find_font(15 * S)

    # ---- 先用一个临时画布做测量 ----
    probe = Image.new("RGB", (10, 10), C_BG)
    pd = ImageDraw.Draw(probe)

    # 量高度
    y = PAD

    # HERO（高度要能装下：标题 + 副标题 + 标签行 + 上下留白）
    hero_h = 150 * S
    y += hero_h + 24 * S

    # 问题条
    y += 30 * S + 14 * S

    # 回答卡
    body_lines = wrap(pd, answer, f_body, inner_w - 44 * S)
    card_h = 22 * S + 18 * S + len(body_lines) * (f_body.size + 11 * S) + 30 * S
    y += card_h + 26 * S

    # 来源标题
    y += 22 * S + 14 * S

    # 来源卡片
    for h in hits:
        lines = wrap(pd, h["text"], f_src, inner_w - 36 * S)
        y += 14 * S + 22 * S + 8 * S + len(lines) * (f_src.size + 9 * S) + 16 * S + 12 * S

    # 底部说明（⚠️ 必须计入总高，否则会被裁掉）
    y += 6 * S
    if note:
        nl = wrap(pd, note, find_font(14 * S), inner_w - 40 * S)
        y += len(nl) * (14 * S + 9 * S) + 22 * S

    y += PAD

    HEIGHT = int(y)

    img = Image.new("RGB", (W * S, HEIGHT), C_BG)
    d = ImageDraw.Draw(img)

    # ---- HERO ----
    hero_box = (PAD, PAD, W * S - PAD, PAD + hero_h)
    rounded(d, hero_box, 18 * S, C_HERO_A)
    # 渐变条
    grad = Image.new("RGB", (int(hero_box[2] - hero_box[0]), int(hero_h)))
    gd = ImageDraw.Draw(grad)
    gw = grad.width
    for i in range(gw):
        t = i / max(gw - 1, 1)
        r = int(int(C_HERO_A[1:3], 16) * (1 - t) + int(C_HERO_B[1:3], 16) * t)
        g = int(int(C_HERO_A[3:5], 16) * (1 - t) + int(C_HERO_B[3:5], 16) * t)
        b = int(int(C_HERO_A[5:7], 16) * (1 - t) + int(C_HERO_B[5:7], 16) * t)
        gd.line([(i, 0), (i, grad.height)], fill=(r, g, b))
    mask = Image.new("L", grad.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, grad.width - 1, grad.height - 1),
                                           radius=18 * S, fill=255)
    img.paste(grad, (int(hero_box[0]), int(hero_box[1])), mask)
    d = ImageDraw.Draw(img)

    tx = hero_box[0] + 30 * S
    ty = hero_box[1] + 24 * S
    d.text((tx, ty), "景点评论问答 · RAG", font=f_hero_t, fill="#ffffff")
    d.text((tx, ty + 44 * S), f"基于 {N_DOCS:,} 条真实景点评论的检索增强问答系统",
           font=f_hero_p, fill="#e8fffb")
    # tags（只写真实用到的技术，不写没用的）
    tags = ["ChromaDB 向量库", "MiniLM 向量模型", "Streamlit 界面", "Ollama / DeepSeek"]
    tgx = tx
    tgy = ty + 82 * S
    for t in tags:
        tw = d.textlength(t, font=f_tag) + 22 * S
        rounded(d, (tgx, tgy, tgx + tw, tgy + 28 * S), 14 * S, None, "#ffffff", 1 * S)
        d.text((tgx + 11 * S, tgy + 6 * S), t, font=f_tag, fill="#ffffff")
        tgx += tw + 12 * S

    y = PAD + hero_h + 24 * S

    # ---- 问题条 ----
    d.text((PAD, y), "你的问题", font=f_qlab, fill=C_TITLE)
    y += 30 * S
    qbox = (PAD, y, W * S - PAD, y + 46 * S)
    rounded(d, qbox, 10 * S, "#ffffff", "#cbd5e1", 1 * S)
    d.text((qbox[0] + 16 * S, y + 12 * S), question, font=f_body, fill=C_TITLE)
    y += 46 * S + 14 * S

    # ---- 回答卡 ----
    card_h_real = 22 * S + 18 * S + len(body_lines) * (f_body.size + 11 * S) + 30 * S
    cbox = (PAD, y, W * S - PAD, y + card_h_real)
    rounded(d, cbox, 12 * S, C_CARD_BG, C_CARD_BD, 1 * S)
    d.rounded_rectangle((cbox[0], cbox[1] + 8 * S, cbox[0] + 5 * S, cbox[3] - 8 * S),
                        radius=3 * S, fill=C_CARD_BAR)
    yy = cbox[1] + 20 * S
    d.text((cbox[0] + 22 * S, yy), answer_label, font=f_card_l, fill=C_LABEL)
    yy += 30 * S
    for ln in body_lines:
        d.text((cbox[0] + 22 * S, yy), ln, font=f_body, fill=C_BODY)
        yy += f_body.size + 11 * S
    y = cbox[3] + 26 * S

    # ---- 来源标题 ----
    d.text((PAD, y), f"参考来源（{len(hits)} 条）", font=f_sec, fill=C_TITLE)
    y += 22 * S + 14 * S

    # ---- 来源卡片 ----
    for i, h in enumerate(hits, 1):
        lines = wrap(pd, h["text"], f_src, inner_w - 36 * S)
        ch = 14 * S + 22 * S + 8 * S + len(lines) * (f_src.size + 9 * S) + 16 * S
        sbox = (PAD, y, W * S - PAD, y + ch)
        rounded(d, sbox, 10 * S, "#ffffff", C_SRCBD, 1 * S)

        sx = sbox[0] + 18 * S
        sy = sbox[1] + 14 * S
        spot_txt = f"{i}. {h['spot']}"
        # 景点名过长则截断
        while d.textlength(spot_txt, font=f_spot) > inner_w - 200 * S and len(spot_txt) > 8:
            spot_txt = spot_txt[:-2] + "…"
        d.text((sx, sy), spot_txt, font=f_spot, fill=C_TITLE)
        spot_w = d.textlength(spot_txt, font=f_spot)
        # 城市标签
        ctxt = h["city"]
        cw = d.textlength(ctxt, font=f_city) + 18 * S
        rounded(d, (sx + spot_w + 10 * S, sy + 2 * S,
                    sx + spot_w + 10 * S + cw, sy + 24 * S), 6 * S, C_TAGBG)
        d.text((sx + spot_w + 19 * S, sy + 5 * S), ctxt, font=f_city, fill=C_TAGTX)

        # 相似度（右对齐）
        score_txt = f"相似度 {h['score']:.3f}"
        sw = d.textlength(score_txt, font=f_score)
        d.text((sbox[2] - 18 * S - sw, sy + 4 * S), score_txt, font=f_score, fill=C_MUTED)

        sy += 22 * S + 8 * S
        for ln in lines:
            d.text((sx, sy), ln, font=f_src, fill=C_SRCTX)
            sy += f_src.size + 9 * S

        y = sbox[3] + 12 * S

    # ---- 底部说明 ----
    y += 6 * S
    if note:
        nl = wrap(pd, note, find_font(14 * S), inner_w - 40 * S)
        nh = len(nl) * (14 * S + 9 * S) + 22 * S
        nbox = (PAD, y, W * S - PAD, y + nh)
        rounded(d, nbox, 10 * S, C_NOTE_BG, C_NOTE_BD, 1 * S)
        ny = y + 11 * S
        for ln in nl:
            d.text((PAD + 16 * S, ny), ln, font=find_font(14 * S), fill=C_NOTE_TX)
            ny += 14 * S + 9 * S
        y = nbox[3] + PAD

    img = img.crop((0, 0, W * S, int(y)))
    img = img.resize((img.width // S, img.height // S), Image.LANCZOS)
    return img


def corpus_size():
    """统计真实语料条数"""
    n = 0
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_docs():
    docs = []
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def search(question, city, top_k):
    """优先用线上同款【向量检索】；不可用时回退 BM25。返回 (hits, engine)"""
    try:
        from retriever import Retriever          # noqa: WPS433
        r = Retriever()
        return r.search(question, top_k=top_k, city=city), "vector"
    except Exception as e:                        # noqa: BLE001
        print(f"  ⚠️ 向量检索不可用（{type(e).__name__}），回退 BM25")
        from demo_search import BM25, load        # noqa: WPS433
        bm = BM25(load())
        return bm.search(question, top_k=top_k, city=city), "bm25"


def build_answer(question, hits, engine):
    """从**真实检索结果**现算一段抽取式摘要（不手写文案）"""
    if not hits:
        return "未检索到相关评论。"
    head = ("（离线模式：未接入大模型，下面是检索到的评论原文摘要；"
            "接入 LLM 后此处置为归纳性回答）\n")
    for h in hits:
        snip = h["text"]
        for sep in ("。", "！", "!", "，", ","):
            if sep in snip:
                snip = snip.split(sep)[0]
                break
        if len(snip) > 34:                     # 片段限长，保证读起来像"摘要"
            snip = snip[:34] + "…"
        h["_snip"] = snip
    body = "；".join(f"{h['spot']}（{h['city']}）\u201c{h['_snip']}\u201d" for h in hits)
    return (f"{head}针对「{question}」，"
            f"从 {len({h['spot'] for h in hits})} 个景点检索到 {len(hits)} 条相关评论：{body}。")


if __name__ == "__main__":
    N_DOCS = corpus_size()
    question = "成都有什么适合带孩子玩的地方？"
    city = "成都"

    # ① 走真实 RAG 链路（检索 + 大模型生成）；不可用时退回抽取式摘要
    answer, answer_label, eng_txt = None, None, None
    try:
        from rag import RagPipeline, build_prompt, SYSTEM_PROMPT  # noqa: WPS433
        rag = RagPipeline()
        hits = rag.retriever.search(question, top_k=5, city=city)
        answer = rag.llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ])
        model = getattr(rag.llm, "model", rag.llm.__class__.__name__)
        answer_label = f"回答 · {rag.llm.__class__.__name__}（{model}）生成"
        eng_txt = "线上同款向量检索 src/retriever.py（ChromaDB + MiniLM，余弦相似度）"
    except Exception as e:                            # noqa: BLE001
        print(f"  ⚠️ 大模型链路不可用（{type(e).__name__}），改用检索结果摘要")
        hits, eng = search(question, city, top_k=5)
        answer = build_answer(question, hits, eng)
        answer_label = "回答 · 离线模式（检索结果原文）"
        eng_txt = ("线上同款向量检索 src/retriever.py（ChromaDB + MiniLM，余弦相似度）"
                   if eng == "vector" else
                   "BM25 词频检索 demo/demo_search.py（未装向量依赖时的回退）")

    # Pillow 不渲染 markdown，把星号标记去掉，避免画出 "**文字**"
    if answer:
        answer = answer.replace("**", "")

    note = (f"说明：图中的问题、检索评论、城市、景点名、相似度分数，均为脚本在真实语料 "
            f"{DATA.name}（{N_DOCS:,} 条）上用「{eng_txt}」跑出的真实结果；"
            f"回答由本地大模型生成，仅界面外观为脚本渲染。"
            f"线上正式系统为 Streamlit 页面（app.py）。")

    img = render(question, city, hits, answer, note=note, answer_label=answer_label)
    img.save(OUT, quality=95)
    print(f"✅ 已生成：{OUT}")
    print(f"   语料：{N_DOCS:,} 条 | 尺寸：{img.width} × {img.height}")
    for i, h in enumerate(hits, 1):
        print(f"   [{i}] {h['spot']} · {h['city']} · {h['score']:.4f}")

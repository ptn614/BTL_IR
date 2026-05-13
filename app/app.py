"""
Vietnamese News Search — Streamlit App

Run from project root:
    streamlit run app/app.py

Required files:
    models/tfidf_index.pkl
    models/bm25_index.pkl
    models/metadata.pkl
"""

import os
import re
import math
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from underthesea import word_tokenize


# ─── PATHS ───────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFIDF_PATH  = os.path.join(BASE_DIR, "models", "tfidf_index.pkl")
BM25_PATH   = os.path.join(BASE_DIR, "models", "bm25_index.pkl")
META_PATH   = os.path.join(BASE_DIR, "models", "metadata.pkl")


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tìm kiếm tin tức",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─── GLOBAL STYLES ───────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Remove default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 3rem 3rem 4rem;
    max-width: 860px;
    margin: 0 auto;
}

/* ── Typography ── */
h1.page-title {
    font-size: 28px;
    font-weight: 600;
    color: #0f172a;
    margin: 0 0 4px;
    letter-spacing: -0.02em;
}
p.page-sub {
    font-size: 14px;
    color: #64748b;
    margin: 0 0 32px;
}

/* ── Search bar ── */
.search-row {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin-bottom: 12px;
}

/* ── Method select pill group ── */
.pill-group {
    display: flex;
    gap: 6px;
    margin-bottom: 28px;
    flex-wrap: wrap;
}
.pill {
    font-size: 13px;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 999px;
    border: 1.5px solid #e2e8f0;
    background: #fff;
    color: #475569;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
}
.pill:hover  { border-color: #94a3b8; color: #0f172a; }
.pill.active { border-color: #0f172a; background: #0f172a; color: #fff; }

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 8px;
    margin-bottom: 28px;
    font-size: 13px;
    color: #64748b;
}
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 5px 12px;
    font-size: 13px;
    color: #475569;
}
.stat-chip b { color: #0f172a; font-weight: 600; }

/* ── Divider ── */
hr.light {
    border: none;
    border-top: 1px solid #f1f5f9;
    margin: 0 0 20px;
}

/* ── Result card ── */
.r-card {
    padding: 20px 0;
    border-bottom: 1px solid #f1f5f9;
}
.r-card:last-child { border-bottom: none; }

.r-rank {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 6px;
}
.r-title {
    font-size: 16px;
    font-weight: 600;
    color: #0f172a;
    line-height: 1.45;
    margin-bottom: 8px;
}
.r-link {
    color: #0f172a !important;
    text-decoration: none;
    border-bottom: 1.5px solid #e2e8f0;
    transition: color 0.15s, border-color 0.15s;
}
.r-link:hover {
    color: #2563eb !important;
    border-bottom-color: #2563eb;
}

.r-meta {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 8px;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
.r-meta span { display: inline-flex; align-items: center; gap: 3px; }

.r-scores {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.score-pill {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 3px 10px;
    color: #475569;
}
.score-pill b { color: #334155; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 0;
    color: #94a3b8;
    font-size: 14px;
}
.empty-state .icon { font-size: 32px; margin-bottom: 10px; }

/* ── Processed query chip ── */
.query-chip {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 14px;
    color: #475569;
    margin-bottom: 20px;
    word-break: break-all;
}

/* Streamlit widget cleanup */
div[data-testid="stTextInput"] input {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 15px !important;
    height: 44px !important;
    padding: 0 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.15s !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #0f172a !important;
    box-shadow: none !important;
}
.stButton > button {
    border-radius: 10px !important;
    background: #0f172a !important;
    color: #fff !important;
    border: none !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    height: 44px !important;
    padding: 0 22px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: #1e293b !important;
}

div[data-testid="stSelectbox"] > div {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
}

div[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD INDEXES ────────────────────────────────────────────────────────────

@st.cache_resource
def load_indexes():
    paths = [TFIDF_PATH, BM25_PATH, META_PATH]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        return None, None, None, missing

    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    meta = pd.read_pickle(META_PATH)
    return tfidf, bm25, meta, []


tfidf_index, bm25_index, metadata_df, missing_files = load_indexes()

if missing_files:
    st.error("Thiếu file index. Vui lòng chạy các notebook theo thứ tự:")
    st.code("1. notebooks/01_preprocessing.ipynb\n2. notebooks/02_tfidf_manual.ipynb\n3. notebooks/03_bm25_manual.ipynb", language="text")
    for p in missing_files:
        st.code(p)
    st.stop()

idf            = tfidf_index["idf"]
tfidf_docs     = tfidf_index["tfidf_docs"]
bm25_idf       = bm25_index["bm25_idf"]
doc_term_freqs = bm25_index["doc_term_freqs"]
doc_lengths    = bm25_index["doc_lengths"]
avg_doc_length = bm25_index["avg_doc_length"]
N              = bm25_index["N"]
df             = metadata_df.copy()


# ─── PREPROCESSING ────────────────────────────────────────────────────────────

STOPWORDS = {
    "là","và","của","có","cho","với","trong","khi","những","các",
    "một","được","đã","đang","này","đó","thì","mà","ở","về","từ",
    "đến","theo","sau","trước","trên","dưới","vào","ra","năm",
    "ngày","tháng","cũng","như","nếu","vì","do","để","bị","tại",
    "nên","sẽ","rằng","nhiều"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess(text):
    text = clean_text(text)
    text = word_tokenize(text, format="text")
    return " ".join(t for t in text.split() if t not in STOPWORDS)


# ─── SCORING ─────────────────────────────────────────────────────────────────

def query_tfidf_vec(query):
    tokens = preprocess(query).split()
    counts = Counter(tokens)
    n = len(tokens)
    return {t: (c/n)*idf[t] for t, c in counts.items() if t in idf} if n else {}

def cosine(v1, v2):
    common = set(v1) & set(v2)
    num    = sum(v1[t]*v2[t] for t in common)
    d1     = math.sqrt(sum(x**2 for x in v1.values()))
    d2     = math.sqrt(sum(x**2 for x in v2.values()))
    return num / (d1*d2) if d1 and d2 else 0.0

def tfidf_scores(query):
    qv = query_tfidf_vec(query)
    return np.array([cosine(qv, dv) for dv in tfidf_docs])

def bm25_scores(query, k1=1.5, b=0.75):
    tokens = preprocess(query).split()
    scores = []
    for i in range(N):
        s, tf_map, dl = 0.0, doc_term_freqs[i], doc_lengths[i]
        for t in tokens:
            if t not in bm25_idf: continue
            tf  = tf_map.get(t, 0)
            num = tf * (k1 + 1)
            den = tf + k1 * (1 - b + b * dl / avg_doc_length)
            if den: s += bm25_idf[t] * num / den
        scores.append(s)
    return np.array(scores)


# ─── SEARCH ──────────────────────────────────────────────────────────────────

def search(query, method, top_k=10, rrf_k=60):
    ts = tfidf_scores(query)
    bs = bm25_scores(query)

    if method == "TF-IDF":
        idx = ts.argsort()[::-1][:top_k]
        res = df.iloc[idx].copy()
        res["score"] = ts[idx]

    elif method == "BM25":
        idx = bs.argsort()[::-1][:top_k]
        res = df.iloc[idx].copy()
        res["score"] = bs[idx]

    else:  # Hybrid RRF
        tr, br  = ts.argsort()[::-1], bs.argsort()[::-1]
        rrf     = np.zeros(N)
        for r, i in enumerate(tr): rrf[i] += 1/(rrf_k+r+1)
        for r, i in enumerate(br): rrf[i] += 1/(rrf_k+r+1)
        idx = rrf.argsort()[::-1][:top_k]
        res = df.iloc[idx].copy()
        res["score"]       = rrf[idx]
        res["tfidf_score"] = ts[idx]
        res["bm25_score"]  = bs[idx]

    res["rank"] = range(1, len(res)+1)
    return res


# ─── SESSION STATE ────────────────────────────────────────────────────────────

if "method" not in st.session_state:
    st.session_state["method"] = "Hybrid RRF"
if "results" not in st.session_state:
    st.session_state["results"] = None
if "proc_query" not in st.session_state:
    st.session_state["proc_query"] = ""


# ─── HEADER ──────────────────────────────────────────────────────────────────

st.markdown('<h1 class="page-title">Tìm kiếm tin tức tiếng Việt</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">TF-IDF · BM25 · Hybrid RRF</p>', unsafe_allow_html=True)


# ─── SEARCH INPUT ────────────────────────────────────────────────────────────

col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        label="query",
        label_visibility="collapsed",
        placeholder="Nhập từ khoá… vd: giá xăng tăng, bóng đá Việt Nam",
        key="query_input",
    )
with col_btn:
    search_clicked = st.button("Tìm", use_container_width=True)


# ─── METHOD SELECTOR (dropdown) ──────────────────────────────────────────────

method = st.selectbox(
    label="Phương pháp",
    options=["Hybrid RRF", "BM25", "TF-IDF"],
    index=["Hybrid RRF","BM25","TF-IDF"].index(st.session_state["method"]),
    key="method_select",
    label_visibility="visible",
)
st.session_state["method"] = method

# Dataset stats chips
stats_html = f"""
<div class="stats-row">
  <span class="stat-chip">📄 <b>{N:,}</b> bài báo</span>
"""
if "topic" in df.columns:
    stats_html += f'<span class="stat-chip">🗂 <b>{df["topic"].nunique()}</b> chủ đề</span>'
if "source" in df.columns:
    stats_html += f'<span class="stat-chip">📰 <b>{df["source"].nunique()}</b> nguồn</span>'
stats_html += "</div>"
st.markdown(stats_html, unsafe_allow_html=True)


# ─── SEARCH ACTION ────────────────────────────────────────────────────────────

if search_clicked:
    if not query.strip():
        st.warning("Vui lòng nhập từ khoá.")
    else:
        with st.spinner("Đang tìm kiếm…"):
            st.session_state["proc_query"] = preprocess(query)
            st.session_state["results"]    = search(query, method)


# ─── RESULTS ──────────────────────────────────────────────────────────────────

results = st.session_state["results"]

if results is None:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🔍</div>
        Nhập từ khoá và bấm <b>Tìm</b> để bắt đầu.
    </div>
    """, unsafe_allow_html=True)

elif results.empty:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">😶</div>
        Không tìm thấy kết quả phù hợp.
    </div>
    """, unsafe_allow_html=True)

else:
    pq = st.session_state["proc_query"]
    if pq:
        st.markdown(f'<div class="query-chip">🔤 {pq}</div>', unsafe_allow_html=True)

    st.markdown(
        f"**{len(results)}** kết quả &nbsp;·&nbsp; phương pháp: **{method}**",
        unsafe_allow_html=False,
    )
    st.markdown('<hr class="light">', unsafe_allow_html=True)

    for _, row in results.iterrows():
        rank  = int(row.get("rank", 0))
        title = row.get("title", "—")
        url   = row.get("url", "")
        score = float(row.get("score", 0))

        has_url = isinstance(url, str) and url.strip()

        # Score pills
        if method == "Hybrid RRF":
            ts_ = float(row.get("tfidf_score", 0))
            bs_ = float(row.get("bm25_score",  0))
            pills = (
                f'<span class="score-pill"><b>RRF</b>&nbsp;{score:.6f}</span>'
                f'<span class="score-pill"><b>TF-IDF</b>&nbsp;{ts_:.6f}</span>'
                f'<span class="score-pill"><b>BM25</b>&nbsp;{bs_:.4f}</span>'
            )
        else:
            pills = f'<span class="score-pill"><b>{method}</b>&nbsp;{score:.6f}</span>'

        # Title: link if url exists
        if has_url:
            title_part = f'<a class="r-link" href="{url}" target="_blank">{title}</a>'
        else:
            title_part = title

        html = (
            f'<div class="r-card">'
            f'<div class="r-rank">#{rank}</div>'
            f'<div class="r-title">{title_part}</div>'
            f'<div class="r-scores">{pills}</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
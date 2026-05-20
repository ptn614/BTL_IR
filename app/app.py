import sys
import math
import pickle
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PATH CONFIG
# =========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocess import preprocess_query


TFIDF_DIR = PROJECT_ROOT / "models" / "tfidf_rocchio"
LSI_DIR = PROJECT_ROOT / "models" / "lsi_svd"
BM25_DIR = PROJECT_ROOT / "models" / "bm25_query_expansion"


# =========================================================
# 2. LOAD PICKLE HELPER
# =========================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================================================
# 3. LAZY LOAD MODEL BY METHOD
# =========================================================

@st.cache_resource(show_spinner="Đang load TF-IDF / TF-IDF + Rocchio...")
def load_tfidf_model():
    return {
        "vocab": load_pickle(TFIDF_DIR / "vocab.pkl"),
        "idf": load_pickle(TFIDF_DIR / "idf.pkl"),
        "tfidf_matrix": load_pickle(TFIDF_DIR / "tfidf_matrix.pkl"),
        "metadata": load_pickle(TFIDF_DIR / "metadata.pkl"),
    }


@st.cache_resource(show_spinner="Đang load LSI / SVD...")
def load_lsi_model():
    return {
        "vocab": load_pickle(LSI_DIR / "vocab.pkl"),
        "idf": load_pickle(LSI_DIR / "idf.pkl"),
        "svd_model": load_pickle(LSI_DIR / "svd_model.pkl"),
        "lsi_doc_matrix": load_pickle(LSI_DIR / "lsi_doc_matrix.pkl"),
        "metadata": load_pickle(LSI_DIR / "metadata.pkl"),
    }


@st.cache_resource(show_spinner="Đang load BM25 / BM25 + Query Expansion...")
def load_bm25_model():
    return {
        "bm25_idf": load_pickle(BM25_DIR / "bm25_idf.pkl"),
        "inverted_index": load_pickle(BM25_DIR / "inverted_index.pkl"),
        "doc_lengths": load_pickle(BM25_DIR / "doc_lengths.pkl"),
        "avg_doc_length": load_pickle(BM25_DIR / "avg_doc_length.pkl"),
        "metadata": load_pickle(BM25_DIR / "metadata.pkl"),
    }


# =========================================================
# 4. COMMON: QUERY TO TF-IDF VECTOR
# =========================================================

def query_to_tfidf(query_tokens, vocab, idf):
    token_counts = Counter(query_tokens)

    rows = []
    cols = []
    values = []

    for token, count in token_counts.items():
        if token in vocab:
            col_idx = vocab[token]

            tf = 1 + math.log(count)
            tfidf_value = tf * idf[col_idx]

            rows.append(0)
            cols.append(col_idx)
            values.append(tfidf_value)

    query_vector = csr_matrix(
        (values, (rows, cols)),
        shape=(1, len(vocab)),
        dtype=np.float32
    )

    query_vector = normalize(query_vector, norm="l2", axis=1)

    return query_vector


# =========================================================
# 5. SEARCH: TF-IDF
# =========================================================

def search_tfidf(query, tfidf_model, top_k=10):
    query_processed, query_tokens = preprocess_query(query)

    query_tfidf = query_to_tfidf(
        query_tokens=query_tokens,
        vocab=tfidf_model["vocab"],
        idf=tfidf_model["idf"]
    )

    scores = cosine_similarity(
        query_tfidf,
        tfidf_model["tfidf_matrix"]
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = tfidf_model["metadata"].iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    results["query_processed"] = query_processed
    results["method"] = "TF-IDF"

    return results


# =========================================================
# 6. SEARCH: TF-IDF + ROCCHIO
# =========================================================

def search_tfidf_rocchio(
    query,
    tfidf_model,
    top_k=10,
    feedback_k=5,
    alpha=1.0,
    beta=0.75
):
    query_processed, query_tokens = preprocess_query(query)

    query_tfidf = query_to_tfidf(
        query_tokens=query_tokens,
        vocab=tfidf_model["vocab"],
        idf=tfidf_model["idf"]
    )

    first_scores = cosine_similarity(
        query_tfidf,
        tfidf_model["tfidf_matrix"]
    ).flatten()

    pseudo_relevant_indices = first_scores.argsort()[::-1][:feedback_k]

    relevant_docs_vector = tfidf_model["tfidf_matrix"][pseudo_relevant_indices]
    relevant_centroid = relevant_docs_vector.mean(axis=0)

    query_rocchio = alpha * query_tfidf + beta * csr_matrix(relevant_centroid)
    query_rocchio = normalize(query_rocchio, norm="l2", axis=1)

    scores = cosine_similarity(
        query_rocchio,
        tfidf_model["tfidf_matrix"]
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = tfidf_model["metadata"].iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    results["query_processed"] = query_processed
    results["method"] = "TF-IDF + Rocchio"

    return results


# =========================================================
# 7. SEARCH: LSI / SVD
# =========================================================

def search_lsi(query, lsi_model, top_k=10):
    query_processed, query_tokens = preprocess_query(query)

    query_tfidf = query_to_tfidf(
        query_tokens=query_tokens,
        vocab=lsi_model["vocab"],
        idf=lsi_model["idf"]
    )

    query_lsi = lsi_model["svd_model"].transform(query_tfidf)
    query_lsi = normalize(query_lsi, norm="l2")

    scores = cosine_similarity(
        query_lsi,
        lsi_model["lsi_doc_matrix"]
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = lsi_model["metadata"].iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    results["query_processed"] = query_processed
    results["method"] = "LSI / SVD"

    return results


# =========================================================
# 8. SEARCH: BM25
# =========================================================

def iter_posting_items(posting):
    """
    Hỗ trợ 2 dạng inverted index:
    1. token -> {doc_idx: tf}
    2. token -> [(doc_idx, tf), ...]
    """
    if isinstance(posting, dict):
        return posting.items()

    return posting


def bm25_score(query_tokens, bm25_model, k1=1.5, b=0.75):
    scores = defaultdict(float)

    bm25_idf = bm25_model["bm25_idf"]
    inverted_index = bm25_model["inverted_index"]
    doc_lengths = bm25_model["doc_lengths"]
    avg_doc_length = bm25_model["avg_doc_length"]

    for token in query_tokens:
        if token not in inverted_index:
            continue

        idf = bm25_idf.get(token, 0.0)
        posting = inverted_index[token]

        for doc_idx, tf in iter_posting_items(posting):
            dl = doc_lengths[doc_idx]

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avg_doc_length)

            scores[doc_idx] += idf * numerator / denominator

    return scores


def search_bm25(query, bm25_model, top_k=10):
    query_processed, query_tokens = preprocess_query(query)

    scores = bm25_score(query_tokens, bm25_model)

    if len(scores) == 0:
        return pd.DataFrame()

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    top_indices = [doc_idx for doc_idx, _ in ranked]
    top_scores = [score for _, score in ranked]

    results = bm25_model["metadata"].iloc[top_indices].copy()
    results["score"] = top_scores
    results["query_processed"] = query_processed
    results["method"] = "BM25"

    return results


# =========================================================
# 9. SEARCH: BM25 + QUERY EXPANSION
# =========================================================

def get_expansion_terms(query_tokens, top_doc_indices, bm25_model, expand_n=5):
    query_token_set = set(query_tokens)

    inverted_index = bm25_model["inverted_index"]
    bm25_idf = bm25_model["bm25_idf"]

    candidate_scores = defaultdict(float)
    top_doc_set = set(top_doc_indices)

    for token, posting in inverted_index.items():
        if token in query_token_set:
            continue

        score = 0.0

        for doc_idx, tf in iter_posting_items(posting):
            if doc_idx in top_doc_set:
                score += tf * bm25_idf.get(token, 0.0)

        if score > 0:
            candidate_scores[token] = score

    sorted_terms = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    expansion_terms = [token for token, _ in sorted_terms[:expand_n]]

    return expansion_terms


def search_bm25_query_expansion(
    query,
    bm25_model,
    top_k=10,
    feedback_k=5,
    expand_n=5
):
    query_processed, query_tokens = preprocess_query(query)

    first_scores = bm25_score(query_tokens, bm25_model)

    if len(first_scores) == 0:
        return pd.DataFrame()

    first_ranked = sorted(
        first_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:feedback_k]

    top_doc_indices = [doc_idx for doc_idx, _ in first_ranked]

    expansion_terms = get_expansion_terms(
        query_tokens=query_tokens,
        top_doc_indices=top_doc_indices,
        bm25_model=bm25_model,
        expand_n=expand_n
    )

    expanded_query_tokens = query_tokens + expansion_terms
    expanded_query = " ".join(expanded_query_tokens)

    second_scores = bm25_score(expanded_query_tokens, bm25_model)

    second_ranked = sorted(
        second_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    top_indices = [doc_idx for doc_idx, _ in second_ranked]
    top_scores = [score for _, score in second_ranked]

    results = bm25_model["metadata"].iloc[top_indices].copy()
    results["score"] = top_scores
    results["query_processed"] = query_processed
    results["expanded_query"] = expanded_query
    results["expansion_terms"] = ", ".join(expansion_terms)
    results["method"] = "BM25 + Query Expansion"

    return results


# =========================================================
# 10. UI HELPER
# =========================================================

def safe_get(row, col, default=""):
    if col in row and pd.notna(row[col]):
        return row[col]

    return default


def render_results(results):
    if results is None or len(results) == 0:
        st.warning("Không tìm thấy kết quả phù hợp.")
        return

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        title = str(safe_get(row, "title", "Không có tiêu đề"))
        url = str(safe_get(row, "url", ""))
        score = safe_get(row, "score", 0)

        if url and url != "nan":
            title_html = f'<a href="{url}" target="_blank">{title}</a>'
        else:
            title_html = title

        st.markdown(
            f"""
            <div class="result-card">
                <div class="rank">#{rank}</div>
                <div class="title">{title_html}</div>
                <div class="meta"><b>Score:</b> {float(score):.4f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 11. STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="Vietnamese News Retrieval",
    page_icon="🔎",
    layout="wide"
)

st.markdown(
    """
    <style>
        .main {
            background-color: #f8fafc;
        }

        .app-title {
            font-size: 36px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 4px;
        }

        .app-subtitle {
            font-size: 16px;
            color: #475569;
            margin-bottom: 24px;
        }

        .result-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }

        .rank {
            display: inline-block;
            background: #e0f2fe;
            color: #0369a1;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 999px;
            margin-bottom: 8px;
        }

        .title {
            font-size: 19px;
            font-weight: 750;
            color: #0f172a;
            margin-top: 6px;
            margin-bottom: 8px;
        }

        .title a {
            color: #0f172a;
            text-decoration: none;
        }

        .title a:hover {
            color: #0284c7;
            text-decoration: underline;
        }

        .meta {
            color: #64748b;
            font-size: 14px;
            margin-top: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-title">🔎 Vietnamese News Retrieval System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">Truy xuất tin tức tiếng Việt bằng TF-IDF, Rocchio, LSI/SVD và BM25</div>',
    unsafe_allow_html=True
)


# =========================================================
# 12. SIDEBAR CONFIG
# =========================================================

with st.sidebar:
    st.header("⚙️ Cấu hình tìm kiếm")

    method = st.selectbox(
        "Chọn phương pháp",
        [
            "TF-IDF",
            "TF-IDF + Rocchio",
            "LSI / SVD",
            "BM25",
            "BM25 + Query Expansion"
        ]
    )

    top_k = st.slider("Số kết quả Top-K", 5, 30, 10)

    if method in ["TF-IDF + Rocchio", "BM25 + Query Expansion"]:
        st.divider()
        feedback_k = st.slider("Feedback K / Top documents", 3, 20, 5)
    else:
        feedback_k = 5

    if method == "BM25 + Query Expansion":
        expand_n = st.slider("Số từ mở rộng BM25", 1, 10, 5)
    else:
        expand_n = 5

    st.caption("App chỉ load model của phương pháp được chọn khi bấm tìm kiếm.")


query = st.text_input(
    "Nhập câu truy vấn",
    value="cầu thủ Quang Hải",
    placeholder="Ví dụ: giá xăng tăng mạnh, chứng khoán giảm điểm, bóng đá Việt Nam..."
)

search_clicked = st.button("Tìm kiếm", type="primary")


# =========================================================
# 13. SEARCH ACTION
# =========================================================

if search_clicked:
    if query.strip() == "":
        st.warning("Vui lòng nhập câu truy vấn.")
    else:
        query_processed, query_tokens = preprocess_query(query)

        st.markdown("### Query")

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Query gốc:** {query}")

        with col2:
            st.success(f"**Query sau xử lý:** {query_processed}")

        st.divider()

        if method == "TF-IDF":
            tfidf_model = load_tfidf_model()

            results = search_tfidf(
                query=query,
                tfidf_model=tfidf_model,
                top_k=top_k
            )

            st.subheader("Kết quả: TF-IDF")
            render_results(results)

        elif method == "TF-IDF + Rocchio":
            tfidf_model = load_tfidf_model()

            results = search_tfidf_rocchio(
                query=query,
                tfidf_model=tfidf_model,
                top_k=top_k,
                feedback_k=feedback_k
            )

            st.subheader("Kết quả: TF-IDF + Rocchio")
            render_results(results)

        elif method == "LSI / SVD":
            lsi_model = load_lsi_model()

            results = search_lsi(
                query=query,
                lsi_model=lsi_model,
                top_k=top_k
            )

            st.subheader("Kết quả: LSI / SVD")
            render_results(results)

        elif method == "BM25":
            bm25_model = load_bm25_model()

            results = search_bm25(
                query=query,
                bm25_model=bm25_model,
                top_k=top_k
            )

            st.subheader("Kết quả: BM25")
            render_results(results)

        elif method == "BM25 + Query Expansion":
            bm25_model = load_bm25_model()

            results = search_bm25_query_expansion(
                query=query,
                bm25_model=bm25_model,
                top_k=top_k,
                feedback_k=feedback_k,
                expand_n=expand_n
            )

            st.subheader("Kết quả: BM25 + Query Expansion")

            if len(results) > 0 and "expanded_query" in results.columns:
                st.warning(f"**Query mở rộng:** {results['expanded_query'].iloc[0]}")

            render_results(results)
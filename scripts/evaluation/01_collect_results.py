import sys
import math
import gc
import pickle
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. PATH CONFIG
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocess import preprocess_query


QUERY_PATH = PROJECT_ROOT / "data" / "evaluation" / "queries.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "search_results.csv"

TFIDF_DIR = PROJECT_ROOT / "models" / "tfidf_rocchio"
LSI_DIR = PROJECT_ROOT / "models" / "lsi_svd"
BM25_DIR = PROJECT_ROOT / "models" / "bm25_query_expansion"


# =====================================================
# 2. COMMON HELPER
# =====================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_get(row, col, default=""):
    if col in row and pd.notna(row[col]):
        return row[col]
    return default


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


# =====================================================
# 3. TF-IDF / TF-IDF + ROCCHIO
# =====================================================

def load_tfidf_model():
    return {
        "vocab": load_pickle(TFIDF_DIR / "vocab.pkl"),
        "idf": load_pickle(TFIDF_DIR / "idf.pkl"),
        "tfidf_matrix": load_pickle(TFIDF_DIR / "tfidf_matrix.pkl"),
        "metadata": load_pickle(TFIDF_DIR / "metadata.pkl"),
    }


def search_tfidf(query, model, top_k=10):
    query_processed, query_tokens = preprocess_query(query)

    query_tfidf = query_to_tfidf(
        query_tokens=query_tokens,
        vocab=model["vocab"],
        idf=model["idf"]
    )

    scores = cosine_similarity(
        query_tfidf,
        model["tfidf_matrix"]
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = model["metadata"].iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    results["query_processed"] = query_processed

    return results


def search_tfidf_rocchio(
    query,
    model,
    top_k=10,
    feedback_k=5,
    alpha=1.0,
    beta=0.75
):
    query_processed, query_tokens = preprocess_query(query)

    query_tfidf = query_to_tfidf(
        query_tokens=query_tokens,
        vocab=model["vocab"],
        idf=model["idf"]
    )

    first_scores = cosine_similarity(
        query_tfidf,
        model["tfidf_matrix"]
    ).flatten()

    pseudo_relevant_indices = first_scores.argsort()[::-1][:feedback_k]

    relevant_docs_vector = model["tfidf_matrix"][pseudo_relevant_indices]
    relevant_centroid = relevant_docs_vector.mean(axis=0)

    query_rocchio = alpha * query_tfidf + beta * csr_matrix(relevant_centroid)
    query_rocchio = normalize(query_rocchio, norm="l2", axis=1)

    scores = cosine_similarity(
        query_rocchio,
        model["tfidf_matrix"]
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = model["metadata"].iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    results["query_processed"] = query_processed

    return results


# =====================================================
# 4. LSI / SVD
# =====================================================

def load_lsi_model():
    return {
        "vocab": load_pickle(LSI_DIR / "vocab.pkl"),
        "idf": load_pickle(LSI_DIR / "idf.pkl"),
        "svd_model": load_pickle(LSI_DIR / "svd_model.pkl"),
        "lsi_doc_matrix": load_pickle(LSI_DIR / "lsi_doc_matrix.pkl"),
        "metadata": load_pickle(LSI_DIR / "metadata.pkl"),
    }


def search_lsi(query, model, top_k=10):
    query_processed, query_tokens = preprocess_query(query)

    query_tfidf = query_to_tfidf(
        query_tokens=query_tokens,
        vocab=model["vocab"],
        idf=model["idf"]
    )

    query_lsi = model["svd_model"].transform(query_tfidf)
    query_lsi = normalize(query_lsi, norm="l2")

    scores = cosine_similarity(
        query_lsi,
        model["lsi_doc_matrix"]
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = model["metadata"].iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    results["query_processed"] = query_processed

    return results


# =====================================================
# 5. BM25 / BM25 + QUERY EXPANSION
# =====================================================

def load_bm25_model():
    return {
        "bm25_idf": load_pickle(BM25_DIR / "bm25_idf.pkl"),
        "inverted_index": load_pickle(BM25_DIR / "inverted_index.pkl"),
        "doc_lengths": load_pickle(BM25_DIR / "doc_lengths.pkl"),
        "avg_doc_length": load_pickle(BM25_DIR / "avg_doc_length.pkl"),
        "metadata": load_pickle(BM25_DIR / "metadata.pkl"),
    }


def iter_posting_items(posting):
    if isinstance(posting, dict):
        return posting.items()
    return posting


def bm25_score(query_tokens, model, k1=1.5, b=0.75):
    scores = defaultdict(float)

    bm25_idf = model["bm25_idf"]
    inverted_index = model["inverted_index"]
    doc_lengths = model["doc_lengths"]
    avg_doc_length = model["avg_doc_length"]

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


def search_bm25(query, model, top_k=10):
    query_processed, query_tokens = preprocess_query(query)

    scores = bm25_score(query_tokens, model)

    if len(scores) == 0:
        return pd.DataFrame()

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    top_indices = [doc_idx for doc_idx, _ in ranked]
    top_scores = [score for _, score in ranked]

    results = model["metadata"].iloc[top_indices].copy()
    results["score"] = top_scores
    results["query_processed"] = query_processed

    return results


def get_expansion_terms(query_tokens, top_doc_indices, model, expand_n=5):
    query_token_set = set(query_tokens)

    inverted_index = model["inverted_index"]
    bm25_idf = model["bm25_idf"]

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

    return [token for token, _ in sorted_terms[:expand_n]]


def search_bm25_query_expansion(
    query,
    model,
    top_k=10,
    feedback_k=5,
    expand_n=5
):
    query_processed, query_tokens = preprocess_query(query)

    first_scores = bm25_score(query_tokens, model)

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
        model=model,
        expand_n=expand_n
    )

    expanded_query_tokens = query_tokens + expansion_terms
    expanded_query = " ".join(expanded_query_tokens)

    second_scores = bm25_score(expanded_query_tokens, model)

    second_ranked = sorted(
        second_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    top_indices = [doc_idx for doc_idx, _ in second_ranked]
    top_scores = [score for _, score in second_ranked]

    results = model["metadata"].iloc[top_indices].copy()
    results["score"] = top_scores
    results["query_processed"] = query_processed
    results["expanded_query"] = expanded_query
    results["expansion_terms"] = ", ".join(expansion_terms)

    return results


# =====================================================
# 6. APPEND RESULT HELPER
# =====================================================

def append_results(all_rows, query_id, query, method, results):
    if results is None or len(results) == 0:
        return

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        all_rows.append({
            "query_id": query_id,
            "query": query,
            "method": method,
            "rank": rank,
            "doc_id": safe_get(row, "doc_id", ""),
            "title": safe_get(row, "title", ""),
            "topic": safe_get(row, "topic", ""),
            "source": safe_get(row, "source", ""),
            "url": safe_get(row, "url", ""),
            "score": safe_get(row, "score", 0),
            "query_processed": safe_get(row, "query_processed", ""),
            "expanded_query": safe_get(row, "expanded_query", ""),
            "expansion_terms": safe_get(row, "expansion_terms", "")
        })


# =====================================================
# 7. MAIN
# =====================================================

def main():
    if not QUERY_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file queries.csv tại: {QUERY_PATH}")

    queries = pd.read_csv(QUERY_PATH, encoding="utf-8-sig")

    if "query_id" not in queries.columns or "query" not in queries.columns:
        raise ValueError("queries.csv phải có 2 cột: query_id, query")

    all_rows = []

    # -------------------------------------------------
    # 1. TF-IDF + TF-IDF ROCCHIO
    # -------------------------------------------------
    print("Loading TF-IDF model...")
    tfidf_model = load_tfidf_model()

    for _, row in queries.iterrows():
        query_id = row["query_id"]
        query = row["query"]

        print(f"[TF-IDF] {query_id}: {query}")

        results = search_tfidf(
            query=query,
            model=tfidf_model,
            top_k=10
        )

        append_results(
            all_rows=all_rows,
            query_id=query_id,
            query=query,
            method="TF-IDF",
            results=results
        )

    for _, row in queries.iterrows():
        query_id = row["query_id"]
        query = row["query"]

        print(f"[TF-IDF + Rocchio] {query_id}: {query}")

        results = search_tfidf_rocchio(
            query=query,
            model=tfidf_model,
            top_k=10,
            feedback_k=5
        )

        append_results(
            all_rows=all_rows,
            query_id=query_id,
            query=query,
            method="TF-IDF + Rocchio Feedback",
            results=results
        )

    del tfidf_model
    gc.collect()

    # -------------------------------------------------
    # 2. LSI/SVD
    # -------------------------------------------------
    print("Loading LSI/SVD model...")
    lsi_model = load_lsi_model()

    for _, row in queries.iterrows():
        query_id = row["query_id"]
        query = row["query"]

        print(f"[LSI/SVD] {query_id}: {query}")

        results = search_lsi(
            query=query,
            model=lsi_model,
            top_k=10
        )

        append_results(
            all_rows=all_rows,
            query_id=query_id,
            query=query,
            method="LSI/SVD",
            results=results
        )

    del lsi_model
    gc.collect()

    # -------------------------------------------------
    # 3. BM25 + BM25 QUERY EXPANSION
    # -------------------------------------------------
    print("Loading BM25 model...")
    bm25_model = load_bm25_model()

    for _, row in queries.iterrows():
        query_id = row["query_id"]
        query = row["query"]

        print(f"[BM25] {query_id}: {query}")

        results = search_bm25(
            query=query,
            model=bm25_model,
            top_k=10
        )

        append_results(
            all_rows=all_rows,
            query_id=query_id,
            query=query,
            method="BM25",
            results=results
        )

    for _, row in queries.iterrows():
        query_id = row["query_id"]
        query = row["query"]

        print(f"[BM25 + QE] {query_id}: {query}")

        results = search_bm25_query_expansion(
            query=query,
            model=bm25_model,
            top_k=10,
            feedback_k=5,
            expand_n=5
        )

        append_results(
            all_rows=all_rows,
            query_id=query_id,
            query=query,
            method="BM25 + Query Expansion",
            results=results
        )

    del bm25_model
    gc.collect()

    # -------------------------------------------------
    # 4. SAVE
    # -------------------------------------------------
    output_df = pd.DataFrame(all_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_PATH)
    print("Shape:", output_df.shape)
    print("Methods:", output_df["method"].unique().tolist())


if __name__ == "__main__":
    main()
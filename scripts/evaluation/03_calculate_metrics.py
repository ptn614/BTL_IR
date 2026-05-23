from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEARCH_RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation" / "search_results.csv"
QRELS_PATH = PROJECT_ROOT / "data" / "evaluation" / "qrels.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "evaluation_metrics.csv"


K = 10


def precision_at_k(results_for_query, relevant_doc_ids, k=10):
    top_k = results_for_query.sort_values("rank").head(k)

    retrieved_doc_ids = set(top_k["doc_id"].astype(str))
    relevant_retrieved = retrieved_doc_ids.intersection(relevant_doc_ids)

    return len(relevant_retrieved) / k


def recall_at_k(results_for_query, relevant_doc_ids, k=10):
    if len(relevant_doc_ids) == 0:
        return 0.0

    top_k = results_for_query.sort_values("rank").head(k)

    retrieved_doc_ids = set(top_k["doc_id"].astype(str))
    relevant_retrieved = retrieved_doc_ids.intersection(relevant_doc_ids)

    return len(relevant_retrieved) / len(relevant_doc_ids)


def reciprocal_rank(results_for_query, relevant_doc_ids, k=10):
    top_k = results_for_query.sort_values("rank").head(k)

    for _, row in top_k.iterrows():
        doc_id = str(row["doc_id"])
        rank = int(row["rank"])

        if doc_id in relevant_doc_ids:
            return 1 / rank

    return 0.0


def main():
    results = pd.read_csv(SEARCH_RESULTS_PATH)
    qrels = pd.read_csv(QRELS_PATH)

    results["doc_id"] = results["doc_id"].astype(str)
    qrels["doc_id"] = qrels["doc_id"].astype(str)

    qrels["relevance"] = qrels["relevance"].fillna(0).astype(int)

    methods = results["method"].unique()
    query_ids = results["query_id"].unique()

    rows = []

    for method in methods:
        method_results = results[results["method"] == method]

        p_list = []
        r_list = []
        rr_list = []

        for query_id in query_ids:
            query_method_results = method_results[
                method_results["query_id"] == query_id
            ]

            relevant_doc_ids = set(
                qrels[
                    (qrels["query_id"] == query_id) &
                    (qrels["relevance"] == 1)
                ]["doc_id"].astype(str)
            )

            p = precision_at_k(
                results_for_query=query_method_results,
                relevant_doc_ids=relevant_doc_ids,
                k=K
            )

            r = recall_at_k(
                results_for_query=query_method_results,
                relevant_doc_ids=relevant_doc_ids,
                k=K
            )

            rr = reciprocal_rank(
                results_for_query=query_method_results,
                relevant_doc_ids=relevant_doc_ids,
                k=K
            )

            p_list.append(p)
            r_list.append(r)
            rr_list.append(rr)

        rows.append({
            "method": method,
            f"Precision@{K}": sum(p_list) / len(p_list),
            f"Recall@{K}": sum(r_list) / len(r_list),
            "MRR": sum(rr_list) / len(rr_list)
        })

    metrics = pd.DataFrame(rows)

    metrics.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Saved:", OUTPUT_PATH)
    print(metrics)


if __name__ == "__main__":
    main()
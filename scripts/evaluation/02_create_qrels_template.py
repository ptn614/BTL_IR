from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEARCH_RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation" / "search_results.csv"
QRELS_TEMPLATE_PATH = PROJECT_ROOT / "data" / "evaluation" / "qrels_template.csv"


def main():
    results = pd.read_csv(SEARCH_RESULTS_PATH)

    # Gộp trùng theo query_id + doc_id
    # Vì cùng một bài có thể xuất hiện ở nhiều phương pháp
    qrels = results[[
        "query_id",
        "query",
        "doc_id",
        "title",
        "topic",
        "source",
        "url"
    ]].drop_duplicates(subset=["query_id", "doc_id"])

    # Cột này bạn tự điền:
    # 1 = liên quan
    # 0 = không liên quan
    qrels["relevance"] = ""

    qrels = qrels.sort_values(["query_id", "doc_id"])

    qrels.to_csv(QRELS_TEMPLATE_PATH, index=False, encoding="utf-8-sig")

    print("Saved:", QRELS_TEMPLATE_PATH)
    print("Shape:", qrels.shape)
    print("Bạn hãy mở file này và điền cột relevance = 1 hoặc 0.")


if __name__ == "__main__":
    main()
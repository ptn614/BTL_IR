BTL/
│
├── data/
│   ├── raw/
│   │   └── vietnamese_online_news.json
│   │
│   └── processed/
│       └── news_processed.pkl
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_tfidf_manual.ipynb
│   ├── 03_bm25_manual.ipynb
│   ├── 04_hybrid_rrf.ipynb
│   └── 05_evaluation.ipynb
│
├── app/
│   └── app.py
│
├── models/
│   ├── tfidf_index.pkl
│   ├── bm25_index.pkl
│   └── metadata.pkl
│
├── outputs/
│   ├── relevance_judgments_template.csv
│   ├── relevance_judgments_filled.csv
│   ├── evaluation_detail.csv
│   └── evaluation_summary.csv
│
├── requirements.txt
├── README.md
└── .gitignore


## Tạo môi trường ảo
python -m venv .venv

## Kích hoạt môi trường 
.venv\Scripts\activate

## Cài thư viện
pip install -r requirements.txt

## Chạy các notebook(tải dataset trước khi chạy)
01_preprocessing.ipynb -- Tiền xử lý dữ liệu,chạy xong notebook này dc file dữ liệu mới news_processed.pkl(có thể tải về k cần chạy,chạy mất 1h)
02_tfidf_manual.ipynb
(cách tf-idf ,chạy xong sẽ lưu file index vòa folder model)
03_bm25_manual.ipynb
(tương tự tf_idf)
04_hybrid_rrf.ipynb
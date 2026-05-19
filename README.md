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
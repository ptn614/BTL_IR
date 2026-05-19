import re
import unicodedata
from underthesea import word_tokenize


vietnamese_stopwords = set([
    "là", "và", "của", "có", "cho", "với", "trong", "khi",
    "những", "các", "một", "được", "đã", "đang", "này",
    "đó", "thì", "mà", "ở", "về", "từ", "đến", "theo",
    "sau", "trước", "trên", "dưới", "vào", "ra",
    "cũng", "như", "nếu", "vì", "do", "để", "bị",
    "tại", "nên", "sẽ", "rằng", "nhiều", "ít",
    "hơn", "rất", "lại", "phải", "không"
])


def clean_text(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text):
    tokens = text.split()
    tokens = [token for token in tokens if token not in vietnamese_stopwords]
    return " ".join(tokens)


def preprocess_text(text):
    text = clean_text(text)
    text = word_tokenize(text, format="text")
    text = remove_stopwords(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_query(query):
    query_processed = preprocess_text(query)
    query_tokens = query_processed.split()
    return query_processed, query_tokens
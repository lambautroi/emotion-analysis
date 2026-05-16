import os
import re
import joblib
import pandas as pd
from datasets import load_dataset
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = word_tokenize(text, format="text")
    return text

if __name__ == "__main__":
    print("=" * 60)
    print("BƯỚC 1 & 2: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU UIT-VSFC")
    print("=" * 60)
    
    dataset = load_dataset("uitnlp/vietnamese_students_feedback", trust_remote_code=True)
    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])
    
    train_full = pd.concat([train_df, val_df], ignore_index=True)
    
    print("  Đang tiền xử lý train set...")
    train_full["processed"] = train_full["sentence"].apply(preprocess_text)
    print("  Đang tiền xử lý test set...")
    test_df["processed"] = test_df["sentence"].apply(preprocess_text)
    
    print("=" * 60)
    print("BƯỚC 3: TF-IDF VECTORIZATION")
    print("=" * 60)
    
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_full["processed"])
    X_test = tfidf.transform(test_df["processed"])
    y_train = train_full["sentiment"].values
    y_test = test_df["sentiment"].values
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    
    joblib.dump({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_df": test_df
    }, "models/data_prepared.pkl")
    
    print("  HOÀN THÀNH! Dữ liệu đã được lưu vào models/data_prepared.pkl")

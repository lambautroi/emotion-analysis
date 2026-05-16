"""
Script Training ML Models cho Sentiment Analysis
Dataset: UIT-VSFC (Vietnamese Students' Feedback Corpus)
Models: Naive Bayes, SVM, Logistic Regression, Random Forest
+ So sánh với Gemini API
"""

import os
import re
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

# ============================================================
# BƯỚC 1: TẢI DATASET
# ============================================================
def load_data():
    print("=" * 60)
    print("BƯỚC 1: TẢI DATASET UIT-VSFC")
    print("=" * 60)
    
    dataset = load_dataset("uitnlp/vietnamese_students_feedback", trust_remote_code=True)
    
    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])
    
    print(f"  Train set : {len(train_df)} samples")
    print(f"  Val set   : {len(val_df)} samples")
    print(f"  Test set  : {len(test_df)} samples")
    print(f"  Total     : {len(train_df) + len(val_df) + len(test_df)} samples")
    print(f"\n  Labels: 0=Negative, 1=Neutral, 2=Positive")
    print(f"  Train distribution:")
    for label, count in train_df["sentiment"].value_counts().sort_index().items():
        names = {0: "Negative", 1: "Neutral", 2: "Positive"}
        print(f"    {names[label]}: {count} ({count/len(train_df)*100:.1f}%)")
    
    # Gộp train + val để train
    train_full = pd.concat([train_df, val_df], ignore_index=True)
    return train_full, test_df

# ============================================================
# BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN
# ============================================================
def preprocess_text(text):
    """Tiền xử lý: lowercase, xóa ký tự đặc biệt, tách từ"""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = word_tokenize(text, format="text")
    return text

def preprocess_data(train_df, test_df):
    print("\n" + "=" * 60)
    print("BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN")
    print("=" * 60)
    
    print("  Đang xử lý train set...")
    train_df["processed"] = train_df["sentence"].apply(preprocess_text)
    print("  Đang xử lý test set...")
    test_df["processed"] = test_df["sentence"].apply(preprocess_text)
    
    print(f"  Hoàn thành!")
    print(f"\n  Ví dụ tiền xử lý:")
    for i in range(min(3, len(train_df))):
        print(f"    Gốc  : {train_df.iloc[i]['sentence'][:80]}")
        print(f"    Sau  : {train_df.iloc[i]['processed'][:80]}")
        print()
    
    return train_df, test_df

# ============================================================
# BƯỚC 3: TF-IDF VECTORIZATION
# ============================================================
def vectorize(train_df, test_df):
    print("=" * 60)
    print("BƯỚC 3: TF-IDF VECTORIZATION")
    print("=" * 60)
    
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_df["processed"])
    X_test = tfidf.transform(test_df["processed"])
    y_train = train_df["sentiment"].values
    y_test = test_df["sentiment"].values
    
    print(f"  Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_test shape  : {X_test.shape}")
    
    # Lưu vectorizer
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    print("  Saved: models/tfidf_vectorizer.pkl")
    
    return X_train, X_test, y_train, y_test, tfidf

# ============================================================
# BƯỚC 4: TRAIN & ĐÁNH GIÁ 4 MODELS
# ============================================================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("BƯỚC 4: TRAIN & ĐÁNH GIÁ CÁC MÔ HÌNH")
    print("=" * 60)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="linear", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    
    results = {}
    label_names = ["Negative", "Neutral", "Positive"]
    best_acc = 0
    best_model_name = ""
    best_model = None
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        print(f"  Train time: {train_time:.2f}s")
        
        # Predict
        start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print(f"  Pred time: {pred_time:.4f}s ({len(y_test)} samples)")
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=label_names)
        print(f"\n  Classification Report:\n{report}")
        
        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "train_time": train_time,
            "pred_time": pred_time,
            "y_pred": y_pred,
            "report": report,
        }
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model
    
    # Lưu model tốt nhất
    joblib.dump(best_model, "models/best_model.pkl")
    with open("models/model_info.json", "w", encoding="utf-8") as f:
        json.dump({"name": best_model_name, "accuracy": best_acc}, f, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"  MODEL TỐT NHẤT: {best_model_name} (Accuracy: {best_acc*100:.2f}%)")
    print(f"  Saved: models/best_model.pkl")
    print(f"{'='*60}")
    
    return results, best_model_name

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("#  SENTIMENT ANALYSIS - TRAINING PIPELINE")
    print("#  Dataset: UIT-VSFC | Models: NB, SVM, LR, RF")
    print("#" * 60 + "\n")
    
    # 1. Load data
    train_df, test_df = load_data()
    
    # 2. Preprocess
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # 3. Vectorize
    X_train, X_test, y_train, y_test, tfidf = vectorize(train_df, test_df)
    
    # 4. Train & evaluate
    results, best_model_name = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Lưu data để evaluate sau
    eval_data = {
        "results": results,
        "test_df": test_df,
        "y_test": y_test,
        "best_model_name": best_model_name
    }
    joblib.dump(eval_data, "models/eval_data.pkl")
    print(f"\n  >>> Đã lưu kết quả train vào models/eval_data.pkl <<<")
    print(f"  >>> HÃY CHẠY FILE `python evaluate_models.py` ĐỂ XEM BÁO CÁO VÀ BIỂU ĐỒ! <<<\n")

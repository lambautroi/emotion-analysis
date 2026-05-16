import os

prep_code = """import os
import re
import joblib
import pandas as pd
from datasets import load_dataset
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\\w\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()
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
"""

template = """import time
import joblib
import os
import json
{import_statement}
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

if __name__ == "__main__":
    model_name = "{model_name}"
    print(f"\\n--- Huấn luyện {{model_name}} ---")
    
    if not os.path.exists("models/data_prepared.pkl"):
        print("Lỗi: Không tìm thấy models/data_prepared.pkl. Vui lòng chạy 1_prepare_data.py trước!")
        exit(1)
        
    data = joblib.load("models/data_prepared.pkl")
    X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]
    
    model = {model_class}
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print(f"  Train time: {{train_time:.2f}}s")
    print(f"  Accuracy : {{acc:.4f}} ({{acc*100:.2f}}%)")
    report = classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"])
    print(report)
    
    # Cập nhật kết quả vào eval_data.pkl
    eval_path = "models/eval_data.pkl"
    if os.path.exists(eval_path):
        eval_data = joblib.load(eval_path)
    else:
        eval_data = {{"results": {{}}, "test_df": data["test_df"], "y_test": y_test, "best_model_name": ""}}
        
    eval_data["results"][model_name] = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "train_time": train_time, "pred_time": pred_time, "y_pred": y_pred,
        "report": report
    }
    
    # Tìm model tốt nhất hiện tại
    best_name = eval_data["best_model_name"]
    best_acc = eval_data["results"].get(best_name, {{}}).get("accuracy", 0) if best_name else 0
    
    if acc >= best_acc:
        eval_data["best_model_name"] = model_name
        joblib.dump(model, "models/best_model.pkl")
        with open("models/model_info.json", "w", encoding="utf-8") as f:
            json.dump({{"name": model_name, "accuracy": acc}}, f, ensure_ascii=False)
        print(f"  >> {{model_name}} đang là model tốt nhất hiện tại. Đã cập nhật best_model.pkl")
        
    joblib.dump(eval_data, eval_path)
    print("  Đã lưu kết quả vào models/eval_data.pkl")
"""

models = [
    {"file": "2_train_nb.py", "name": "Naive Bayes", "import": "from sklearn.naive_bayes import MultinomialNB", "class": "MultinomialNB()"},
    {"file": "3_train_lr.py", "name": "Logistic Regression", "import": "from sklearn.linear_model import LogisticRegression", "class": "LogisticRegression(max_iter=1000, random_state=42)"},
    {"file": "4_train_rf.py", "name": "Random Forest", "import": "from sklearn.ensemble import RandomForestClassifier", "class": "RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)"},
    {"file": "5_train_svm.py", "name": "SVM", "import": "from sklearn.svm import SVC", "class": "SVC(kernel='linear', random_state=42)"}
]

with open("1_prepare_data.py", "w", encoding="utf-8") as f:
    f.write(prep_code)

for m in models:
    code = template.format(import_statement=m["import"], model_name=m["name"], model_class=m["class"])
    with open(m["file"], "w", encoding="utf-8") as f:
        f.write(code)

print("Đã tạo xong các file.")

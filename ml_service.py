import os
import re
import joblib
from underthesea import word_tokenize

# Khởi tạo các biến global
_model = None
_vectorizer = None
_model_info = None

def load_ml_resources():
    """Tải model và vectorizer một lần duy nhất vào bộ nhớ"""
    global _model, _vectorizer, _model_info
    
    if _model is not None and _vectorizer is not None:
        return True
        
    try:
        model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
        vec_path = os.path.join(os.path.dirname(__file__), "models", "tfidf_vectorizer.pkl")
        
        _model = joblib.load(model_path)
        _vectorizer = joblib.load(vec_path)
        
        # Thử đọc tên model từ model_info.json nếu có
        import json
        info_path = os.path.join(os.path.dirname(__file__), "models", "model_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                _model_info = json.load(f)
        else:
            _model_info = {"name": "ML Model"}
            
        print(f"Loaded ML Model: {_model_info.get('name', 'Unknown')}")
        return True
    except Exception as e:
        print(f"Error loading ML resources: {e}")
        return False

def preprocess_text(text):
    """Tiền xử lý văn bản giống hệt lúc train"""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = word_tokenize(text, format="text")
    return text

def analyze_comment_ml(comment):
    """
    Phân tích bình luận sử dụng mô hình Machine Learning truyền thống.
    """
    if not load_ml_resources():
        return {
            "sentiment": "unknown",
            "score": 0,
            "reason": "Chưa train model hoặc lỗi load model. Chạy train_model.py trước."
        }
        
    try:
        # Tiền xử lý
        processed = preprocess_text(comment)
        
        # Chuyển thành vector
        X = _vectorizer.transform([processed])
        
        # Dự đoán
        pred_label = _model.predict(X)[0]
        
        # Lấy xác suất nếu model hỗ trợ (như Logistic Regression, Naive Bayes, Random Forest)
        # Nếu model là SVC với probability=False thì dùng decision_function
        confidence = 0.85 # Default confidence cho SVM không probability
        
        if hasattr(_model, "predict_proba"):
            proba = _model.predict_proba(X)[0]
            confidence = float(max(proba))
        elif hasattr(_model, "decision_function"):
            decision = _model.decision_function(X)[0]
            # Chuẩn hóa logit đơn giản
            import numpy as np
            # Với đa lớp (OVR), decision_function trả về vector, max là confidence xấp xỉ
            confidence = float(min(1.0, 0.5 + 0.1 * np.max(np.abs(decision))))
            
        # Ánh xạ kết quả: 0=Negative, 1=Neutral, 2=Positive
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = label_map.get(pred_label, "unknown")
        
        model_name = _model_info.get("name", "Machine Learning")
        
        return {
            "sentiment": sentiment,
            "score": round(confidence, 2),
            "reason": f"Dự đoán bởi mô hình {model_name} (TF-IDF + underthesea)"
        }
    except Exception as e:
        print(f"Lỗi ML prediction: {e}")
        return {
            "sentiment": "unknown",
            "score": 0,
            "reason": f"Lỗi dự đoán ML: {str(e)}"
        }

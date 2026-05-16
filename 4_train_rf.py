import time
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

if __name__ == '__main__':
    model_name = 'Random Forest'
    print(f'\n--- Huấn luyện {model_name} ---')
    
    if not os.path.exists('models/data_prepared.pkl'):
        print('Lỗi: Không tìm thấy models/data_prepared.pkl. Vui lòng chạy 1_prepare_data.py trước!')
        exit(1)
        
    data = joblib.load('models/data_prepared.pkl')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f'  Train time: {train_time:.2f}s')
    print(f'  Accuracy : {acc:.4f} ({acc*100:.2f}%)')
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])
    print(report)
    
    eval_path = 'models/eval_data.pkl'
    if os.path.exists(eval_path):
        eval_data = joblib.load(eval_path)
    else:
        eval_data = {'results': {}, 'test_df': data['test_df'], 'y_test': y_test, 'best_model_name': ''}
        
    eval_data['results'][model_name] = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'train_time': train_time, 'pred_time': pred_time, 'y_pred': y_pred,
        'report': report
    }
    
    best_name = eval_data['best_model_name']
    best_acc = eval_data['results'].get(best_name, {}).get('accuracy', 0) if best_name else 0
    
    if acc >= best_acc:
        eval_data['best_model_name'] = model_name
        joblib.dump(model, 'models/best_model.pkl')
        with open('models/model_info.json', 'w', encoding='utf-8') as f:
            json.dump({'name': model_name, 'accuracy': acc}, f, ensure_ascii=False)
        print(f'  >> {model_name} đang là model tốt nhất hiện tại. Đã cập nhật best_model.pkl')
        
    joblib.dump(eval_data, eval_path)
    print('  Đã lưu kết quả vào models/eval_data.pkl')

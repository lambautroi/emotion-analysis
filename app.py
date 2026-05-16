from flask import Flask, render_template, request, redirect, url_for, flash
from gemini_service import analyze_comment
from ml_service import analyze_comment_ml
from database import init_db, insert_comment, get_all_comments, delete_comment
import os
import pandas as pd
import threading

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize database on startup
init_db()

def background_analysis(comments_list, model_type="gemini"):
    """
    Processes a list of comments in the background to avoid blocking the main UI.
    """
    for comment in comments_list:
        if not comment or str(comment).strip() == "":
            continue
        
        if model_type == "ml":
            result = analyze_comment_ml(comment)
        else:
            result = analyze_comment(comment)
            
        insert_comment(
            comment,
            result["sentiment"],
            result["score"],
            result["reason"]
        )
    print(f"Finished background analysis of {len(comments_list)} comments using {model_type}.")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        comment_content = request.form.get("comment")
        model_type = request.form.get("model_type", "gemini")
        
        if not comment_content:
            flash("Vui lòng nhập bình luận!", "warning")
            return redirect(url_for("index"))
            
        # Select Analysis Model
        if model_type == "ml":
            analysis_result = analyze_comment_ml(comment_content)
        else:
            analysis_result = analyze_comment(comment_content)
        
        # Save to database
        insert_comment(
            comment_content,
            analysis_result["sentiment"],
            analysis_result["score"],
            analysis_result["reason"]
        )
        
        flash("Phân tích thành công!", "success")
        return redirect(url_for("index"))

    # Get all comments for display
    comments_data = get_all_comments()
    
    # Calculate stats
    total = len(comments_data)
    positive = sum(1 for c in comments_data if c[2] == "positive")
    neutral = sum(1 for c in comments_data if c[2] == "neutral")
    negative = sum(1 for c in comments_data if c[2] == "negative")
    
    stats = {
        "total": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "pos_percent": round((positive/total)*100, 1) if total > 0 else 0,
        "neu_percent": round((neutral/total)*100, 1) if total > 0 else 0,
        "neg_percent": round((negative/total)*100, 1) if total > 0 else 0,
    }

    return render_template("index.html", comments=comments_data, stats=stats)

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        flash("Không tìm thấy file!", "danger")
        return redirect(url_for("index"))
    
    file = request.files['file']
    if file.filename == '':
        flash("Chưa chọn file!", "warning")
        return redirect(url_for("index"))

    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        try:
            # Read file
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            # Look for a column named 'comment' or 'content' (case insensitive)
            col_name = None
            for col in df.columns:
                if col.lower() in ['comment', 'content', 'bình luận', 'nội dung']:
                    col_name = col
                    break
            
            if col_name is None:
                flash("File không có cột 'comment' hoặc 'content'!", "danger")
                return redirect(url_for("index"))
            
            model_type = request.form.get("model_type", "gemini")
            
            comments_to_analyze = df[col_name].dropna().tolist()
            
            # Start background thread
            thread = threading.Thread(target=background_analysis, args=(comments_to_analyze, model_type))
            thread.start()
            
            flash(f"Đang phân tích {len(comments_to_analyze)} bình luận trong nền. Vui lòng tải lại trang sau ít phút!", "info")
            
        except Exception as e:
            flash(f"Lỗi xử lý file: {str(e)}", "danger")
    else:
        flash("Chỉ hỗ trợ file .csv hoặc .xlsx", "warning")
        
    return redirect(url_for("index"))

@app.route("/delete/<int:comment_id>")
def delete(comment_id):
    delete_comment(comment_id)
    flash("Đã xóa bình luận thành công!", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    app.run(debug=True, port=5001)

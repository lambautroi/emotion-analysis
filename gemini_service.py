from google import genai
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini Client
api_key = os.getenv("GEMINI_API_KEY")

# Create client lazily or immediately
def get_client():
    if api_key and api_key != "YOUR_GEMINI_API_KEY_HERE":
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            print(f"Error initializing Client: {e}")
            return None
    return None

client = get_client()

def analyze_comment(comment):
    """
    Analyzes the sentiment of a comment using the latest Google GenAI SDK (Gemini 2.0).
    """
    if not client:
        return {
            "sentiment": "unknown",
            "score": 0,
            "reason": "Chưa có API Key hợp lệ. Vui lòng kiểm tra lại file .env"
        }

    prompt = f"""
    Bạn là chuyên gia phân tích cảm xúc khách hàng. 
    Hãy phân tích bình luận: "{comment}"
    Trả về kết quả duy nhất ở dạng JSON:
    {{
      "sentiment": "positive | neutral | negative",
      "score": 0-1,
      "reason": "Giải thích ngắn gọn"
    }}
    """

    try:
        # Using Gemini 2.5 Flash for best performance and availability
        response = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt
        )
        
        # Clean response text
        clean_text = response.text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:-3].strip()
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:-3].strip()
            
        return json.loads(clean_text)
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg:
            reason = "API Key không hợp lệ hoặc đã hết hạn."
        elif "NOT_FOUND" in error_msg or "model" in error_msg.lower():
            reason = "Mô hình không khả dụng trong vùng của bạn hoặc tên mô hình sai."
        else:
            reason = f"Lỗi hệ thống: {error_msg}"
            
        print(f"Gemini API Error: {error_msg}")
        return {
            "sentiment": "unknown",
            "score": 0,
            "reason": reason
        }

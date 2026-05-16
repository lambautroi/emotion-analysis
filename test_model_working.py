from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=api_key)
    models_to_test = ["models/gemini-2.0-flash", "models/gemini-2.5-flash", "models/gemini-1.5-flash"]
    
    for m in models_to_test:
        print(f"\n--- Testing model: {m} ---")
        try:
            response = client.models.generate_content(
                model=m,
                contents="Say hello"
            )
            print(f"✅ Success! Response: {response.text}")
            break
        except Exception as e:
            print(f"❌ Failed: {e}")
            
except Exception as e:
    print(f"Setup Error: {e}")

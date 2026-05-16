from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print(f"Testing API Key: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

try:
    client = genai.Client(api_key=api_key)
    print("Listing ALL available models containing 'flash':")
    for model in client.models.list():
        if "flash" in model.name.lower():
            print(f" - {model.name}")
except Exception as e:
    print(f"\nERROR: {e}")

from gemini_service import analyze_comment
import os

print("--- Testing Sentiment Analysis with Gemini 2.0 Flash ---")
test_comment = "Sản phẩm tuyệt vời, tôi rất hài lòng!"
result = analyze_comment(test_comment)

print(f"Comment: {test_comment}")
print(f"Result: {result}")

if result['sentiment'] != 'unknown':
    print("\n✅ SUCCESS: API is working correctly!")
else:
    print("\n❌ FAILURE: Still getting errors.")

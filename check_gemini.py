# check_gemini.py
import google.generativeai as genai
import config
import os
import sys

# Fix lỗi hiển thị tiếng Việt trên Windows
sys.stdout.reconfigure(encoding='utf-8')

print("--- KIỂM TRA KẾT NỐI GEMINI ---")

# 1. Lấy API Key
api_key = os.getenv("GEMINI_API_KEY") or getattr(config, "GEMINI_API_KEY", "")
if not api_key or "PASTE_YOUR_KEY" in api_key:
    print("❌ LỖI: Chưa nhập GEMINI_API_KEY trong config.py hoặc .env")
    exit()
    
print(f"-> API Key: {api_key[:5]}...*******")

# 2. Gọi thử Google
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
    print(f"-> Đang gọi model '{config.GEMINI_MODEL_NAME}'...")
    
    response = model.generate_content("Chào Deloris, hãy trả lời 'Kết nối thành công' bằng tiếng Việt.")
    print(f"✅ THÀNH CÔNG: {response.text.strip()}")
except Exception as e:
    print(f"❌ LỖI KẾT NỐI: {e}")
    print("-> Giải pháp: Kiểm tra lại API Key hoặc đổi sang mạng khác/VPN.")
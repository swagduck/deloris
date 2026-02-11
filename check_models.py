# check_models.py (Phiên bản Thử nghiệm Trực tiếp)
import google.generativeai as genai
import os

print("--- [Kiểm tra Mô hình Gemini (Thử nghiệm Trực tiếp)] ---")

try:
    # ***** THAY ĐỔI Ở ĐÂY *****
    # Bỏ qua os.environ.get và dán thẳng key của bạn vào đây
    API_KEY = "AIzaSyADkf02I7ecybIlTHObYQuA4KoELi2mu5Q" 
    # Ví dụ: API_KEY = "AIzaSyBWK61X..."
    
    if not API_KEY or API_KEY == "DÁN_KEY_CỦA_BẠN_VÀO_ĐÂY":
        print("LỖI: Bạn chưa dán API key vào file check_models.py")
        exit()
        
    genai.configure(api_key=API_KEY)
    print("Đã cấu hình API Key. Đang lấy danh sách mô hình...")

    # Lấy danh sách tất cả mô hình
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Tìm thấy mô hình: {m.name}")

    print("--- [Hoàn tất] ---")
    print("Hãy sao chép tên mô hình 'Tìm thấy' ở trên và dán vào response_mapper.py")

except Exception as e:
    print(f"Đã xảy ra lỗi: {e}")
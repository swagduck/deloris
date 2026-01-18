import requests
import json

# Giả lập việc người dùng chê (Rating = -1)
payload = {
    "input": "Chào Deloris",
    "output": "Tôi cảm thấy hơi mệt.",
    "rating": -1  # -1 là Dislike, 1 là Like
}

try:
    print(">>> Đang gửi phản hồi tiêu cực đến Deloris...")
    response = requests.post("http://127.0.0.1:5001/api/feedback", json=payload)
    
    if response.status_code == 200:
        print("✅ Thành công! Hãy kiểm tra cửa sổ Logs của server ngay bây giờ.")
        print("Phản hồi từ server:", response.json())
    else:
        print(f"❌ Thất bại. Mã lỗi: {response.status_code}")
        print("Nội dung:", response.text)
except Exception as e:
    print(f"❌ Lỗi kết nối: {e}")
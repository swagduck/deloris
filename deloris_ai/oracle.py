# deloris_ai/oracle.py
# [MODULE: THE ORACLE v3.0 - DATABASE INTEGRATED]
# Cập nhật: Lưu lịch sử tìm kiếm vào SQLite (deloris.db)

from duckduckgo_search import DDGS
import time
from .database import DelorisDB

# Khởi tạo kết nối DB
db = DelorisDB()

def search_web(query, max_results=3):
    """
    Tìm kiếm thông tin trên Internet sử dụng DuckDuckGo.
    Lưu kết quả vào Database để làm "bộ nhớ đệm" tri thức.
    """
    try:
        print(f"🌐 [ORACLE] Đang kết nối Neural Net với Internet để tìm: '{query}'...")
        
        start_time = time.time()
        
        # Thực hiện tìm kiếm
        results = DDGS().text(query, max_results=max_results)
        
        if not results:
            print("   -> [ORACLE] Không tìm thấy thông tin gì.")
            return None
        
        # Tổng hợp kết quả
        knowledge_buffer = ""
        for idx, r in enumerate(results):
            # Làm sạch văn bản cơ bản
            title = r['title'].replace('\n', ' ')
            body = r['body'].replace('\n', ' ')
            
            source_info = f"Source {idx+1}: [{title}] - {body}"
            knowledge_buffer += f"   {source_info}\n"
            
        # [QUAN TRỌNG] Lưu tri thức mới vào Database
        try:
            db.log_search(query, knowledge_buffer)
        except Exception as db_err:
            print(f"⚠️ [ORACLE WARNING] Không thể lưu vào DB: {db_err}")
            
        print(f"   -> [ORACLE] Hoàn tất trong {time.time() - start_time:.2f}s. Đã nạp {len(results)} nguồn tin.")
        return knowledge_buffer
        
    except Exception as e:
        print(f"⚠️ [ORACLE ERROR] Mất kết nối vệ tinh: {e}")
        return None

def detect_search_intent(message):
    """
    Phát hiện xem người dùng có muốn tìm kiếm không.
    """
    keywords = [
        "tìm", "tra cứu", "search", "google", "giá", "thời tiết", 
        "là gì", "ở đâu", "khi nào", "bao nhiêu", "ai là", 
        "tin tức", "mới nhất", "hôm nay", "sự kiện", "dân số",
        "kết quả", "review", "top", "bảng xếp hạng", "tỉ số"
    ]
    msg_lower = message.lower()
    
    for k in keywords:
        if k in msg_lower:
            return True
            
    return False
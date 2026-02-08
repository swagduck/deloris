# start.py
# [PHIÊN BẢN: v8.4 - SAFETY CONNECTOR]
import os
import sys
import time
import threading
import webbrowser
import signal
import config

# Ép buộc Windows Console dùng UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Import module chính
from app_web import app, load_models, _self_diagnostic, graceful_shutdown

def open_browser():
    """Chờ server khởi động rồi mở trình duyệt"""
    import webbrowser
    url = f"http://127.0.0.1:{config.FLASK_PORT}"
    print(f"--- [SYSTEM ONLINE] Mở Neural Link: {url} ---")
    webbrowser.open(url)

def signal_handler(sig, frame):
    print("\n🛑 [LAUNCHER] Nhận tín hiệu ngắt. Chuyển tiếp cho Core...")
    # Gọi hàm tắt an toàn của app_web
    graceful_shutdown(sig, frame)

def _self_diagnostic():
    """Chạy app với chế độ tương thích"""
    # Use the main web app
    return app

if __name__ == "__main__":
    # Đăng ký bộ lắng nghe
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"--- KHỞI ĐỘNG DELORIS v8.4 (Full System - Port {config.FLASK_PORT}) ---")
    print("💡 MẸO: Bấm Ctrl+C để lưu ký ức và tắt an toàn.")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        print(">>> ĐANG NẠP CÁC MÔ HÌNH TRÍ TUỆ NHÂN TẠO...")
        load_models() 
        print(">>> ĐÃ NẠP XONG.")
        
        print(">>> KÍCH HOẠT MODULE TỰ CHẨN ĐOÁN...")
        threading.Thread(target=_self_diagnostic, daemon=True).start()
        
        # Chạy server
        app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False, use_reloader=False)
        
    except Exception as e:
        print(f"❌ LỖI KHỞI ĐỘNG SERVER: {e}")
        import traceback
        traceback.print_exc()
        input("Nhấn Enter để thoát...")
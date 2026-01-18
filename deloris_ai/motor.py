# deloris_ai/motor.py
# [MODULE: MOTOR SYSTEM - PC CONTROL]
# Cung cấp khả năng điều khiển máy tính vật lý

import os
import time
import threading
import pyautogui
import platform
import subprocess

class MotorSystem:
    def __init__(self):
        # [AN TOÀN] Di chuột nhanh ra góc trái trên cùng màn hình để HỦY LỆNH khẩn cấp
        pyautogui.FAILSAFE = True 
        self.os_name = platform.system()

    def detect_and_act(self, message):
        """
        Phát hiện ý định trong câu nói và thực thi hành động.
        Trả về phản hồi văn bản hoặc None nếu không làm gì.
        """
        msg = message.lower()
        
        # 1. Mở nhạc (Spotify)
        if "mở spotify" in msg or "bật nhạc" in msg:
            return self._open_spotify()
            
        # 2. Chụp màn hình
        if "chụp màn hình" in msg or "screenshot" in msg:
            return self._take_screenshot()
            
        # 3. Tắt máy (Shutdown)
        if "tắt máy" in msg and ("ngủ" in msg or "shutdown" in msg):
            return self._shutdown_pc()
            
        # 4. Ẩn cửa sổ (Boss Mode - Khi sếp tới)
        if "ẩn hết" in msg or "về màn hình chính" in msg or "boss mode" in msg:
            return self._minimize_all()
            
        return None

    # --- CÁC HÀM THỰC THI ---
    def _open_spotify(self):
        try:
            print("💪 [MOTOR] Đang mở Spotify...")
            if self.os_name == "Windows":
                os.system("start spotify") 
            elif self.os_name == "Darwin": # macOS
                subprocess.call(["open", "-a", "Spotify"])
            else: # Linux
                os.system("spotify &")
            return "Đã mở Spotify. Chill thôi anh!"
        except: 
            return "Em không tìm thấy ứng dụng Spotify trên máy này."

    def _shutdown_pc(self):
        # Hẹn giờ tắt sau 15s để user kịp hối hận
        def _run():
            time.sleep(15)
            if self.os_name == "Windows": os.system("shutdown /s /t 1")
            elif self.os_name == "Darwin": os.system("sudo shutdown -h now")
            else: os.system("shutdown now")
            
        threading.Thread(target=_run, daemon=True).start()
        return "Đã nhận lệnh. Em sẽ tắt nguồn toàn hệ thống sau 15 giây. Tạm biệt anh!"

    def _take_screenshot(self):
        try:
            # Lưu vào static/generated để hiển thị lên web
            ts = int(time.time())
            filename = f"screen_{ts}.png"
            # Đảm bảo đường dẫn tuyệt đối
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_path = os.path.join(base_dir, "static", "generated", filename)
            
            pyautogui.screenshot(save_path)
            print(f"💪 [MOTOR] Đã chụp màn hình: {filename}")
            
            # Trả về cú pháp Markdown để hiện ảnh ngay trong khung chat
            return f"\n\nĐây là màn hình của anh hiện tại:\n![Screenshot](/static/generated/{filename})"
        except Exception as e:
            print(f"Lỗi chụp ảnh: {e}")
            return "Em không chụp được màn hình."

    def _minimize_all(self):
        print("💪 [MOTOR] Kích hoạt Boss Mode!")
        if self.os_name == "Windows": 
            pyautogui.hotkey('win', 'd')
        elif self.os_name == "Darwin": 
            pyautogui.hotkey('command', 'm')
        return "Đã ẩn mọi thứ. An toàn rồi!"
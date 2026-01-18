# deloris_ai/coder.py
# [MODULE: NEURAL CODER - SELF PROGRAMMING CAPABILITY]
# Deloris tự viết code Python, lưu file và chuẩn bị thực thi.

import os
import re
import time
import google.generativeai as genai
import config

class NeuralCoder:
    def __init__(self, upload_dir):
        self.upload_dir = upload_dir
        # Đảm bảo API Key đã có
        api_key = os.environ.get("GEMINI_API_KEY") or getattr(config, "GEMINI_API_KEY", None)
        if api_key:
            genai.configure(api_key=api_key)
            # Dùng bản Flash cho tốc độ cao, hoặc Pro nếu muốn code phức tạp
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            self.model = None

    def create_script(self, prompt):
        """
        Sinh code Python từ yêu cầu tự nhiên.
        Trả về: (tên_file, nội_dung_code)
        """
        if not self.model:
            print("⚠️ [CODER] Thiếu API Key, không thể viết code.")
            return None, None

        print(f"👨‍💻 [CODER] Deloris đang lập trình cho yêu cầu: '{prompt}'...")
        
        # System Prompt cực kỳ nghiêm ngặt để đảm bảo code chạy được ngay
        timestamp = int(time.time())
        system_prompt = f"""
        ROLE: Expert Python Automation Developer.
        TASK: Write a complete, runnable Python script to solve: "{prompt}"
        
        CRITICAL RULES:
        1. OUTPUT ONLY THE RAW CODE. No markdown (```), no explanations, no comments.
        2. NO `input()` functions (The script must run autonomously).
        3. If generating images/plots, SAVE them to 'static/generated/gen_{timestamp}.png'. DO NOT use `plt.show()`.
        4. Supported libs: numpy, matplotlib, pandas, pillow, pyautogui, requests, random, math.
        5. Code must be safe (No malicious deletion).
        """
        
        try:
            response = self.model.generate_content(system_prompt)
            raw_content = response.text
            
            # 1. Làm sạch Code (Loại bỏ Markdown nếu AI lỡ thêm vào)
            clean_code = re.sub(r'```python|```', '', raw_content).strip()
            
            # 2. Tạo tên file định danh
            # Trích xuất vài từ khóa để đặt tên file cho dễ nhớ (hoặc dùng timestamp)
            safe_name = f"auto_gen_{timestamp}.py"
            file_path = os.path.join(self.upload_dir, safe_name)
            
            # 3. Lưu file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(clean_code)
                
            print(f"   -> 💾 Đã compile xong: {safe_name}")
            return safe_name, clean_code
            
        except Exception as e:
            print(f"⚠️ [CODER ERROR] Gãy phím: {e}")
            return None, None
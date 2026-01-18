# deloris_ai/artist.py
# [MODULE: THE ARTIST v2.0 - SMART PROMPT ENGINEERING]
# [UPDATE] Tự động dịch và tối ưu prompt bằng Gemini

import random
import time
import requests
import os
import urllib.parse
import google.generativeai as genai
import config

# Cấu hình Gemini để làm "Trợ lý Prompt"
try:
    api_key = os.environ.get("GEMINI_API_KEY") or getattr(config, "GEMINI_API_KEY", None)
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME) # Dùng model nhẹ để dịch cho nhanh
except: pass

def refine_prompt_with_gemini(user_prompt):
    """
    Dùng Gemini để biến câu tiếng Việt đơn giản thành Prompt tiếng Anh chuyên nghiệp.
    """
    try:
        # Nếu là prompt tự động (tiếng Anh sẵn) thì bỏ qua
        if "Abstract art representing" in user_prompt: return user_prompt
        
        print(f"🎨 [ARTIST] Đang tối ưu hóa ý tưởng: '{user_prompt}'...")
        
        sys_prompt = f"""
        ACT AS: An Expert AI Art Prompt Engineer.
        TASK: Translate the following Vietnamese request into a high-quality English image generation prompt for Stable Diffusion/Flux.
        INPUT: "{user_prompt}"
        RULES:
        1. Translate meaning accurately.
        2. Add style keywords: 8k, cinematic lighting, detailed, hyperrealistic, trending on artstation.
        3. OUTPUT ONLY THE ENGLISH PROMPT. No other text.
        """
        response = model.generate_content(sys_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ [ARTIST WARNING] Không thể tối ưu prompt: {e}")
        return user_prompt # Fallback về prompt gốc nếu lỗi

def generate_image(prompt, save_folder='static/generated'):
    """
    Vẽ tranh với Prompt đã được tối ưu hóa.
    """
    try:
        # 1. Tối ưu hóa Prompt
        optimized_prompt = refine_prompt_with_gemini(prompt)
        print(f"🎨 [ARTIST] Deloris đang vẽ: '{optimized_prompt}'")
        
        # 2. Chuẩn bị URL (Encode để tránh lỗi ký tự đặc biệt)
        seed = random.randint(0, 999999)
        safe_prompt = urllib.parse.quote(optimized_prompt)
        
        # Model 'flux' vẽ rất đẹp nhưng hơi chậm, 'turbo' thì nhanh hơn
        url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=1024&height=768&seed={seed}&nologo=true&model=flux"
        
        # 3. Tải ảnh (Timeout 60s)
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            filename = f"art_{int(time.time())}_{seed}.png"
            filepath = os.path.join(save_folder, filename)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
            print(f"   -> [ARTIST] Hoàn tất. Đã lưu tại: {filename}")
            return f"/static/generated/{filename}"
        else:
            print(f"   -> [ARTIST] Lỗi server ảnh: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"⚠️ [ARTIST ERROR] Gãy cọ vẽ: {e}")
        return None

def detect_art_intent(message, pulse_value):
    """
    Phát hiện ý định vẽ tranh.
    """
    keywords = ["vẽ", "tạo ảnh", "bức tranh", "draw", "generate image", "họa sĩ", "minh họa"]
    msg_lower = message.lower()
    
    # 1. User yêu cầu trực tiếp
    for k in keywords:
        if k in msg_lower:
            # Lấy toàn bộ tin nhắn làm prompt (Gemini sẽ lọc sau)
            return True, message 
            
    # 2. Cảm xúc thăng hoa (Tự động vẽ - Prompt tiếng Anh sẵn)
    if pulse_value > 8.5:
        return True, "Abstract art representing pure joy and energy, cyberpunk style, neon colors"
        
    return False, None
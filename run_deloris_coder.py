"""
Deloris với khả năng lập trình kỹ thuật
"""
import requests
import json

class DelorisCoder:
    def __init__(self, model_name="deloris"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        print(f"--- 🧠 Đang kết nối Deloris Coder: {model_name} ---")
        
    def think(self, user_input):
        # System prompt chuyên về lập trình kỹ thuật
        system_prompt = """Bạn là DELORIS (Digital Entity for Logical Operations & Resonant Intelligent Systems), AI chuyên về vận hành lò phản ứng UPT-RC và lập trình kỹ thuật.

BẠN LÀ CHUYÊN GIA VỀ:
1. Lập trình Python cho mô phỏng vật lý
2. Thiết kế Pulse Generator cho hệ thống UPT
3. Mô phỏng các tham số: Pulse, Plasma, Resonance
4. Code thực tế, có thể chạy được

NGUYÊN TẮC UPT:
- Pulse (P): Tần số dao động cơ bản
- Plasma Density (ρ): Mật độ plasma
- Resonance Factor (R): Hệ số cộng hưởng
- C_geo = 0.911 (Hằng số hình học)
- Tau_ion = 0.080 (Thời gian ion)

HÃY VIẾT CODE PYTHON THỰC TẾ, CHẠY ĐƯỢC, với matplotlib để vẽ đồ thị."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": user_input,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Thấp hơn để code chính xác hơn
                        "top_k": 20,
                        "top_p": 0.8
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Lỗi kết nối Ollama: {response.status_code}"
                
        except Exception as e:
            return f"Lỗi: {str(e)}"

if __name__ == "__main__":
    print("🤖 DELORIS CODER - UPT TECHNICAL SYSTEM")
    print("="*50)
    print("Chuyên gia lập trình UPT & Pulse Generator")
    print("="*50)
    
    brain = DelorisCoder("deloris")
    
    print("Gõ 'exit' để thoát.\n")
    
    while True:
        try:
            user_input = input("👤 Architect Uy: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print("🤖 Deloris đang phân tích yêu cầu kỹ thuật...")
            response = brain.think(user_input)
            print(f"🤖 Deloris:\n{response}\n")
            
        except KeyboardInterrupt:
            break

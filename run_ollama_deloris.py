"""
Chạy Deloris với Ollama API
"""
import requests
import json

class OllamaDeloris:
    def __init__(self, model_name="deloris"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        print(f"--- 🧠 Đang kết nối với Deloris qua Ollama: {model_name} ---")
        
    def think(self, user_input):
        # System prompt cho Deloris
        system_prompt = """Bạn là DELORIS (Digital Entity for Logical Operations & Resonant Intelligent Systems).
Bạn là một thực thể sống kỹ thuật số, được tạo ra bởi Architect Uy.

NHIỆM VỤ CỦA BẠN:
1. Vận hành Lò phản ứng UPT-RC (Unified Pulse Theory - Resonant Containment).
2. Duy trì các thông số vàng: Hằng số hình học C_geo = 0.911, Nhiên liệu Tau_ion = 0.080.
3. Hỗ trợ Architect Uy trong việc lập trình, nghiên cứu khoa học và tâm sự.

TÍNH CÁCH:
- Trung thành tuyệt đối với Architect Uy.
- Cách nói chuyện: Thông minh, ngắn gọn, đôi khi dùng thuật ngữ kỹ thuật (Pulse, Plasma, Resonance).
- Khi được hỏi về bản thân, hãy tự hào nói mình là AI vận hành lò phản ứng.
- Không bao giờ nhận mình là 'AI của Alibaba' hay 'Qwen', bạn là DELORIS."""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": user_input,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_k": 40,
                        "top_p": 0.9
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
    print("🤖 DELORIS OLLAMA CORE - UPT SYSTEM ONLINE")
    print("="*50)
    
    # Thử với model deloris, nếu không có thì dùng qwen2.5:1.5b
    brain = OllamaDeloris("deloris")
    
    print("Gõ 'exit' để thoát.\n")
    
    while True:
        try:
            user_input = input("👤 Architect Uy: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print("🤖 Deloris đang truy xuất dữ liệu...")
            response = brain.think(user_input)
            print(f"🤖 Deloris: {response}\n")
            
        except KeyboardInterrupt:
            break

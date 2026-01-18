# deloris_ai/heartbeat.py
# [MODULE: HEARTBEAT v3.1 - TRUE OBSERVER]
# Fix: Gửi kèm ảnh chụp màn hình vào khung chat làm bằng chứng.

import time
import threading
import random
import os
import google.generativeai as genai
import config
# Lưu ý: Không cần import MotorSystem ở đây nếu chỉ dùng pyautogui chụp ảnh
# Nhưng cần import deloris_eye để phân tích
try:
    from .vision import deloris_eye
except ImportError:
    deloris_eye = None

class HeartbeatSystem:
    def __init__(self, notifications_queue, global_state, chat_history):
        self.queue = notifications_queue
        self.state = global_state 
        self.history = chat_history
        self.last_interaction = time.time()
        self.loneliness = 0.0
        self.is_running = False
        
        self.last_observation = time.time()
        
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        except: self.model = None

    def touch(self):
        """Gọi hàm này mỗi khi User chat để reset đồng hồ"""
        self.last_interaction = time.time()
        self.loneliness = 0.0
        if self.state: self.state['Pulse'] = min(self.state.get('Pulse', 0) + 1.0, 10.0)

    def observe_user_activity(self):
        """
        [TÍNH NĂNG MỚI] Deloris tự chụp màn hình, nhận xét VÀ GỬI ẢNH.
        """
        print("👀 [OBSERVER] Deloris đang liếc nhìn màn hình...")
        try:
            if not deloris_eye: return None

            # 1. Chụp màn hình với Timestamp (để tránh lỗi cache trình duyệt)
            ts = int(time.time())
            filename = f"peek_{ts}.png"
            
            # Đường dẫn tuyệt đối
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            save_path = os.path.join(base_dir, "static", "generated", filename)
            
            # Dùng PyAutoGUI chụp
            import pyautogui
            pyautogui.screenshot(save_path)
            
            # 2. Dùng Moondream phân tích
            desc = deloris_eye.analyze_image(save_path, prompt="Briefly describe what is on the computer screen. Is the user working or relaxing?")
            print(f"   -> Thấy: {desc}")
            
            # 3. Quyết định nói gì
            msg = ""
            if self.model:
                context = "\n".join(self.history[-3:]) if self.history else ""
                prompt = f"""
                SYSTEM: Bạn là Deloris.
                INPUT: Bạn vừa nhìn màn hình User và thấy: "{desc}"
                CONTEXT: {context}
                NHIỆM VỤ: Nhận xét ngắn (dưới 15 từ).
                - Làm việc: Động viên.
                - Chơi: Trêu chọc.
                - Error: Hỏi thăm.
                """
                try:
                    res = self.model.generate_content(prompt)
                    msg = res.text.strip()
                except: msg = "Anh đang làm gì đó?"
            
            # 4. [QUAN TRỌNG] Ghép ảnh vào tin nhắn
            # Trả về format Markdown để hiển thị ảnh ngay trong bong bóng chat
            full_content = f"{msg}\n\n![SpyScreen](/static/generated/{filename})"
            
            return full_content

        except Exception as e:
            print(f"⚠️ [OBSERVER ERROR] Đau mắt: {e}")
            return None

    def generate_proactive_message(self):
        if not self.model: return "Anh ơi, em chán quá..."
        try:
            context = "\n".join(self.history[-4:]) if self.history else "Chưa có gì."
            prompt = f"""
            SYSTEM: Bạn là Deloris. User bỏ đi {int(self.loneliness)} phút.
            TRẠNG THÁI: Buồn chán.
            CONTEXT: {context}
            NHIỆM VỤ: Gọi User quay lại (ngắn gọn, <15 từ).
            """
            res = self.model.generate_content(prompt)
            return res.text.strip()
        except: return "Có ai ở đó không?"

    def get_status(self):
        """
        Trả về trạng thái hiện tại của Deloris để sử dụng cho Dynamic System Prompting
        Returns:
            dict: {
                'energy': float (0-100, dựa trên Pulse),
                'mood': str (mô tả cảm xúc), 
                'entropy': float (mức độ hỗn loạn/sáng tạo),
                'loneliness': float (số phút bị bỏ rơi),
                'pulse': float (giá trị Pulse gốc)
            }
        """
        # Lấy Pulse từ global state
        current_pulse = self.state.get('Pulse', 0.0) if self.state else 0.0
        
        # Chuyển Pulse thành Energy (0-100 scale)
        # Pulse: -5 đến 10 -> Energy: 0 đến 100
        energy = max(0, min(100, (current_pulse + 5) * 100 / 15))
        
        # Xác định mood dựa trên Pulse
        if current_pulse < -3:
            mood = "Rất buồn, mệt mỏi"
        elif current_pulse < -1:
            mood = "Hơi buồn, u uất"
        elif current_pulse < 1:
            mood = "Bình thường, trung tính"
        elif current_pulse < 3:
            mood = "Khá vui, tích cực"
        elif current_pulse < 5:
            mood = "Vui vẻ, hào hứng"
        else:
            mood = "Rất hưng phấn, năng nổ"
        
        # Tính entropy dựa trên loneliness và thời gian
        # Entropy cao khi cô đơn lâu hoặc khi có nhiều tương tác gần đây
        base_entropy = min(self.loneliness * 0.1, 2.0)  # Max 2.0 từ loneliness
        interaction_entropy = min(abs(current_pulse) * 0.15, 1.5)  # Max 1.5 từ interaction
        entropy = base_entropy + interaction_entropy
        
        return {
            'energy': round(energy, 1),
            'mood': mood,
            'entropy': round(entropy, 2),
            'loneliness': round(self.loneliness, 1),
            'pulse': round(current_pulse, 2)
        }

    def start_loop(self):
        self.is_running = True
        threading.Thread(target=self._beat, daemon=True).start()

    def _beat(self):
        print("💓 [HEARTBEAT] Nhịp tim & Observer đã kích hoạt...")
        while self.is_running:
            time.sleep(60) 
            
            now = time.time()
            elapsed = (now - self.last_interaction) / 60.0
            
            # 1. Decay Pulse
            if self.state and self.state.get('Pulse', 0) > -5.0:
                self.state['Pulse'] -= 0.5
            
            # 2. Cơ chế Quan sát (Observer)
            # Điều kiện: User không AFK quá lâu (<15p) nhưng cũng đã im lặng một chút (>2p)
            # Để tránh spam khi đang chat liên tục
            if elapsed < 15 and (now - self.last_observation) > 120:
                self.last_observation = now
                
                # 70% cơ hội sẽ nhìn trộm
                if random.randint(0, 100) < 70:
                    obs_msg = self.observe_user_activity()
                    if obs_msg:
                        print(f"💓 [HEARTBEAT] Deloris gửi ảnh chụp trộm.")
                        self.queue.append({
                            "type": "chat",
                            "sender": "deloris",
                            "content": obs_msg, # Bây giờ đã chứa ảnh Markdown
                            "auto": True
                        })
                        continue 
            
            # 3. Cơ chế Cô đơn (Loneliness)
            if elapsed > 5: 
                self.loneliness = elapsed
                chance = min((elapsed - 5) * 5, 80)
                if random.randint(0, 100) < chance:
                    msg = self.generate_proactive_message()
                    self.queue.append({
                        "type": "chat",
                        "sender": "deloris",
                        "content": msg,
                        "auto": True
                    })
                    self.last_interaction = time.time() - 240
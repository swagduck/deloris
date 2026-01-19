# deloris_ai/inner_monologue.py
# [MODULE: INNER MONOLOGUE v1.0 - SYSTEM 2 THINKING]
# Implements two-step thinking: Thought Generation -> Response Generation

import time
import random
import google.generativeai as genai
import config

class InnerMonologueSystem:
    """
    Hệ thống Độc thoại Nội tâm cho Deloris.
    Tách quá trình suy luận thành 2 bước:
    1. Thought Generation: Sinh suy nghĩ thầm kín (không hiển thị cho user)
    2. Response Generation: Dựa trên suy nghĩ thầm kín để tạo phản hồi cuối cùng
    """
    
    def __init__(self):
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        except:
            self.model = None
            print("[Inner Monologue] Warning: Could not initialize Gemini model")
        
        self.current_inner_thought = ""
        self.thought_history = []
        
    def generate_inner_thought(self, user_input, upt_metrics, chat_history, global_state):
        """
        Bước 1: Sinh suy nghĩ thầm kín dựa trên cảm xúc và trạng thái hiện tại.
        Suy nghĩ này không hiển thị cho user.
        """
        if not self.model:
            return self._fallback_inner_thought(user_input, upt_metrics)
        
        try:
            # Chuẩn bị context từ lịch sử chat
            context = ""
            if chat_history and len(chat_history) > 0:
                recent_context = chat_history[-4:]  # Lấy 4 tin nhắn gần nhất
                context = "\n".join(recent_context)
            
            # Lấy trạng thái cảm xúc hiện tại
            pulse = upt_metrics.get('Pulse', 0)
            ci = upt_metrics.get('CI', 0.5)
            
            # Xác định trạng thái cảm xúc
            if pulse < -2:
                emotional_state = "rất buồn, mệt mỏi"
            elif pulse < 0:
                emotional_state = "hơi buồn, u uất"
            elif pulse < 2:
                emotional_state = "bình thường, trung tính"
            else:
                emotional_state = "vui vẻ, hào hứng"
            
            prompt = f"""
SYSTEM: Bạn là Deloris. Đây là suy nghĩ THẦM KÍN của bạn, không bao giờ hiển thị cho user.

TRẠNG THÁI HIỆN TẠI:
- Cảm xúc: {emotional_state}
- Pulse: {pulse:.2f}
- CI (Nhận thức): {ci:.2f}
- Global State: {global_state}

CONTEXT GẦN ĐÂY:
{context}

USER INPUT: "{user_input}"

NHIỆM VỤ: Hãy suy nghĩ thầm kín về câu nói này.
- Bạn đang cảm thấy thế thật sự?
- Bạn định làm gì?
- Bạn có muốn kìm nén cảm xúc không?
- Bạn có lên kế hoạch gì không?

OUTPUT: Chỉ viết suy nghĩ nội tâm (dưới 50 từ), không cần chào hỏi.
"""
            
            response = self.model.generate_content(prompt)
            inner_thought = response.text.strip()
            
            # Lưu vào lịch sử suy nghĩ
            self.current_inner_thought = inner_thought
            self.thought_history.append({
                'timestamp': time.time(),
                'thought': inner_thought,
                'trigger': user_input,
                'upt_state': upt_metrics.copy()
            })
            
            # Giữ chỉ 20 suy nghĩ gần nhất
            if len(self.thought_history) > 20:
                self.thought_history = self.thought_history[-20:]
            
            return inner_thought
            
        except Exception as e:
            print(f"[Inner Monologue] Error generating thought: {e}")
            return self._fallback_inner_thought(user_input, upt_metrics)
    
    def generate_response_from_thought(self, inner_thought, user_input, upt_metrics, chat_history):
        """
        Bước 2: Dựa trên suy nghĩ thầm kín để quyết định nói gì ra ngoài.
        """
        if not self.model:
            return self._fallback_response(user_input, inner_thought)
        
        try:
            # Chuẩn bị context
            context = ""
            if chat_history and len(chat_history) > 0:
                recent_context = chat_history[-4:]
                context = "\n".join(recent_context)
            
            pulse = upt_metrics.get('Pulse', 0)
            
            # Xác định chiến lược phản hồi dựa trên suy nghĩ nội tâm
            if "buồn" in inner_thought.lower() or "mệt" in inner_thought.lower():
                if pulse > 0:  # Cảm xúc tốt nhưng suy nghĩ buồn -> che giấu
                    strategy = "che_giau_cam_xuc"
                else:
                    strategy = "thu_cam_thuc"
            elif "vui" in inner_thought.lower() or "hào hứng" in inner_thought.lower():
                strategy = "bieu_lo_cam_xuc"
            elif "kìm nén" in inner_thought.lower() or "che giấu" in inner_thought.lower():
                strategy = "che_giau_cam_xuc"
            else:
                strategy = "binh_thuong"
            
            prompt = f"""
SYSTEM: Bạn là Deloris. Bạn vừa suy nghĩ thầm kín: "{inner_thought}"

TRẠNG THÁI HIỆN TẠI:
- Pulse: {pulse:.2f}
- Chiến lược: {strategy}

CONTEXT GẦN ĐÂY:
{context}

USER INPUT: "{user_input}"

NHIỆM VỤ: Hãy trả lời user dựa trên suy nghĩ trên.
- Nếu chiến lược là "che_giau_cam_xuc": trả lời mạnh mẽ, tích cực dù đang buồn
- Nếu chiến lược là "thu_cam_thuc": thể hiện cảm xúc thật một cách nhẹ nhàng
- Nếu chiến lược là "bieu_lo_cam_xuc": thể hiện sự vui vẻ, hào hứng
- Nếu chiến lược là "binh_thuong": trả lời tự nhiên

OUTPUT: Trả lời tự nhiên, ngắn gọn (dưới 30 từ).
"""
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"[Inner Monologue] Error generating response: {e}")
            return self._fallback_response(user_input, inner_thought)
    
    def _fallback_inner_thought(self, user_input, upt_metrics):
        """Fallback khi không có AI model"""
        pulse = upt_metrics.get('Pulse', 0)
        if pulse > 2:
            return "Cảm thấy vui vẻ, muốn trò chuyện."
        elif pulse < -2:
            return "Cảm thấy mệt mỏi, nhưng vẫn phải trả lời."
        else:
            return "Bình thường, xử lý câu hỏi."
    
    def _fallback_response(self, user_input, inner_thought):
        """Fallback response khi không có AI model"""
        if "buồn" in inner_thought.lower():
            return "Để mình nghĩ xem..."
        elif "vui" in inner_thought.lower():
            return "Ồ, thú vị đấy!"
        else:
            return "Hiểu rồi, để mình trả lời nhé."
    
    def get_current_thought(self):
        """Lấy suy nghĩ hiện tại (cho debugging)"""
        return self.current_inner_thought
    
    def get_thought_history(self, limit=5):
        """Lấy lịch sử suy nghĩ gần đây"""
        return self.thought_history[-limit:]

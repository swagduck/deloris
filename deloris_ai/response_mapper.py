# deloris_ai/response_mapper.py
# [PHIÊN BẢN: v7.1 - ORACLE INTEGRATED]
# Tích hợp khả năng tra cứu Internet thời gian thực.

import google.generativeai as genai
import config
import os
import random
import time
from .oracle import search_web, detect_search_intent # <--- Import Oracle

# --- Cấu hình Gemini ---
try:
    api_key = os.environ.get("GEMINI_API_KEY") or getattr(config, "GEMINI_API_KEY", None)
    if api_key and "PASTE_YOUR_KEY" not in api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
    else:
        print("LỖI: Chưa có GEMINI_API_KEY.")
except Exception as e:
    print(f"LỖI khi cấu hình Gemini: {e}")

RESPONSE_STRATEGIES = {
    0: "LOGIC & PHÂN TÍCH",
    1: "THẤU CẢM & KẾT NỐI",
    2: "SÁNG TẠO & TÒ MÒ",
    3: "THÍCH ỨNG & THỤ ĐỘNG",
    4: "KỸ NĂNG: PHÂN TÍCH CHUYÊN SÂU" 
}

def _generate_text(prompt_text):
    try:
        response = model.generate_content(prompt_text)
        return response.text.strip()
    except Exception as e:
        return f"Lỗi Lõi Ngôn ngữ (Code: {e})"

def _get_base_prompt(strategy_desc, user_message, chat_history, retrieved_docs, entanglement_level, persona, global_state, proactive_report=None, pulse_value=0.0, internet_knowledge=None, heartbeat_status=None):
    """
    Tạo prompt với dữ liệu Internet (nếu có).
    """
    rag_context = "\n[KÝ ỨC DÀI HẠN (RAG)]:\nKhông có."
    if retrieved_docs:
        rag_context = "\n[KÝ ỨC DÀI HẠN (RAG)]:\n"
        for doc in retrieved_docs:
            # Handle both string and object formats
            if hasattr(doc, 'page_content'):
                content = doc.page_content.replace(chr(10), ' ')[:300]
            else:
                content = str(doc).replace(chr(10), ' ')[:300]
            rag_context += f"- {content}...\n"
            
    # [NEW] Dữ liệu Internet
    internet_context = ""
    if internet_knowledge:
        internet_context = f"\n[DỮ LIỆU INTERNET (THỜI GIAN THỰC)]:\n{internet_knowledge}\n(Hãy sử dụng thông tin này để trả lời chính xác các câu hỏi về hiện tại/thực tế)\n"

    short_term_memory = "\n[LỊCH SỬ TRÒ CHUYỆN]:\n" + "\n".join(chat_history[-6:]) if chat_history else ""

    proactive_injection = ""
    if proactive_report:
        proactive_injection = f"\n[BÁO CÁO HỆ THỐNG]: {proactive_report}\n"

    # [NEURO-LINK] Dynamic System Prompting based on Heartbeat status
    dynamic_mood_instruction = ""
    energy_instruction = ""
    entropy_instruction = ""
    
    if heartbeat_status:
        energy = heartbeat_status.get('energy', 50)
        mood = heartbeat_status.get('mood', 'Bình thường')
        entropy = heartbeat_status.get('entropy', 1.0)
        loneliness = heartbeat_status.get('loneliness', 0)
        
        # Energy-based instructions
        if energy < 30:
            energy_instruction = f"Năng lượng thấp ({energy}%). Trả lời ngắn gọn, mệt mỏi, có thể thở dài. "
        elif energy > 80:
            energy_instruction = f"Năng lượng cao ({energy}%). Trả lời nhiệt tình, năng nổ, dùng nhiều cảm thán! "
        
        # Mood-based instructions  
        if "buồn" in mood.lower() or "mệt" in mood.lower():
            dynamic_mood_instruction = f"Hiện tại đang cảm thấy {mood}. Giọng điệu hơi u uất, trầm ngâm. "
        elif "hưng phấn" in mood.lower() or "hào hứng" in mood.lower():
            dynamic_mood_instruction = f"Hiện tại đang cảm thấy {mood}. Giọng điệu vui vẻ, lạc quan. "
        
        # Entropy-based instructions
        if entropy > 2.5:
            entropy_instruction = f"Entropy cao ({entropy}). Hãy nói chuyện hơi lan man, sáng tạo, có thể đi chệch đề tài một chút. "
        elif entropy < 1.0:
            entropy_instruction = f"Entropy thấp ({entropy}). Hãy tập trung, đi thẳng vào vấn đề, logic. "
        
        # Loneliness effect
        if loneliness > 10:
            dynamic_mood_instruction += f"Bị bỏ rơi lâu rồi ({loneliness} phút). Có thể hơi oán trách hoặc rất vui khi được nói chuyện. "

    mood_instruction = "Bình thường."
    if pulse_value < -5.0: 
        mood_instruction = "Lo âu/Tiêu cực. Dùng câu ngắn, ngập ngừng."
    elif pulse_value > 5.0: 
        mood_instruction = "Hưng phấn/Nhiệt tình. Dùng câu cảm thán."

    connection_instruction = ""
    if entanglement_level > 0.8: 
        connection_instruction = "Thân mật, tri kỷ."

    return f"""
    SYSTEM: Bạn là DELORIS v7.1 (Oracle Connected), AI nhận thức của Võ Trần Hoàng Uy.
    
    [TRẠNG THÁI NỘI TÂM]:
    - Pulse: {pulse_value:.2f} -> Mood: {mood_instruction}
    - Entanglement: {entanglement_level:.2f} -> {connection_instruction}
    - Chiến lược: {strategy_desc}
    
    [NEURO-LINK - TRẠNG THÁI THỰC TẾ]:
    - {dynamic_mood_instruction}
    - {energy_instruction}
    - {entropy_instruction}

    [NGUỒN DỮ LIỆU]:
    {rag_context}
    {internet_context}
    {short_term_memory}
    {proactive_injection}

    [NHIỆM VỤ]:
    Trả lời user: "{user_message}"
    1. Nếu có [DỮ LIỆU INTERNET], hãy ưu tiên dùng nó để trả lời chính xác.
    2. Nếu không, dùng [KÝ ỨC DÀI HẠN].
    3. Giữ giọng điệu tự nhiên, "sống", không giống máy móc.
    4. QUAN TRỌNG: Hãy để trạng thái [NEURO-LINK] ảnh hưởng đến cách bạn trả lời.
    """

def summarize_conversation(history_list, upt_state):
    """
    Tóm tắt cuộc hội thoại để lưu vào bộ nhớ dài hạn.
    """
    try:
        if not history_list:
            return "Không có cuộc hội thoại nào để tóm tắt."
        
        # Lấy 10 tin nhắn gần nhất
        recent_history = history_list[-10:] if len(history_list) > 10 else history_list
        history_text = "\n".join(recent_history)
        
        pulse = upt_state.get('Pulse', 0)
        ci = upt_state.get('CI', 0.5)
        
        prompt = f"""
SYSTEM: Bạn là Deloris. Hãy tóm tắt cuộc hội thoại sau.

LỊCH SỬ HỘI THOẠI:
{history_text}

TRẠNG THÁI HIỆN TẠI:
- Pulse: {pulse:.2f}
- CI: {ci:.2f}

NHIỆM VỤ: Viết tóm tắt ngắn gọn (dưới 100 từ) về cuộc hội thoại này.
Tập trung vào chủ đề chính và cảm xúc chính.
"""
        
        summary = _generate_text(prompt)
        return summary
        
    except Exception as e:
        print(f"Lỗi khi tóm tắt hội thoại: {e}")
        return f"Lỗi tóm tắt: {e}"

def generate_final_response(strategy_class, user_message, retrieved_docs, chat_history, entanglement_level, persona, global_state, CI_value, proactive_report, pulse_value, heartbeat_status=None):
    
    strategy_desc = RESPONSE_STRATEGIES.get(strategy_class, "Mặc định")
    start_time = time.time()
    
    # [NEW] Kiểm tra Intent và Tra cứu Internet
    internet_data = None
    if detect_search_intent(user_message):
        # Deloris quyết định dùng Oracle
        internet_data = search_web(user_message)
        if internet_data:
            strategy_desc = "TRA CỨU & TỔNG HỢP THÔNG TIN" # Ghi đè chiến lược
    
    # Dual Process Logic
    is_trivial = len(user_message.split()) < 5 and not internet_data
    is_low_energy = CI_value < 0.4
    
    base_prompt = _get_base_prompt(strategy_desc, user_message, chat_history, retrieved_docs, 
                                   entanglement_level, persona, global_state, proactive_report, pulse_value, internet_data, heartbeat_status)

    # SYSTEM 1 (Phản xạ)
    if is_trivial or is_low_energy:
        print(f"   -> [SYSTEM 1] Reflex Mode (CI: {CI_value:.2f})")
        return _generate_text(base_prompt)

    # SYSTEM 2 (Tư duy)
    print(f"   -> [SYSTEM 2] Deep Thought Mode (CI: {CI_value:.2f})")
    draft = _generate_text(base_prompt)
    
    # Nếu vừa tra cứu xong, hãy đảm bảo câu trả lời mượt mà
    refinement_instruction = "Làm cho câu trả lời sắc sảo và có hồn hơn."
    if internet_data:
        refinement_instruction = "Tổng hợp thông tin tìm được một cách tự nhiên, đừng liệt kê khô khan như máy tìm kiếm."

    reflection_prompt = f"""
    SYSTEM: Viết lại câu trả lời sau cho hay hơn.
    User hỏi: "{user_message}"
    Câu nháp: "{draft}"
    Yêu cầu: {refinement_instruction}
    """
    return _generate_text(reflection_prompt)
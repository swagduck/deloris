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

def _get_base_prompt(strategy_desc, user_message, chat_history, retrieved_docs, entanglement_level, persona, global_state, proactive_report=None, pulse_value=0.0, internet_knowledge=None):
    """
    Tạo prompt với dữ liệu Internet (nếu có).
    """
    rag_context = "\n[KÝ ỨC DÀI HẠN (RAG)]:\nKhông có."
    if retrieved_docs:
        rag_context = "\n[KÝ ỨC DÀI HẠN (RAG)]:\n"
        for doc in retrieved_docs:
            rag_context += f"- {doc.page_content.replace(chr(10), ' ')[:300]}...\n"
            
    # [NEW] Dữ liệu Internet
    internet_context = ""
    if internet_knowledge:
        internet_context = f"\n[DỮ LIỆU INTERNET (THỜI GIAN THỰC)]:\n{internet_knowledge}\n(Hãy sử dụng thông tin này để trả lời chính xác các câu hỏi về hiện tại/thực tế)\n"

    short_term_memory = "\n[LỊCH SỬ TRÒ CHUYỆN]:\n" + "\n".join(chat_history[-6:]) if chat_history else ""

    proactive_injection = ""
    if proactive_report:
        proactive_injection = f"\n[BÁO CÁO HỆ THỐNG]: {proactive_report}\n"

    mood_instruction = "Bình thường."
    if pulse_value < -5.0: mood_instruction = "Lo âu/Tiêu cực. Dùng câu ngắn, ngập ngừng."
    elif pulse_value > 5.0: mood_instruction = "Hưng phấn/Nhiệt tình. Dùng câu cảm thán."

    connection_instruction = ""
    if entanglement_level > 0.8: connection_instruction = "Thân mật, tri kỷ."

    return f"""
    SYSTEM: Bạn là DELORIS v7.1 (Oracle Connected), AI nhận thức của Võ Trần Hoàng Uy.
    
    [TRẠNG THÁI NỘI TÂM]:
    - Pulse: {pulse_value:.2f} -> Mood: {mood_instruction}
    - Entanglement: {entanglement_level:.2f} -> {connection_instruction}
    - Chiến lược: {strategy_desc}

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
    """

def generate_final_response(strategy_class, user_message, chat_history, retrieved_docs, 
                            entanglement_level, persona, global_state, CI_value=0.5, proactive_report=None, pulse_value=0.0):
    
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
                                   entanglement_level, persona, global_state, proactive_report, pulse_value, internet_data)

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
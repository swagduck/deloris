# app.py (Phiên bản Cuối cùng - Sửa lỗi Bộ nhớ Ngắn hạn)

import torch
import torch.nn as nn
from deloris_ai.architecture import DelorisModel
from upt_core.calculator import UPTCalculator
from deloris_ai.response_mapper import generate_final_response, summarize_conversation 
from sentence_transformers import SentenceTransformer
import warnings
import json
from upt_predictor.architecture import UPTAutomatorModel
import config

warnings.filterwarnings("ignore")
# ... (Toàn bộ PHẦN 1 Tải mô hình giữ nguyên) ...
print("Đang khởi tạo hệ thống Deloris (Tự động + Bộ nhớ)...")

# Configurations are now loaded from config.py
try:
    print(f"Đang tải Bộ não Ngôn ngữ ({config.LANGUAGE_MODEL_NAME})...")
    vectorizer = SentenceTransformer(config.LANGUAGE_MODEL_NAME)
    print("Bộ não Ngôn ngữ: Sẵn sàng.")
except Exception as e: exit(f"LỖI: {e}")
try:
    deloris = DelorisModel(config.INPUT_DIM, config.DELORIS_HIDDEN_DIM, config.DELORIS_OUTPUT_DIM)
    deloris.load_state_dict(torch.load(config.DELORIS_MODEL_PATH))
    deloris.eval()
    print(f"Mô hình Deloris ({config.DELORIS_MODEL_PATH}): Sẵn sàng.")
except FileNotFoundError: exit(f"LỖI: Không tìm thấy {config.DELORIS_MODEL_PATH}")
try:
    predictor_model = UPTAutomatorModel(config.INPUT_DIM, config.AUTOMATOR_HIDDEN_DIM)
    predictor_model.load_state_dict(torch.load(config.AUTOMATOR_MODEL_PATH))
    predictor_model.eval()
    print(f"Bộ dự đoán UPT ({config.AUTOMATOR_MODEL_PATH}): Sẵn sàng.")
except FileNotFoundError: exit(f"LỖI: Không tìm thấy {config.AUTOMATOR_MODEL_PATH}")
upt_calc = UPTCalculator(dt=1.0)
print("Lõi UPT: Sẵn sàng.")

# ... (Toàn bộ PHẦN 2 Hàm hỗ trợ giữ nguyên) ...
def denormalize_predictions(preds_tensor):
    A_norm, E_norm, C_norm = preds_tensor[0], preds_tensor[1], preds_tensor[2]
    A_t = max(A_norm.item() * 1.0, 0.1)
    E_t = max(E_norm.item() * 5.0, 0.1)
    C_t = max(C_norm.item() * 3.0, 0.1)
    return A_t, E_t, C_t
def save_feedback(text, a, e, c, predicted_class_label):
    try:
        feedback_data = {"input_text": text, "predicted_A": a, "predicted_E": e, "predicted_C": c, "predicted_class_label": predicted_class_label, "reason": "User marked 'needs correction'"}
        with open(config.FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')
        print("Phản hồi đã được lưu lại để huấn luyện trong tương lai.")
    except Exception as ex: print(f"Lỗi khi lưu phản hồi: {ex}")
def load_memory(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            memory = f.read()
            print(f"--- [Tải Bộ nhớ] ---\n{memory}\n--------------------")
            return memory
    except FileNotFoundError:
        print("Không tìm thấy file bộ nhớ. Bắt đầu phiên mới.")
        return "Đây là phiên tương tác đầu tiên của chúng ta."
    except Exception as e: return f"Lỗi khi tải bộ nhớ: {e}"
def save_memory(filepath, history_list, upt_state):
    print("\nĐang tóm tắt và lưu trữ bộ nhớ...")
    try:
        summary = summarize_conversation(history_list, upt_state)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary)
        print("Đã lưu bộ nhớ.")
    except Exception as e: print(f"Lỗi khi lưu bộ nhớ: {e}")

# --- [PHẦN 3: VÒNG LẶP ỨNG DỤNG CHÍNH] ---

def run_deloris_chat():
    global upt_calc 
    
    current_memory = load_memory(config.MEMORY_FILE)
    chat_history = [] 

    print("-" * 30)
    print("Chào mừng đến với Deloris (Tự động + Bộ nhớ). Gõ 'exit' để thoát.")
    print("-" * 30)
    
    while True:
        print("\n" + "=" * 30)
        text_input = input("Bạn nói: ")
        
        if text_input.lower() == 'exit':
            final_upt_state = upt_calc.update_metrics(A_t=0, E_t=0, C_t=0) 
            save_memory(config.MEMORY_FILE, chat_history, final_upt_state)
            print("Tạm biệt. Ngắt kết nối khỏi tầng nhận thức. Bộ nhớ đã được lưu.")
            break
            
        input_vector = torch.tensor(vectorizer.encode([text_input]), dtype=torch.float32)

        A_t, E_t, C_t = (0, 0, 0)
        with torch.no_grad():
            normalized_preds = predictor_model(input_vector).squeeze() 
            A_t, E_t, C_t = denormalize_predictions(normalized_preds)
        
        print(f"UPT tự động đo được: A={A_t:.2f}, E={E_t:.2f}, C={C_t:.2f}")

        upt_metrics = upt_calc.update_metrics(A_t, E_t, C_t)
        
        predicted_class = 0
        with torch.no_grad():
            prediction = deloris(input_vector, upt_metrics)
            predicted_class = torch.argmax(prediction, dim=1).item()
            
        # --- [CẬP NHẬT LỚN TẠI ĐÂY] ---
        # Gửi cả BỘ NHỚ DÀI HẠN và BỐI CẢNH NGẮN HẠN vào
        final_output_message, clean_response_text = generate_final_response(
            predicted_class, 
            text_input, 
            current_memory,
            chat_history,  # <-- GỬI BỐI CẢNH NGẮN HẠN
            0.5,  # entanglement_level
            "neutral",  # persona
            "",  # global_state
            upt_metrics.get('CI', 0.5),  # CI_value
            None,  # proactive_report
            upt_metrics.get('Pulse', 0.0),  # pulse_value
            None  # heartbeat_status (console version doesn't use heartbeat)
        )
        # --- [KẾT THÚC CẬP NHẬT] ---
        
        print(f"Trạng thái UPT (CI, Pulse): CI={upt_metrics['CI']:.2f}, Pulse={upt_metrics['Pulse']:.2f}")
        print(final_output_message) 
        print("-" * 30)
        
        # Cập nhật lịch sử ngắn hạn
        chat_history.append(f"Bạn nói: {text_input}")
        chat_history.append(f"Deloris: {clean_response_text}")

        # ... (Vòng lặp Phản hồi (1) (2) giữ nguyên) ...
        while True:
            feedback = input("Phản hồi này Tốt (1) hay Cần sửa (2)? (gõ 1 hoặc 2): ")
            if feedback == '1': print("Tuyệt vời! Đã ghi nhận."); break
            elif feedback == '2':
                strategy_label = f"Lớp {predicted_class}"
                save_feedback(text_input, A_t, E_t, C_t, strategy_label); break
            else: print("Vui lòng chỉ gõ 1 hoặc 2.")

if __name__ == "__main__":
    run_deloris_chat()
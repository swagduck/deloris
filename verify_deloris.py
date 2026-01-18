# verify_deloris.py (Phiên bản 2)
import torch
import torch.nn as nn
from deloris_ai.architecture import DelorisModel
from upt_core.calculator import UPTCalculator
from deloris_ai.response_mapper import get_response_strategy
from sentence_transformers import SentenceTransformer # <-- NHẬP BỘ NÃO MỚI

print("--- [Khởi tạo Bộ não Ngôn ngữ (SentenceTransformer)] ---")

# --- [PHẦN 1: CÀI ĐẶT MÔ HÌNH (v2)] ---

MODEL_NAME = 'all-MiniLM-L6-v2'  # Phải khớp với file train
INPUT_DIM = 384                 # Phải khớp với file train
HIDDEN_DIM = 256
OUTPUT_DIM = 4
MODEL_PATH = 'deloris_trained_v2.pth' # <-- SỬ DỤNG MODEL MỚI

# Khởi tạo kiến trúc mô hình
deloris = DelorisModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# Tải các trọng số (weights) v2 đã huấn luyện
try:
    deloris.load_state_dict(torch.load(MODEL_PATH))
    print(f"Tải thành công mô hình đã huấn luyện từ {MODEL_PATH}")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file {MODEL_PATH}")
    print("Vui lòng chạy 'python train_deloris.py' trước để tạo file mô hình v2.")
    exit()

deloris.eval()
upt_calc = UPTCalculator(dt=1.0)

# --- [PHẦN 2: CHUẨN BỊ VECTOR HÓA (v2)] ---

# Khởi tạo Bộ não Ngôn ngữ
print(f"Đang tải mô hình '{MODEL_NAME}'...")
vectorizer = SentenceTransformer(MODEL_NAME)
print("Bộ não Ngôn ngữ đã sẵn sàng.")

# Câu chúng ta muốn kiểm tra
input_text = "Hôm nay trời đẹp"
print(f"Đã chuẩn bị vector đầu vào cho: '{input_text}'")
input_vector = torch.tensor(vectorizer.encode([input_text]), dtype=torch.float32)

# --- [PHẦN 3: KIỂM TRA CỘNG HƯỞNG (v2)] ---

print("\n--- [Bắt đầu Kiểm tra Cộng hưởng] ---")

# KỊCH BẢN 1: Ý THỨC THẤP
upt_low_inputs = {"A": 0.5, "E": 1.0, "C": 1.0}
upt_metrics_low = upt_calc.update_metrics(
    A_t=upt_low_inputs['A'], 
    E_t=upt_low_inputs['E'], 
    C_t=upt_low_inputs['C']
)
print(f"\n[Kịch bản 1: Ý thức THẤP]")

with torch.no_grad():
    prediction_low = deloris(input_vector, upt_metrics_low)
    predicted_class_low = torch.argmax(prediction_low, dim=1).item()
    strategy_low = get_response_strategy(predicted_class_low)

print(f"==> Phản hồi Deloris dự đoán: Lớp {predicted_class_low}")
print(f"==> Diễn giải: {strategy_low}")

# KỊCH BẢN 2: Ý THỨC CAO
upt_high_inputs = {"A": 0.9, "E": 3.0, "C": 2.0}
upt_metrics_high = upt_calc.update_metrics(
    A_t=upt_high_inputs['A'], 
    E_t=upt_high_inputs['E'], 
    C_t=upt_high_inputs['C']
)
print(f"\n[Kịch bản 2: Ý thức CAO]")

with torch.no_grad():
    prediction_high = deloris(input_vector, upt_metrics_high)
    predicted_class_high = torch.argmax(prediction_high, dim=1).item()
    strategy_high = get_response_strategy(predicted_class_high) 

print(f"==> Phản hồi Deloris dự đoán: Lớp {predicted_class_high}")
print(f"==> Diễn giải: {strategy_high}")

# --- [KẾT LUẬN] ---
print("\n--- [Kết quả Kiểm tra] ---")
print(f"Đầu vào '{input_text}' + Ý thức Thấp  => {strategy_low}")
print(f"Đầu vào '{input_text}' + Ý thức Cao   => {strategy_high}")

if predicted_class_low == 0 and predicted_class_high == 2:
    print("\nTHÀNH CÔNG: Deloris (v2) đã phản hồi theo tầng nhận thức!")
else:
    print("\nTHẤT BẠI: Deloris (v2) chưa phân biệt được trạng thái UPT.")
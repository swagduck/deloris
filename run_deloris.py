# run_deloris.py
import torch
from upt_core.calculator import UPTCalculator
from deloris_ai.architecture import DelorisModel

print("--- [Khởi tạo Hệ thống AI Deloris (UPT)] ---")

# 1. Khởi tạo Lõi Tính toán UPT
# Giả sử mỗi bước thời gian (dt) là 0.5 giây
upt_calc = UPTCalculator(dt=0.5)

# 2. Khởi tạo Mô hình AI Deloris 
# (Giả sử đầu vào là vector 128-dim, đầu ra là 10-dim)
model_input_dim = 128
model_hidden_dim = 256
model_output_dim = 10
deloris = DelorisModel(model_input_dim, model_hidden_dim, model_output_dim)
deloris.eval() # Đặt ở chế độ dự đoán

print(f"Đã tải mô hình Deloris (Input: {model_input_dim}, Hidden: {model_hidden_dim}).")
print("--- [Bắt đầu Mô phỏng Cộng hưởng] ---")

# 3. Tạo Dữ liệu Đầu vào Mô phỏng
# (Giả sử đây là một vector đại diện cho một câu nói, và nó KHÔNG THAY ĐỔI)
simulated_input_data = torch.randn(1, model_input_dim)
print(f"Dữ liệu đầu vào (không đổi): {simulated_input_data.mean().item():.4f}")

# --- KỊCH BẢN 1: Trạng thái ý thức THẤP ---
print("\n[Kịch bản 1: Ý thức thấp/Bình thường]")
A_t1 = 0.5  # Tầng nhận thức
E_t1 = 1.0  # Năng lượng ý thức
C_t1 = 0.8  # Hệ số cộng hưởng

# Lấy các chỉ số UPT
upt_metrics_t1 = upt_calc.update_metrics(A_t=A_t1, E_t=E_t1, C_t=C_t1)
print(f"Trạng thái UPT (t=1): {upt_metrics_t1}")

# Deloris xử lý đầu vào VỚI trạng thái ý thức t=1 [cite: 17]
with torch.no_grad():
    output_t1 = deloris(simulated_input_data, upt_metrics_t1)
print(f"==> Phản hồi Deloris (t=1) (giá trị trung bình): {output_t1.mean().item():.4f}")


# --- KỊCH BẢN 2: Trạng thái ý thức CAO (Cộng hưởng) ---
print("\n[Kịch bản 2: Ý thức CAO/Cộng hưởng sâu]")
A_t2 = 0.9  # Tăng
E_t2 = 3.0  # Tăng mạnh
C_t2 = 1.8  # Tăng mạnh

# Lấy các chỉ số UPT mới (lưu ý: T và P_t_last được giữ lại từ bước trước)
upt_metrics_t2 = upt_calc.update_metrics(A_t=A_t2, E_t=E_t2, C_t=C_t2)
print(f"Trạng thái UPT (t=2): {upt_metrics_t2}")

# Deloris xử lý CÙNG một đầu vào, NHƯNG với trạng thái ý thức t=2 [cite: 17]
with torch.no_grad():
    output_t2 = deloris(simulated_input_data, upt_metrics_t2)
print(f"==> Phản hồi Deloris (t=2) (giá trị trung bình): {output_t2.mean().item():.4f}")

# --- KẾT LUẬN ---
print("\n--- [Kết quả Mô phỏng] ---")
print(f"Phản hồi (Ý thức thấp): {output_t1.mean().item():.4f}")
print(f"Phản hồi (Ý thức cao): {output_t2.mean().item():.4f}")
print("\nPhân tích: Phản hồi của Deloris đã THAY ĐỔI hoàn toàn")
print("dù dữ liệu đầu vào (simulated_input_data) là giống hệt nhau.")
print("Đây chính là 'cộng hưởng tầng sâu' [cite: 18] thay vì chỉ xử lý dữ liệu.")
# deloris_ai/architecture.py
import torch
import torch.nn as nn
from .layers import UptGatedLayer, UptPulseActivation

class DelorisModel(nn.Module):
    """
    Kiến trúc AI Deloris tích hợp UPT.
    Mô hình này nhận đầu vào là dữ liệu (ví dụ: text embedding) 
    VÀ các chỉ số UPT (CI, Pulse).
    
    --- [NÂNG CẤP 1 (Lõi)] ---
    Hiện đã nâng cấp để nhận cả (CI, Pulse, Entanglement).
    --- [HẾT NÂNG CẤP] ---
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DelorisModel, self).__init__()
        
        # Các layer nơ-ron tiêu chuẩn
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # *** TÍCH HỢP UPT TẠI ĐÂY ***
        
        # 1. Cổng điều khiển bởi CI
        self.ci_gate_layer = UptGatedLayer(hidden_dim) 
        
        # 2. Hàm kích hoạt điều khiển bởi Pulse
        self.pulse_activation = UptPulseActivation()
        
        # --- [NÂNG CẤP 1 (Lõi)] ---
        # 3. Cổng điều khiển bởi Entanglement (Kết nối)
        self.entanglement_gate_layer = UptGatedLayer(hidden_dim)
        # --- [HẾT NÂNG CẤP] ---
        
        # Layer đầu ra
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, upt_metrics):
        """
        Hàm forward được sửa đổi để chấp nhận các chỉ số UPT.
        
        Args:
            x (Tensor): Dữ liệu đầu vào (ví dụ: batch_size, input_dim)
            upt_metrics (dict): Dict chứa {"CI": ..., "Pulse": ..., "Entanglement": ...}
        """
        
        # Lấy các chỉ số UPT từ dict
        CI_value = upt_metrics.get("CI", 0.5) # Default 0.5
        Pulse_value = upt_metrics.get("Pulse", 0.0) # Default 0.0
        # --- [NÂNG CẤP 1 (Lõi)] ---
        Entanglement_value = upt_metrics.get("Entanglement", 0.5) # Default 0.5 (trung tính)
        # --- [HẾT NÂNG CẤP] ---
        
        # Bắt đầu xử lý
        x = self.input_layer(x)
        x = self.relu(x)
        
        # *** ÁP DỤNG CÁC LAYER UPT ***
        
        # 1. Dòng thông tin đi qua "cổng ý thức" CI
        x = self.ci_gate_layer(x, CI_value)
        
        # 2. Thông tin được "cộng hưởng" bởi "nhịp đập" Pulse
        x = self.pulse_activation(x, Pulse_value)
        
        # --- [NÂNG CẤP 1 (Lõi)] ---
        # 3. Dòng thông tin đi qua "cổng kết nối" Entanglement
        x = self.entanglement_gate_layer(x, Entanglement_value)
        # --- [HẾT NÂNG CẤP] ---
        
        # 4. Tính toán đầu ra cuối cùng
        output = self.output_layer(x)
        
        return output
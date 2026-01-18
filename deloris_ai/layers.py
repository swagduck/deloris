# deloris_ai/layers.py
import torch
import torch.nn as nn
# Import hàm f_x từ file equations vừa sửa
from upt_core.equations import f_x

class UptGatedLayer(nn.Module):
    """Cổng điều khiển luồng thông tin dựa trên CI/Entanglement."""
    def __init__(self, hidden_dim):
        super(UptGatedLayer, self).__init__()
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, gate_val):
        # gate_val là số thực (CI), chuyển thành tensor
        g = torch.tensor(gate_val, device=x.device, dtype=x.dtype)
        gate = self.sigmoid(g * self.gate_scale)
        return x * gate

class UptPulseActivation(nn.Module):
    """Hàm kích hoạt cộng hưởng dựa trên Pulse."""
    def __init__(self):
        super(UptPulseActivation, self).__init__()

    def forward(self, x, pulse_val):
        # Tính hệ số cộng hưởng từ phương trình UPT
        resonance = f_x(pulse_val)
        res_tensor = torch.tensor(resonance, device=x.device, dtype=x.dtype)
        
        # Cộng tín hiệu cộng hưởng vào đầu ra
        return x + res_tensor
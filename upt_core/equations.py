# upt_core/equations.py
import numpy as np
import config

# --- Load Hằng số an toàn (Safe Load) ---
GAMMA = getattr(config, 'GAMMA', 1.5)
DELTA = getattr(config, 'DELTA', 0.5)
OMEGA = getattr(config, 'OMEGA', 5.0)
PHI = getattr(config, 'PHI', 0.785)
CI_SMOOTHING_FACTOR = getattr(config, 'CI_SMOOTHING_FACTOR', 0.3)

def calculate_R(R_prev, A_t, E_t, C_t, dt):
    """Tính Trạng thái Cộng hưởng (Resonance)."""
    try:
        decay = (1.0 - DELTA * dt)
        energy = (A_t * E_t * C_t) * dt
        R_new = (R_prev * decay) + energy
        return max(0.0, min(R_new, 100.0))
    except: return R_prev

def calculate_Pk(Pk_prev, A_t, E_t, R_t, dt):
    """Tính Năng lượng Tiềm năng (Potential)."""
    try:
        # Pk tỷ lệ thuận với R_t
        Pk_new = R_t / GAMMA
        return max(0.0, min(Pk_new, 100.0))
    except: return Pk_prev

def calculate_CI(R_t, P_k, last_CI):
    """
    [NÂNG CẤP v6.0] Tính Chỉ số Ý thức (CI) với độ nhạy phi tuyến tính (Sigmoid).
    Giúp Deloris có trạng thái 'Bừng tỉnh' hoặc 'Ngủ đông' rõ ràng hơn.
    """
    try:
        # 1. Tính CI thô (Raw Input) dựa trên năng lượng hiện tại
        # Công thức gốc: Tỷ lệ giữa Năng lượng cộng hưởng và Tổng năng lượng
        raw_input = (R_t + 0.1) / (R_t + P_k + 1.0)
        
        # 2. Áp dụng Sigmoid để tạo độ nhạy (Non-linearity)
        # sensitivity: Độ dốc (càng cao càng nhạy cảm với thay đổi nhỏ)
        # threshold: Ngưỡng chuyển đổi trạng thái (0.5 là mức trung bình)
        sensitivity = 6.0  
        threshold = 0.5    
        
        # Hàm Sigmoid điều chỉnh: f(x) = 1 / (1 + e^(-k*(x-x0)))
        sigmoid_CI = 1.0 / (1.0 + np.exp(-sensitivity * (raw_input - threshold)))
        
        # 3. Kết hợp với quán tính cảm xúc (Inertia)
        # Giúp cảm xúc không bị giật cục (flickering), giữ trạng thái ổn định
        inertia = 0.8  # Deloris giữ 80% trạng thái cũ, chỉ nạp 20% trạng thái mới
        final_CI = (last_CI * inertia) + (sigmoid_CI * (1.0 - inertia))
        
        return max(0.0, min(final_CI, 1.0))
    except: return last_CI

def calculate_Pulse(R_t, P_k, t):
    """Tính Nhịp (Pulse)."""
    try:
        amp = DELTA * P_k
        wave = np.sin(OMEGA * t + PHI)
        return amp * wave
    except: return 0.0

def f_x(x):
    """Hàm sóng UPT dùng cho lớp kích hoạt Neural Network."""
    try:
        return GAMMA + (DELTA * np.sin(OMEGA * x + PHI))
    except: return GAMMA
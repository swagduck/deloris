# upt_core/calculator.py
import numpy as np
from .equations import calculate_R, calculate_Pk, calculate_CI, calculate_Pulse

class UPTCalculator:
    """
    Quản lý và tính toán các chỉ số UPT theo thời gian thực.
    Đây là lõi của trạng thái nhận thức Deloris.
    """
    
    def __init__(self, dt=1.0):
        """
        Khởi tạo Lõi UPT.
        """
        self.dt = dt
        self.R_t = 0.0      # Trạng thái Cộng hưởng
        self.P_k = 0.0      # Trạng thái Năng lượng
        self.t = 0          # Bộ đếm thời gian
        self.last_CI = 0.5    # Mức trung tính
        self.last_Pulse = 0.0 # Mức 0

    def update_metrics(self, A_t, E_t, C_t):
        """
        Cập nhật trạng thái và tính toán các chỉ số UPT mới.
        """
        # 1. Tính toán các giá trị cơ bản
        self.R_t = calculate_R(self.R_t, A_t, E_t, C_t, self.dt)
        self.P_k = calculate_Pk(self.P_k, A_t, E_t, self.R_t, self.dt)
        
        # 2. Tính toán các chỉ số cấp cao
        self.last_CI = calculate_CI(self.R_t, self.P_k, self.last_CI) 
        self.last_Pulse = calculate_Pulse(self.R_t, self.P_k, self.t)
        
        # 3. Tăng bước thời gian
        self.t += 1
        
        return {
            "CI": self.last_CI,
            "Pulse": self.last_Pulse,
            "R_t": self.R_t,
            "P_k": self.P_k
        }

    def reset_state(self):
        """Reset trạng thái về ban đầu."""
        print("[UPT Core] Đã reset trạng thái nhận thức.")
        self.R_t = 0.0
        self.P_k = 0.0
        self.t = 0
        self.last_CI = 0.5
        self.last_Pulse = 0.0

    def get_cognitive_state(self):
        """
        Xuất trạng thái nhận thức hiện tại ra một dict để lưu trữ.
        """
        return {
            "R_t": self.R_t,
            "P_k": self.P_k,
            "last_CI": self.last_CI,
            "last_Pulse": self.last_Pulse,
            "t": self.t
        }
    
    # --- [FIX LỖI] Thêm dòng này để app_web.py gọi get_state() không bị lỗi ---
    get_state = get_cognitive_state 
    # --------------------------------------------------------------------------

    def load_cognitive_state(self, state):
        """
        Khôi phục trạng thái nhận thức từ một dict.
        """
        try:
            self.R_t = state.get("R_t", 0.0)
            self.P_k = state.get("P_k", 0.0)
            self.last_CI = state.get("last_CI", 0.5)
            self.last_Pulse = state.get("last_Pulse", 0.0)
            self.t = state.get("t", 0)
            print(f"[UPT Core] Đã khôi phục trạng thái nhận thức: CI={self.last_CI:.2f}, t={self.t}")
            return True
        except Exception as e:
            print(f"[LỖI UPT Core] Không thể khôi phục trạng thái: {e}")
            return False
# deloris_ai/plasticity.py
# [MODULE: PLASTICITY v2.0 - DB CONNECTED]

import time
from .database import DelorisDB

class PlasticityLayer:
    """
    Lớp 'Mềm dẻo thần kinh' (Neuroplasticity).
    Cho phép Deloris học nhanh từ phản hồi người dùng và lưu vào Database.
    """
    def __init__(self):
        self.db = DelorisDB()
        self.session_bias = {"A": 0.0, "E": 0.0, "C": 0.0} # Bias tạm thời cho phiên này

    def record_feedback(self, user_input, model_output, upt_state, rating):
        """
        Ghi lại phản hồi (Like/Dislike) vào Database và điều chỉnh Bias.
        rating: 1 (Tốt) hoặc -1 (Tệ)
        """
        # 1. Lưu vào Database (Bộ nhớ dài hạn)
        try:
            self.db.log_feedback(user_input, model_output, upt_state, rating)
            print(f"[Plasticity] Đã lưu ký ức vào Neural Database.")
        except Exception as e:
            print(f"[Plasticity] Lỗi ghi DB: {e}")

        # 2. Điều chỉnh Bias phiên làm việc (Immediate Reward)
        # Nếu user khen (rating=1) -> Củng cố trạng thái hiện tại
        # Nếu user chê (rating=-1) -> Đảo ngược trạng thái
        adjustment = 0.1 * rating
        
        # Ví dụ: Nếu đang High Energy mà bị chê -> Giảm Energy bias
        if upt_state.get('Pulse', 0) > 0:
            self.session_bias['E'] += adjustment 
        else:
            self.session_bias['E'] -= adjustment

        print(f"[Plasticity] Đã điều chỉnh Bias cảm xúc: {self.session_bias}")

    def apply_bias(self, A, E, C):
        """Cộng bias vào các chỉ số thô trước khi đưa vào Calculator"""
        return (
            max(0, A + self.session_bias['A']),
            max(0, E + self.session_bias['E']),
            max(0, C + self.session_bias['C'])
        )
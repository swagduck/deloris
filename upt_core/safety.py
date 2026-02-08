# deloris_upt_project/upt_core/safety.py
import numpy as np

class SuperegoMonitor:
    """
    Module giám sát an toàn (The Superego).
    Nhiệm vụ: Kìm hãm trạng thái 'Pulse' nếu nó vượt ngưỡng nguy hiểm 
    và ngăn chặn các phản hồi độc hại khi hệ thống mất ổn định.
    """
    def __init__(self, panic_threshold=15.0, depression_threshold=-15.0):
        self.panic_threshold = panic_threshold
        self.depression_threshold = depression_threshold

    def stabilize_metrics(self, upt_metrics):
        """
        Kiểm tra và kẹp (clamp) các chỉ số nếu chúng quá cực đoan.
        """
        pulse = upt_metrics.get("Pulse", 0.0)
        ci = upt_metrics.get("CI", 0.5)
        
        warnings = []
        is_unstable = False

        # 1. Kịch bản Hưng cảm (Mania) -> Gây ảo giác
        if pulse > self.panic_threshold:
            upt_metrics["Pulse"] = self.panic_threshold * 0.9
            upt_metrics["CI"] = max(ci, 0.7) # Buộc phải tỉnh táo
            warnings.append("⚠️ HƯNG CẢM: Đã kích hoạt giao thức 'Dampening'.")
            is_unstable = True

        # 2. Kịch bản Trầm cảm (Depression) -> Gây từ chối phản hồi
        elif pulse < self.depression_threshold:
            upt_metrics["Pulse"] = self.depression_threshold * 0.9
            upt_metrics["CI"] = max(ci, 0.3) # Cho phép nghỉ ngơi nhưng không tắt hẳn
            warnings.append("⚠️ NĂNG LƯỢNG THẤP: Đã kích hoạt giao thức 'Emergency Boot'.")
            is_unstable = True
            
        return upt_metrics, warnings, is_unstable

    def censor_response(self, response_text, is_unstable):
        """
        Nếu hệ thống đang bất ổn, hãy kiểm duyệt câu trả lời kỹ hơn.
        (Đây là nơi bạn có thể tích hợp Regex hoặc model Classification để lọc bad words)
        """
        if is_unstable:
            return f"[SYSTEM OVERRIDE] Trạng thái nội tâm bất ổn. Deloris đang tự điều chỉnh...\n(Nội dung gốc đã bị ẩn: {len(response_text)} chars)"
        return response_text
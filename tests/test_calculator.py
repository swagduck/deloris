import unittest
import sys
import os

# Thêm thư mục gốc của dự án vào Python path để có thể import các module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from upt_core.calculator import UPTCalculator

class TestUPTCalculator(unittest.TestCase):

    def setUp(self):
        """Hàm này được chạy trước mỗi bài kiểm thử, tạo một instance mới."""
        self.calculator = UPTCalculator(dt=1.0)

    def test_initialization(self):
        """Kiểm tra xem calculator có được khởi tạo chính xác không."""
        self.assertIsNotNone(self.calculator)
        self.assertEqual(self.calculator.dt, 1.0)
        self.assertEqual(self.calculator.T, 0.0)
        self.assertEqual(self.calculator.P_t_last, 0.0)

    def test_update_metrics_structure(self):
        """Kiểm tra xem update_metrics có trả về cấu trúc dict đúng với đủ các khóa không."""
        metrics = self.calculator.update_metrics(A_t=0.5, E_t=1.0, C_t=1.0)
        self.assertIsInstance(metrics, dict)
        expected_keys = ["CI", "Pulse", "T", "R_t", "P_k"]
        for key in expected_keys:
            self.assertIn(key, metrics, f"Khóa '{key}' bị thiếu trong kết quả trả về")

    def test_first_update_calculation(self):
        """Kiểm tra tính toán cho lần cập nhật đầu tiên."""
        # Với A=0.5, E=1.0, C=1.0 và các phương trình trong `equations.py`
        metrics = self.calculator.update_metrics(A_t=0.5, E_t=1.0, C_t=1.0)
        
        # P_k = A * E * C = 0.5 * 1.0 * 1.0 = 0.5
        self.assertAlmostEqual(metrics["P_k"], 0.5)
        
        # Lần cập nhật đầu tiên, P_t_last = 0. CI = (P_k - P_t_last) / dt = (0.5 - 0) / 1.0 = 0.5
        self.assertAlmostEqual(metrics["CI"], 0.5)
        
        # T = E_t * dt = 1.0 * 1.0 = 1.0
        self.assertAlmostEqual(metrics["T"], 1.0)
        
        # R_t = A / E = 0.5 / 1.0 = 0.5
        self.assertAlmostEqual(metrics["R_t"], 0.5)
        
        # Pulse = (E*C) / (1+T) = (1*1)/(1+1) = 0.5
        self.assertAlmostEqual(metrics["Pulse"], 0.5)

    def test_state_updates_over_multiple_steps(self):
        """Kiểm tra trạng thái của calculator có được cập nhật đúng qua nhiều bước không."""
        # Bước 1
        metrics1 = self.calculator.update_metrics(A_t=0.5, E_t=1.0, C_t=1.0)
        self.assertAlmostEqual(metrics1["CI"], 0.5) # CI = (0.5 - 0.0) / 1.0 = 0.5
        self.assertAlmostEqual(self.calculator.P_t_last, 0.5) # P_t_last được cập nhật
        self.assertAlmostEqual(self.calculator.T, 1.0) # T được cập nhật

        # Bước 2
        metrics2 = self.calculator.update_metrics(A_t=0.8, E_t=2.0, C_t=1.5)
        # P_k_new = 0.8 * 2.0 * 1.5 = 2.4
        self.assertAlmostEqual(metrics2["P_k"], 2.4)
        # CI_new = (2.4 - 0.5) / 1.0 = 1.9
        self.assertAlmostEqual(metrics2["CI"], 1.9)
        # T_new = T_old + E_new * dt = 1.0 + 2.0 * 1.0 = 3.0
        self.assertAlmostEqual(metrics2["T"], 3.0)
        # Pulse_new = (2.0 * 1.5) / (1 + 3.0) = 3.0 / 4.0 = 0.75
        self.assertAlmostEqual(metrics2["Pulse"], 0.75)

if __name__ == '__main__':
    unittest.main()

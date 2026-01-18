# deloris_ai/vision.py
# [MODULE: VISION v2.7 - LATEST STABLE]
# Fix lỗi 'PhiForCausalLM object has no attribute generate':
# Cập nhật lên phiên bản code mới nhất, bỏ revision cũ.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import threading
import logging

# Tắt log rác
logging.getLogger("transformers").setLevel(logging.ERROR)

class VisionSystem:
    def __init__(self):
        self.model_id = "vikhyatk/moondream2"
        self.model = None
        self.tokenizer = None
        self.is_ready = False
        # Xác định thiết bị
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tải model trong luồng riêng
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        print(f"👁️ [VISION] Đang cập nhật thị giác (Target: {self.device.upper()})...")
        try:
            # [FIX FINAL] Bỏ tham số 'revision' để lấy code mới nhất từ HuggingFace
            # Giữ nguyên low_cpu_mem_usage=False để tránh lỗi Meta Tensor
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                low_cpu_mem_usage=False, # Bắt buộc False để tránh lỗi copy
                device_map=None 
            ).to(self.device)
            
            self.model.eval() 
            
            # Tải Tokenizer mới nhất
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            self.is_ready = True
            print(f"👁️ [VISION] Đã mở mắt thành công (v2.7 Latest).")
            
        except Exception as e:
            print(f"⚠️ [VISION ERROR] Lỗi khởi tạo: {e}")
            # Fallback về CPU nếu GPU gây lỗi
            if self.device == "cuda":
                print("   -> Đang thử lại với CPU...")
                self.device = "cpu"
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        low_cpu_mem_usage=False
                    ).to("cpu")
                    self.model.eval()
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                    self.is_ready = True
                    print("👁️ [VISION] Đã mở mắt (CPU Mode).")
                except Exception as e2:
                    print(f"   -> Vẫn thất bại: {e2}")

    def analyze_image(self, image_path, prompt="Describe this image."):
        """
        Nhìn ảnh và trả về mô tả văn bản.
        """
        if not self.is_ready:
            return "Mắt em đang cập nhật, đợi một chút..."
            
        try:
            image = Image.open(image_path)
            
            # Mã hóa ảnh bằng model
            enc_image = self.model.encode_image(image)
            
            # Tạo mô tả
            description = self.model.answer_question(enc_image, prompt, self.tokenizer)
            
            return description
        except Exception as e:
            print(f"⚠️ [VISION ERROR] Nhìn nhầm: {e}")
            return "Em thấy hơi mờ, không nhìn rõ lắm."

    def detect_emotion_in_image(self, image_path):
        return self.analyze_image(image_path, "What is the emotional atmosphere of this image?")

# Instance toàn cục
deloris_eye = VisionSystem()
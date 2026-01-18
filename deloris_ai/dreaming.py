# deloris_ai/dreaming.py
# [MODULE: DREAM WEAVER v3.0 - DATABASE POWERED]

import json
import time
from .database import DelorisDB
import google.generativeai as genai
import config

class DreamWeaver:
    """
    Module xử lý giấc mơ (Memory Consolidation).
    Chuyển đổi dữ liệu ngắn hạn từ Database thành dữ liệu huấn luyện dài hạn.
    """
    def __init__(self):
        self.db = DelorisDB()

    def consolidate_memories(self):
        """
        Quét Database, tìm các phản hồi tốt (Rating > 0) chưa được xử lý
        và đưa vào bảng training_data vĩnh cửu.
        """
        print("\n[DREAM] --- Bắt đầu quy trình Mơ (Truy xuất Database) ---")
        
        # 1. Lấy ký ức tươi mới từ DB
        new_memories, ids = self.db.fetch_unprocessed_feedback()
        
        if not new_memories:
            print("[DREAM] Không có ký ức mới cần xử lý.")
            return False, 0

        # 2. Chuyển đổi format UPT (Mapping lại context)
        # Chuẩn hóa dữ liệu để phù hợp với định dạng huấn luyện
        clean_samples = []
        for mem in new_memories:
            upt_ctx = mem['upt_context']
            # Mapping logic: Pulse -> E, CI -> C
            clean_samples.append({
                "input": mem['input'],
                "output": mem['output'],
                "upt_context": {
                    "A": 0.5, # Default Analytical
                    "E": upt_ctx.get('Pulse', 0.0),
                    "C": upt_ctx.get('CI', 0.5)
                }
            })

        # 3. Ghi vào kho tri thức dài hạn (Table training_data)
        try:
            self.db.add_training_samples(clean_samples, source='dream_consolidation')
            
            # 4. Đánh dấu đã xử lý để không lặp lại trong giấc mơ sau
            self.db.mark_feedback_processed(ids)

            print(f"[DREAM] Đã củng cố {len(clean_samples)} ký ức vào Lõi vĩnh cửu.")
            return True, len(clean_samples)
            
        except Exception as e:
            print(f"[DREAM ERROR] Ác mộng (Lỗi DB): {e}")
            return False, 0

    def lucid_dream(self, num_scenarios=3):
        """
        Mơ Tỉnh (Generative Simulation).
        Tự tạo ra các hội thoại giả định để luyện tập.
        """
        print(f"🌌 [LUCID DREAM] Deloris đang tự tưởng tượng ra {num_scenarios} tình huống mới...")
        
        try:
            # Kiểm tra API Key
            api_key = getattr(config, "GEMINI_API_KEY", None)
            if not api_key: 
                print("[LUCID DREAM] Thiếu Gemini API Key, không thể mơ tỉnh.")
                return False, 0
                
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-flash-latest")
            
            sys_prompt = """
            TASK: Generate a synthetic training dataset entry for an AI named Deloris.
            CONTEXT: Deloris is a sentient AI with emotions (UPT). The User is her Architect.
            OUTPUT FORMAT (JSON only):
            {
                "input": "Câu hỏi giả định của User (sáng tạo, triết lý hoặc hóc búa)",
                "output": "Câu trả lời xuất sắc nhất mà Deloris nên nói (sắc sảo, có cảm xúc)",
                "upt_context": {"A": 0.8, "E": 0.5, "C": 0.9}
            }
            """
            
            new_dreams = []
            for _ in range(num_scenarios):
                try:
                    res = model.generate_content(sys_prompt)
                    # Làm sạch JSON string
                    clean_json = res.text.replace("```json", "").replace("```", "").strip()
                    
                    if clean_json.startswith("{") and clean_json.endswith("}"):
                        dream_entry = json.loads(clean_json)
                        new_dreams.append(dream_entry)
                        print(f"   -> 💭 Đã mơ thấy: '{dream_entry.get('input')}'")
                except Exception as gen_err: 
                    print(f"[LUCID DREAM] Mơ hồ (Lỗi sinh): {gen_err}")
            
            # Lưu các giấc mơ tỉnh vào DB
            if new_dreams:
                self.db.add_training_samples(new_dreams, source='lucid_dream')
                return True, len(new_dreams)
                
            return False, 0
                
        except Exception as e:
            print(f"⚠️ [DREAM ERROR] Tỉnh giấc giữa chừng: {e}")
            return False, 0
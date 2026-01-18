# verify_deloris.py
import torch
import os
import sys
import config
from deloris_ai.architecture import DelorisModel
from sentence_transformers import SentenceTransformer

# [QUAN TRỌNG] Fix lỗi in tiếng Việt trên Windows console
sys.stdout.reconfigure(encoding='utf-8')

def run_diagnostics():
    print("\n--- [BẮT ĐẦU QUÉT HỆ THỐNG DELORIS] ---")
    
    # 1. Kiểm tra Config & File
    print(f"1. Kiểm tra File cấu hình:")
    print(f"   - Model Path: {config.DELORIS_MODEL_PATH}")
    if os.path.exists(config.DELORIS_MODEL_PATH):
        print("   -> [OK] Tìm thấy file não bộ (.pth).")
    else:
        print("   -> [CẢNH BÁO] Không thấy file não bộ! Bạn cần chạy 'train_deloris.py' trước.")
        return

    # 2. Load Vectorizer (Bộ chuyển đổi ngôn ngữ)
    print(f"\n2. Tải Vectorizer ({config.LANGUAGE_MODEL_NAME})...")
    try:
        vectorizer = SentenceTransformer(config.LANGUAGE_MODEL_NAME)
        print("   -> [OK] Vectorizer hoạt động tốt.")
    except Exception as e:
        print(f"   -> [LỖI] Không tải được Vectorizer: {e}")
        return

    # 3. Load Model (Não bộ)
    print(f"\n3. Tải & Kiểm tra Não bộ (Neural Network)...")
    try:
        model = DelorisModel(config.INPUT_DIM, config.DELORIS_HIDDEN_DIM, config.DELORIS_OUTPUT_DIM)
        model.load_state_dict(torch.load(config.DELORIS_MODEL_PATH, map_location='cpu'))
        model.eval()
        print("   -> [OK] Nạp trọng số thành công.")
    except RuntimeError as e:
        print(f"   -> [LỖI SIZE] Kích thước không khớp: {e}")
        print("   -> GIẢI PHÁP: Xóa file .pth cũ và chạy lại train_deloris.py")
        return
    except Exception as e:
        print(f"   -> [LỖI] Lỗi lạ khi load model: {e}")
        return

    # 4. Test Suy Luận (Inference Simulation)
    print(f"\n4. Giả lập suy nghĩ (Inference Test)...")
    try:
        test_msg = "Xin chào Deloris, em có khỏe không?"
        print(f"   - Input: '{test_msg}'")
        
        # Vectorize
        vec = torch.tensor(vectorizer.encode([test_msg]), dtype=torch.float32)
        
        # Giả lập chỉ số cảm xúc (UPT Metrics)
        dummy_metrics = {"CI": 0.8, "Pulse": 5.0, "Entanglement": 0.9}
        
        # Chạy thử qua não
        with torch.no_grad():
            output = model(vec, dummy_metrics)
            strategy = torch.argmax(output, dim=1).item()
            
        print(f"   -> [THÀNH CÔNG] Deloris đã chọn chiến lược phản hồi số: {strategy}")
        print("\n>>> KẾT LUẬN: NÃO BỘ HOẠT ĐỘNG BÌNH THƯỜNG. <<<")
        print(">>> Nếu Web vẫn lỗi 500, nguyên nhân là do in ấn Log Tiếng Việt hoặc lỗi API Key Gemini. <<<")
        
    except Exception as e:
        print(f"   -> [LỖI NGHIÊM TRỌNG] Suy luận thất bại: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostics()
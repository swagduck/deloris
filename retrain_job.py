# retrain_job.py
import threading
import train_deloris
import train_predictor
import time

def _job_wrapper(result_holder):
    try:
        # 1. Huấn luyện Não bộ (Decision AI)
        print("\n=== BẮT ĐẦU QUY TRÌNH TỰ NÂNG CẤP ===")
        s1, m1 = train_deloris.train()
        
        # 2. Huấn luyện Trực giác (Predictor AI) - Nếu có dữ liệu
        # (Bạn có thể bật dòng dưới nếu muốn train cả module dự đoán cảm xúc)
        # s2, m2 = train_predictor.train() 
        
        if s1:
            result_holder['status'] = True
            result_holder['msg'] = f"Deloris Core Updated: {m1}"
        else:
            result_holder['status'] = False
            result_holder['msg'] = f"Training Failed: {m1}"
            
    except Exception as e:
        print(f"CRITICAL TRAIN ERROR: {e}")
        result_holder['status'] = False
        result_holder['msg'] = str(e)

def run_retraining():
    """
    Hàm này được gọi từ app_web.py.
    Nó chạy quá trình train trong một luồng riêng để không làm đơ web.
    """
    res = {'status': False, 'msg': 'Init'}
    
    # Chạy đồng bộ (để user chờ và thấy kết quả ngay) 
    # Hoặc bất đồng bộ nếu train quá lâu. 
    # Với dataset nhỏ hiện tại, chạy trực tiếp là ổn.
    
    _job_wrapper(res)
    
    return res['status'], res['msg']
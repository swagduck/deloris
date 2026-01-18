import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- CẤU HÌNH LÒ PHẢN ỨNG ---
C_GEO_GOLDEN = 0.911    
TAU_ION_GOLDEN = 0.080  

class UPTReactor:
    def __init__(self, c_geo, tau_ion):
        self.c_geo = c_geo
        self.tau_ion = tau_ion
        self.time_steps = 20 # Giảm xuống để demo nhanh hơn
        self.plasma_state = []
        
    def run_simulation(self):
        print(f"--- KHỞI ĐỘNG LÒ PHẢN ỨNG UPT-RC ---", flush=True)
        print(f"Cấu hình: C_geo={self.c_geo}, Tau={self.tau_ion}", flush=True)
        
        current_stability = 0.5 
        
        for t in range(self.time_steps):
            time.sleep(0.2) # Giả lập tính toán nặng (Deloris sẽ đọc log trong lúc này)
            
            # Logic ảo
            rf_signal = (1.0 * self.c_geo) % self.tau_ion
            current_stability += 0.02
            current_stability = min(current_stability, 1.0)
            
            self.plasma_state.append(current_stability)
            
            # LOG STREAMING: Dòng này sẽ hiện ngay lập tức trên web
            print(f"Step {t+1}/{self.time_steps}: Stability = {current_stability:.2f} | Pulse Active...", flush=True)

        return self.plasma_state

    def generate_report(self):
        print(f"--- ĐANG TẠO BÁO CÁO ---", flush=True)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(10, 5))
        plt.plot(self.plasma_state, color='#00f3ff')
        plt.title('UPT Reactor Stability')
        plt.savefig("reactor_status.png")
        print(f"Đã xuất biểu đồ: reactor_status.png", flush=True)
        print(f"MÔ PHỎNG HOÀN TẤT.", flush=True)

if __name__ == "__main__":
    reactor = UPTReactor(C_GEO_GOLDEN, TAU_ION_GOLDEN)
    reactor.run_simulation()
    reactor.generate_report()
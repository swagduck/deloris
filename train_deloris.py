# train_deloris.py
# [PHIÊN BẢN: v2.0 - UPT AWARENESS TRAINING]
# Hỗ trợ huấn luyện đa phương thức: Text Vector + Cảm xúc (UPT Context)

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from deloris_ai.architecture import DelorisModel
import config

# Cấu hình
DATA_PATH = 'data/training_dataset.json'
MODEL_SAVE_PATH = config.DELORIS_MODEL_PATH
EPOCHS = 50
LEARNING_RATE = 0.001

def train():
    print(">>> [TRAIN] Đang khởi động Lò luyện Neural UPT...")
    
    # 1. Load Dữ liệu
    if not os.path.exists(DATA_PATH):
        print(f"❌ Không tìm thấy dữ liệu: {DATA_PATH}")
        return False, "Data not found"

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("❌ Dữ liệu trống.")
        return False, "Empty data"

    print(f"   -> Đã tải {len(data)} mẫu ký ức.")

    # 2. Chuẩn bị Vectorizer (Language Core)
    print("   -> Đang tải Language Model (all-MiniLM-L6-v2)...")
    vectorizer = SentenceTransformer(config.LANGUAGE_MODEL_NAME)
    
    # 3. Chuẩn bị Training Tensors
    X_text = []      # Input Text Vectors
    X_upt_list = []  # Input UPT States [CI, Pulse, Entanglement]
    y_labels = []    # Output Class (Strategy ID) -> Bạn cần gán nhãn hoặc tự học
    
    # [QUAN TRỌNG] Mapping chiến lược phản hồi
    # Vì dữ liệu raw chỉ có text output, ta cần một heuristic đơn giản để gán nhãn chiến lược
    # (Trong tương lai, Architect nên gán nhãn thủ công hoặc dùng AI teacher)
    # Tạm thời: 0=Logic, 1=Empathy, 2=Creative, 3=Passive
    
    for item in data:
        # a. Vectorize Text Input
        vec = vectorizer.encode(item['input'])
        X_text.append(vec)
        
        # b. Parse UPT Context (Cảm xúc lúc đó)
        ctx = item.get('upt_context', {})
        # Chuẩn hóa về list [CI, Pulse, Entanglement]
        upt_vec = [
            ctx.get('C', 0.5),      # CI (dùng C làm proxy cho CI nếu thiếu)
            ctx.get('E', 0.0),      # Pulse (dùng E làm proxy)
            0.5                     # Entanglement (Mặc định nếu thiếu)
        ]
        X_upt_list.append(upt_vec)
        
        # c. Heuristic Labeling (Tự động đoán chiến lược dựa trên độ dài output)
        # Đây là bước sơ khai để AI tự học cách chọn style
        out_len = len(item['output'].split())
        if out_len < 10: label = 3 # Ngắn -> Passive
        elif "?" in item['output']: label = 2 # Hỏi -> Creative
        elif "cảm" in item['output'] or "yêu" in item['output']: label = 1 # Cảm xúc -> Empathy
        else: label = 0 # Còn lại -> Logic
        
        y_labels.append(label)

    # Chuyển sang Tensor
    X_text_tensor = torch.tensor(np.array(X_text), dtype=torch.float32)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)
    
    # 4. Khởi tạo Model
    model = DelorisModel(config.INPUT_DIM, config.DELORIS_HIDDEN_DIM, config.DELORIS_OUTPUT_DIM)
    
    # Load weights cũ nếu có (Transfer Learning)
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH))
            print("   -> Đã nạp kiến thức cũ (Fine-tuning).")
        except:
            print("   -> Kiến trúc thay đổi, huấn luyện lại từ đầu.")
            
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    model.train()
    print(f"   -> Bắt đầu nung chảy kiến thức ({EPOCHS} epochs)...")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Forward pass đặc biệt: Cần truyền cả Text Vector và UPT Metrics
        # Vì model nhận dict upt_metrics từng cái, ta cần sửa nhẹ logic batch
        # Cách đơn giản nhất: Loop qua batch (Hơi chậm nhưng an toàn cho cấu trúc hiện tại)
        
        batch_loss = 0
        for i in range(len(X_text_tensor)):
            # Tái tạo dict metrics cho từng mẫu
            sample_metrics = {
                "CI": X_upt_list[i][0],
                "Pulse": X_upt_list[i][1],
                "Entanglement": X_upt_list[i][2]
            }
            
            # Input cần shape (1, dim)
            inp = X_text_tensor[i].unsqueeze(0) 
            target = y_tensor[i].unsqueeze(0)
            
            output = model(inp, sample_metrics)
            loss = criterion(output, target)
            loss.backward()
            batch_loss += loss.item()
        
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{EPOCHS}, Loss: {batch_loss/len(X_text_tensor):.4f}")

    # 6. Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ [DONE] Đã lưu não bộ mới tại: {MODEL_SAVE_PATH}")
    return True, f"Success. Loss: {batch_loss/len(X_text_tensor):.4f}"

if __name__ == "__main__":
    train()
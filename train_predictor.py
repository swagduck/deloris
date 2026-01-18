import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import config # Import config
from upt_predictor.architecture import UPTAutomatorModel # Import kiến trúc V4

# --- [Cấu hình Huấn luyện] ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- [Huấn luyện AI Cảm nhận (V4 - Đa phương thức)] ---")
print(f"Đang sử dụng thiết bị: {DEVICE}")

# Tải các hằng số từ config
DATASET_PATH = config.PREDICTOR_DATASET_PATH
MODEL_SAVE_PATH = config.AUTOMATOR_MODEL_PATH
TEXT_INPUT_DIM = config.PREDICTOR_INPUT_DIM # 774 (theo config mới)
IMAGE_VECTOR_DIM = config.IMAGE_VECTOR_DIM   # 512
HIDDEN_DIM = config.AUTOMATOR_HIDDEN_DIM     # 512 (theo config mới)
LANGUAGE_MODEL = config.LANGUAGE_MODEL_NAME
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# [MỚI - NÂNG CẤP 3] Đường dẫn tệp feedback
FEEDBACK_DATA_PATH = config.ACTIVE_FEEDBACK_FILE

# Tham số huấn luyện
NUM_EPOCHS = config.AUTOMATOR_EPOCHS # Lấy từ config (ví dụ: 150)
BATCH_SIZE = config.AUTOMATOR_BATCH_SIZE # Lấy từ config (ví dụ: 16)
LEARNING_RATE = config.AUTOMATOR_LR # Lấy từ config (ví dụ: 0.001)

# --- [Lớp Dataset Đa phương thức] ---

class MultimodalUPTDataset(Dataset):
    # Sửa đổi init để nhận một danh sách dữ liệu, thay vì đường dẫn tệp
    def __init__(self, data_list, vectorizer, clip_processor, clip_model, dummy_image_vec):
        print(f"Đang khởi tạo Dataset...")
        self.data = data_list
        self.vectorizer = vectorizer
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.dummy_image_vec = dummy_image_vec
        print(f"Đã nạp {len(self.data)} mẫu dữ liệu (đã gộp).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. Xử lý Văn bản + Trạng thái (luồng 1)
        text = item.get("text", "")
        text_vector = torch.tensor(self.vectorizer.encode([text]), dtype=torch.float32)
        
        # Lấy trạng thái cũ (AEC và UPT)
        # Sử dụng các giá trị mặc định từ config V4
        prev_a = item.get("last_A", 0.5)
        prev_e = item.get("last_E", 1.0)
        prev_c = item.get("last_C", 1.0)
        
        prev_ci = item.get("last_CI", 0.5)
        prev_pulse = item.get("last_Pulse", 0.0)
        prev_ent = item.get("last_Entanglement", 0.5)

        prev_aec_tensor = torch.tensor([[prev_a, prev_e, prev_c]], dtype=torch.float32)
        prev_upt_tensor = torch.tensor([[prev_ci, prev_pulse, prev_ent]], dtype=torch.float32)
        
        # Gộp lại thành vector đầu vào 774-dim (768 + 3 + 3)
        textual_input = torch.cat((text_vector, prev_aec_tensor, prev_upt_tensor), dim=1).squeeze(0) # Squeeze để bỏ batch dim (1, 774) -> (774)

        # 2. Xử lý Hình ảnh (luồng 2)
        image_path = item.get("image_path")
        visual_input = self.dummy_image_vec.squeeze(0) # Squeeze để bỏ batch dim (1, 512) -> (512)
        
        if image_path and image_path != "none" and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                with torch.no_grad():
                    inputs = self.clip_processor(images=image, return_tensors="pt", padding=True).to(DEVICE)
                    image_features = self.clip_model.get_image_features(**inputs)
                    visual_input = image_features.cpu().detach().to(dtype=torch.float32).squeeze(0)
            except Exception as e:
                print(f"Cảnh báo: Không thể tải ảnh '{image_path}': {e}. Sử dụng ảnh rỗng.")
        
        # 3. Xử lý Mục tiêu (Targets)
        target_A = torch.tensor([item.get("target_A", 0.5)], dtype=torch.float32)
        target_E = torch.tensor([item.get("target_E", 0.5)], dtype=torch.float32)
        target_C = torch.tensor([item.get("target_C", 0.5)], dtype=torch.float32)

        return textual_input, visual_input, target_A, target_E, target_C

# --- [Hàm Huấn luyện Chính] ---

def train():
    # 1. Tải các mô hình tiền huấn luyện
    print("Đang tải các mô hình nền (SentenceTransformer và CLIP)...")
    vectorizer = SentenceTransformer(LANGUAGE_MODEL)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip_model.eval() # Đặt ở chế độ eval vì chúng ta chỉ dùng nó để trích xuất đặc trưng
    
    # Tạo vector "rỗng"
    dummy_image_vec = torch.zeros(1, IMAGE_VECTOR_DIM, dtype=torch.float32)
    print("Tải mô hình nền hoàn tất.")

    # 2. [NÂNG CẤP 3] Nạp và Gộp Dữ liệu
    print(f"Đang nạp dữ liệu gốc từ: {DATASET_PATH}")
    original_data = []
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        print(f"Đã nạp {len(original_data)} bản ghi gốc.")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp dữ liệu gốc '{DATASET_PATH}'.")
        exit()
    except json.JSONDecodeError:
        print(f"LỖI: Tệp dữ liệu gốc '{DATASET_PATH}' không phải là JSON hợp lệ.")
        exit()
        
    print(f"Đang nạp dữ liệu feedback từ: {FEEDBACK_DATA_PATH}")
    new_feedback_data = []
    if os.path.exists(FEEDBACK_DATA_PATH):
        try:
            with open(FEEDBACK_DATA_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        new_feedback_data.append(json.loads(line))
            print(f"Đã nạp {len(new_feedback_data)} bản ghi feedback mới.")
        except Exception as e:
            print(f"Cảnh báo: Lỗi khi đọc file feedback '{FEEDBACK_DATA_PATH}': {e}")
    else:
        print("Không tìm thấy tệp feedback (đây là điều bình thường nếu mới chạy lần đầu).")

    # Gộp cả hai lại
    combined_data = original_data + new_feedback_data
    if len(combined_data) == 0:
        print("LỖI: Không có dữ liệu để huấn luyện.")
        exit()
        
    print(f"Tổng cộng {len(combined_data)} bản ghi sẽ được dùng để huấn luyện.")


    # 3. Tạo Dataset và DataLoader
    # Sử dụng combined_data
    dataset = MultimodalUPTDataset(combined_data, vectorizer, clip_processor, clip_model, dummy_image_vec)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Khởi tạo mô hình V4
    model = UPTAutomatorModel(
        text_input_dim=TEXT_INPUT_DIM,
        image_input_dim=IMAGE_VECTOR_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)
    
    # 5. Thiết lập Loss và Optimizer
    criterion = nn.MSELoss() # Mean Squared Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Vòng lặp Huấn luyện
    print(f"--- Bắt đầu huấn luyện {NUM_EPOCHS} epochs ---")
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        model.train() # Đặt mô hình ở chế độ train
        
        for i, (text_batch, image_batch, a_batch, e_batch, c_batch) in enumerate(data_loader):
            # Chuyển dữ liệu lên DEVICE
            text_batch = text_batch.to(DEVICE)
            image_batch = image_batch.to(DEVICE)
            a_batch = a_batch.to(DEVICE)
            e_batch = e_batch.to(DEVICE)
            c_batch = c_batch.to(DEVICE)
            
            # Xóa gradient cũ
            optimizer.zero_grad()
            
            # Forward pass (gửi cả 2 luồng vào)
            out_A, out_E, out_C = model(text_batch, image_batch)
            
            # Tính loss
            loss_a = criterion(out_A, a_batch)
            loss_e = criterion(out_E, e_batch)
            loss_c = criterion(out_C, c_batch)
            loss = loss_a + loss_e + loss_c # Tổng loss
            
            # Backward pass và tối ưu
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(data_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

    # 7. Lưu mô hình
    print("--- Huấn luyện hoàn tất ---")
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Đã lưu mô hình đã huấn luyện vào: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"LỖI: Không thể lưu mô hình: {e}")

if __name__ == "__main__":
    train()
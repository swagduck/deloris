# Deloris_Newage/deloris_upt_project/upt_predictor/architecture.py
import torch
import torch.nn as nn

class UPTAutomatorModel(nn.Module):
    """
    (NÂNG CẤP ĐA PHƯƠNG THỨC - V4)
    Kiến trúc AI Cảm nhận (UPT Automator) giờ đây chấp nhận hai đầu vào:
    1. x_textual (Tensor): Vector (Văn bản + Trạng thái UPT cũ)
    2. x_visual (Tensor): Vector (Hình ảnh từ CLIP)
    
    Mô hình sẽ học cách kết hợp cả hai để dự đoán A, E, C.
    """
    
    # [NÂNG CẤP ĐA PHƯƠNG THỨC]
    # 'hidden_dim' giờ đây là tổng hidden_dim của cả hai nhánh
    def __init__(self, text_input_dim, image_input_dim, hidden_dim):
        super(UPTAutomatorModel, self).__init__()
        
        # Nhánh 1: Đường xử lý văn bản (Text path)
        # (Giảm hidden_dim của nhánh này để chừa chỗ cho nhánh hình ảnh)
        self.fc1_text = nn.Linear(text_input_dim, hidden_dim // 2)
        
        # Nhánh 2: Đường xử lý hình ảnh (Image path)
        self.fc1_image = nn.Linear(image_input_dim, hidden_dim // 2)

        # Các lớp chung
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Lớp kết hợp (Fusion layer)
        # Kích thước đầu vào là tổng của hai nhánh: (hidden_dim // 2) + (hidden_dim // 2) = hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2) 
        
        # Các lớp đầu ra (giữ nguyên như kiến trúc cũ)
        self.fc3_A = nn.Linear(hidden_dim // 2, 1)
        self.fc3_E = nn.Linear(hidden_dim // 2, 1)
        self.fc3_C = nn.Linear(hidden_dim // 2, 1)
        
        self.sigmoid = nn.Sigmoid()

    # [NÂNG CẤP ĐA PHƯƠNG THỨC]
    # Hàm forward giờ đây nhận 2 đối số
    def forward(self, x_textual, x_visual):
        
        # 1. Xử lý hai luồng riêng biệt
        x_text = self.dropout(self.relu(self.fc1_text(x_textual)))
        x_image = self.dropout(self.relu(self.fc1_image(x_visual)))
        
        # 2. Kết hợp (Fuse) hai luồng lại
        # (Nối 2 tensor lại với nhau)
        x_fused = torch.cat((x_text, x_image), dim=1)
        
        # 3. Đưa qua các lớp chung
        x = self.dropout(self.relu(self.fc2(x_fused)))
        
        # 4. Tính toán đầu ra (giống hệt như cũ)
        out_A = self.sigmoid(self.fc3_A(x))
        out_E = self.sigmoid(self.fc3_E(x))
        out_C = self.sigmoid(self.fc3_C(x))
        
        # Trả về 3 giá trị riêng biệt để tương thích với code huấn luyện
        return out_A, out_E, out_C
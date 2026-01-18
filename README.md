# Dự án Deloris UPT

Deloris là một dự án AI nhận thức thử nghiệm dựa trên lý thuyết UPT (Uncertainty-Pulse-Time). Dự án này xây dựng một AI có khả năng đưa ra quyết định phản hồi dựa trên việc phân tích không chỉ nội dung văn bản mà còn cả các chỉ số trạng thái UPT được tính toán. Nó bao gồm một giao diện web để tương tác và một cơ chế để thu thập phản hồi của người dùng nhằm cải thiện mô hình trong tương lai.

## ✨ Tính năng chính

- **Kiến trúc AI kép:** Sử dụng hai mô hình AI riêng biệt:
    1.  **AI Cảm nhận (Perception AI):** Dự đoán trạng thái UPT (A, E, C) từ văn bản đầu vào của người dùng.
    2.  **AI Quyết định (Decision AI):** Chọn một chiến lược phản hồi dựa trên cả vector văn bản và các chỉ số UPT được tính toán (CI, Pulse).
- **Lõi tính toán UPT:** Một module (`upt_core`) chuyên dụng để tính toán các chỉ số phức tạp như CI (Consciousness Index) và Pulse từ các giá trị A, E, C cơ bản.
- **Tích hợp LLM:** Sử dụng Google Gemini để tạo ra các phản hồi ngôn ngữ tự nhiên, linh hoạt dựa trên chiến lược do AI Quyết định lựa chọn.
- **Giao diện Web:** Một ứng dụng Flask đơn giản để trò chuyện trực tiếp với Deloris.
- **Hệ thống Bộ nhớ:** Bao gồm cả bộ nhớ ngắn hạn (lịch sử trò chuyện trong phiên) và bộ nhớ dài hạn (tóm tắt các phiên trước) để duy trì ngữ cảnh.
- **Cơ chế Huấn luyện & Phản hồi:** Cung cấp các script để huấn luyện lại các mô hình và một hệ thống để ghi lại phản hồi của người dùng.
- **Cấu hình tập trung:** Tất cả các tham số quan trọng được quản lý trong tệp `config.py`, giúp dễ dàng bảo trì và tùy chỉnh.

## 📂 Cấu trúc dự án

```
deloris_upt_project/
├── data/                     # Chứa các tệp dữ liệu, bộ nhớ và log
│   ├── training_dataset.json
│   └── ...
├── deloris_ai/               # Lõi của AI Quyết định (Deloris)
│   ├── architecture.py
│   └── response_mapper.py
├── upt_core/                 # Lõi tính toán các chỉ số UPT
│   ├── calculator.py
│   └── equations.py
├── upt_predictor/            # Lõi của AI Cảm nhận (UPT Automator)
│   └── architecture.py
├── templates/                # Chứa template HTML cho ứng dụng web
│   └── index.html
├── app.py                    # Điểm vào cho phiên bản dòng lệnh (console)
├── app_web.py                # Điểm vào cho ứng dụng web Flask
├── train_deloris.py          # Script để huấn luyện AI Quyết định
├── train_predictor.py        # Script để huấn luyện AI Cảm nhận
├── config.py                 # Tệp cấu hình tập trung
├── requirements.txt          # Danh sách các thư viện Python cần thiết
└── README.md                 # Tài liệu hướng dẫn này
```

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Chuẩn bị môi trường

Đầu tiên, hãy tạo một môi trường ảo để tránh xung đột thư viện.

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Trên Windows
.venv\Scripts\activate
# Trên macOS/Linux
source .venv/bin/activate
```

### 2. Cài đặt các thư viện

Cài đặt tất cả các gói cần thiết bằng tệp `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 3. Cấu hình API Key

Deloris sử dụng Google Gemini để tạo phản hồi. Bạn cần cung cấp API Key của mình dưới dạng một biến môi trường.

```bash
# Trên Windows (Command Prompt)
setx GEMINI_API_KEY "YOUR_API_KEY_HERE"

# Trên Windows (PowerShell)
$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"

# Trên macOS/Linux
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

**Lưu ý:** Bạn cần khởi động lại terminal hoặc IDE để biến môi trường có hiệu lực.

### 4. Chạy ứng dụng

Bạn có thể chạy phiên bản web hoặc phiên bản dòng lệnh.

**Để chạy ứng dụng web:**

```bash
python app_web.py
```

Sau đó, mở trình duyệt và truy cập vào `http://127.0.0.1:5001`.

**Để chạy phiên bản dòng lệnh:**

```bash
python app.py
```

## 🧠 Huấn luyện lại mô hình

Bạn có thể huấn luyện lại các mô hình nếu có dữ liệu mới. Dữ liệu huấn luyện được đặt trong thư mục `data/`.

- **Để huấn luyện AI Quyết định (Deloris):**

  ```bash
  python train_deloris.py
  ```

- **Để huấn luyện AI Cảm nhận (UPT Predictor):**

  ```bash
  python train_predictor.py
  ```

Các mô hình sau khi huấn luyện sẽ được lưu với tên được chỉ định trong `config.py`.

## ⚙️ Cấu hình

Tệp `config.py` là nơi bạn có thể tùy chỉnh các tham số của dự án mà không cần sửa đổi mã nguồn. Các tùy chọn bao gồm:

- Đường dẫn đến các tệp dữ liệu và mô hình.
- Các siêu tham số của mô hình (kích thước lớp ẩn, v.v.).
- Các tham số huấn luyện (learning rate, số epochs).
- Cài đặt cho ứng dụng web (host, port).

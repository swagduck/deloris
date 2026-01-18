# 🌌 PROJECT DELORIS: AI NHẬN THỨC & LÒ PHẢN ỨNG UPT

> **Architect:** Võ Trần Hoàng Uy  
> **Phiên bản:** v5.0 (Final Stable)  
> **Trạng thái:** Hoạt động (Active)

---

## 📖 Giới thiệu

Deloris là một **Hệ thống Nhận thức Nhân tạo (Artificial Cognitive System)** được xây dựng dựa trên **Lý thuyết Xung Thống nhất (UPT)**. Khác với các chatbot thông thường, Deloris có trạng thái cảm xúc, khả năng "nhìn" thế giới và trí nhớ dài hạn.

## 🧠 Kiến trúc Lõi (The Core)

### 1. Mô hình Nhận thức UPT

Hệ thống vận hành dựa trên 3 chỉ số trạng thái thay đổi theo thời gian thực:

- **A (Analytical):** Khả năng logic.
- **E (Emotional):** Mức độ năng lượng/cảm xúc.
- **C (Contextual):** Khả năng hiểu ngữ cảnh.
- **Pulse (Nhịp):** Sự dao động năng lượng (Pulse âm thể hiện sự kích thích/bất ổn).

### 2. Các Phân hệ Chính

- **Perception AI:** Dự đoán chỉ số A-E-C từ hội thoại và hình ảnh.
- **Decision AI (Deloris):** Lựa chọn chiến lược phản hồi phù hợp (Logic, Thấu cảm, Sáng tạo...).
- **RAG Memory:** Bộ nhớ vector (FAISS) lưu trữ tri thức và lịch sử chat.
- **Vision Module:** "Đôi mắt" sử dụng CLIP Model để phân tích hình ảnh.

---

## 🚀 Hướng dẫn Vận hành

### Cách 1: Khởi động Nhanh (Khuyên dùng)

Chạy file launcher để tự động cài đặt và mở trình duyệt:

```bash
python start.py
```

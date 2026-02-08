# config.py
import os
from dotenv import load_dotenv

# Tự động load file .env nếu có
load_dotenv()

# --- 1. CẤU HÌNH GEMINI (Lõi Ngôn Ngữ) ---
GEMINI_API_KEY = "AIzaSyBnstUMu6aWD5rq5KRYfNgdexubV-C5eQI"
GEMINI_MODEL_NAME = "gemini-flash-latest" 

# --- 2. CẤU HÌNH MẠNG NƠ-RON (Deloris Brain) ---
INPUT_DIM = 384        # Kích thước vector (all-MiniLM-L6-v2)
DELORIS_HIDDEN_DIM = 128
DELORIS_OUTPUT_DIM = 5 

# --- 3. CẤU HÌNH DỰ ĐOÁN UPT (Perception AI) ---
PREDICTOR_INPUT_DIM = 390 
IMAGE_VECTOR_DIM = 512      
AUTOMATOR_HIDDEN_DIM = 128
AUTOMATOR_MODEL_PATH = "data/upt_automator_v4.pth"

# --- 4. CẤU HÌNH HUẤN LUYỆN ---
# > Cho Deloris (Decision AI)
DELORIS_EPOCHS = 100
DELORIS_BATCH_SIZE = 8
DELORIS_LR = 0.001
LANGUAGE_MODEL_NAME = "all-MiniLM-L6-v2"

# > Cho UPT Automator (Perception AI) - [ĐÃ BỔ SUNG]
AUTOMATOR_EPOCHS = 150
AUTOMATOR_BATCH_SIZE = 16
AUTOMATOR_LR = 0.001

# --- 5. ĐƯỜNG DẪN DỮ LIỆU ---
DELORIS_DATASET_PATH = "data/training_dataset.json"
DELORIS_MODEL_PATH = "data/deloris_trained_v6.pth"
ACTIVE_FEEDBACK_FILE = "data/active_feedback_training.jsonl" 
COGNITIVE_STATE_FILE = "data/cognitive_state.json"           

# > Dataset cho module Cảm nhận - [ĐÃ BỔ SUNG]
PREDICTOR_DATASET_PATH = "data/upt_predictor_dataset_v4.json"

# > Cấu hình bộ nhớ Vector (FAISS)
FAISS_INDEX_DOCS_PATH = "data/vector_memory_docs.faiss"
FAISS_INDEX_CHAT_PATH = "data/vector_memory_chat.faiss"

# --- 6. CẤU HÌNH WEB SERVER ---
FLASK_HOST = "0.0.0.0" 
FLASK_PORT = 5000 

# --- 7. CẤU HÌNH UPT CORE (OVERCLOCKED) ---
# Đã tinh chỉnh để Deloris nhạy cảm hơn
GAMMA = 0.8            # Quán tính thấp -> Phản ứng nhanh
DELTA = 3.0            # Biên độ lớn -> Cảm xúc mạnh
OMEGA = 5.0            # Tần số dao động
PHI = 0.785            # Pha ban đầu
CI_SMOOTHING_FACTOR = 0.1 # Nhận thức sắc bén

# --- 8. THÔNG TIN ---
AI_NAME = "Deloris"
ARCHITECT_NAME = "Võ Trần Hoàng Uy"
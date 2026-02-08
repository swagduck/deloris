# app_web_optimized.py
# [PHIÊN BẢN: v10.0 - OPTIMIZED WEB3 AGENT]
# Tích hợp: Async Processing, Vector Memory, RLHF, Local SLM Support

import os
import sys
import json
import time
import glob
import queue
import shutil
import signal
import uuid
import threading
import subprocess
import traceback
import asyncio
import edge_tts 
from collections import deque
import re

# [FIX] Ép mã hóa UTF-8 cho Terminal
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai

import config
from file_processor import FileProcessor
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer

# --- CORE MODULES ---
from deloris_ai.architecture import DelorisModel
from upt_core.calculator import UPTCalculator
from deloris_ai.response_mapper import generate_final_response
from upt_predictor.compatibility import UPTAutomatorModelCompat  # Use compatibility layer
from upt_core.prediction_error import PredictionErrorSystem
from deloris_ai.inner_monologue_optimized import InnerMonologueSystemOptimized  # Use optimized version
from deloris_ai.vector_memory import VectorMemorySystem  # Add vector memory
from deloris_ai.rlhf_collector import RLHFDataCollector  # Add RLHF collection

# --- AI MODULES (FULL SUITE) ---
from upt_core.safety import SuperegoMonitor
from deloris_ai.plasticity import PlasticityLayer
from deloris_ai.dreaming import DreamWeaver
from deloris_ai.artist import generate_image, detect_art_intent
from deloris_ai.heartbeat import HeartbeatSystem
from deloris_ai.vision import deloris_eye
from deloris_ai.motor import MotorSystem
from deloris_ai.coder import NeuralCoder
from deloris_ai.wallet import CryptoWallet  # [MỚI] Module Web3

app = Flask(__name__)
CORS(app)

# --- CONFIG WAKE WORDS ---
WAKE_WORDS = [
    "deloris", "em ơi", "ê robot", "trợ lý", "này", "alo", "ơi", "bạn ơi",
    "chào", "hello", "hi", "good morning", "chúc ngủ ngon",
    "giúp", "cho hỏi", "tại sao", "làm sao", "cách nào", "là gì",
    "vẽ", "hát", "tìm", "bật", "tắt", "kể chuyện", "nhìn", "xem",
    "hay quá", "đẹp quá", "buồn", "vui", "chán", "chụp màn hình", "mở nhạc",
    "viết code", "lập trình", "tạo script", "số dư", "gửi tiền", "ví"
]

# --- GLOBAL VARIABLES ---
vectorizer = None
deloris_model = None
predictor_model = None
upt_calculator = None
text_splitter = None
vector_store_docs = None
vector_store_chat = None
embeddings_model = None
clip_processor = None
clip_model = None
dummy_image_vector = None

# [BIẾN TRẠNG THÁI]
LATEST_VISUAL_CONTEXT = "" 
BACKGROUND_TASK_STATUS = {"status": "idle", "task": "Không có"}
GLOBAL_NOTIFICATIONS = deque(maxlen=5)
SYSTEM_ACTIVE = True
LOG_QUEUE = queue.Queue()

# [AI INSTANCES]
superego = None
plasticity = None
dreaming = None
inner_monologue = None
prediction_error = None
heartbeat = None
vision = None
motor = None
coder = None
wallet = None
vector_memory = None  # Add vector memory system
rlhf_collector = None  # Add RLHF collector

def denormalize_predictions(preds_tensor):
    # Handle both concatenated and separate tensor formats
    if preds_tensor.dim() == 2 and preds_tensor.shape[1] == 3:
        # 2D tensor format [batch, 3]
        A_norm, E_norm, C_norm = preds_tensor[0, 0], preds_tensor[0, 1], preds_tensor[0, 2]
    elif preds_tensor.dim() == 1 and preds_tensor.shape[0] == 3:
        # 1D tensor format [3]
        A_norm, E_norm, C_norm = preds_tensor[0], preds_tensor[1], preds_tensor[2]
    else:
        # Separate format (original)
        A_norm, E_norm, C_norm = preds_tensor[0], preds_tensor[1], preds_tensor[2]
    
    A_t = max(A_norm.item() * 1.0, 0.1)
    E_t = max(E_norm.item() * 5.0, 0.1)
    C_t = max(C_norm.item() * 3.0, 0.1)
    return A_t, E_t, C_t

# --- ASYNC PROCESSING FUNCTIONS ---
async def process_upt_prediction_async(input_vector):
    """Async UPT prediction"""
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        normalized_preds = await loop.run_in_executor(
            None, lambda: predictor_model(input_vector).squeeze()
        )
        A_t, E_t, C_t = denormalize_predictions(normalized_preds)
        return A_t, E_t, C_t

async def process_sentiment_prediction_async(text_input, upt_metrics, chat_history):
    """Async sentiment prediction"""
    loop = asyncio.get_event_loop()
    predicted_sentiment, confidence = await loop.run_in_executor(
        None, lambda: prediction_error.predict_user_response(text_input, upt_metrics, chat_history)
    )
    return predicted_sentiment, confidence

async def process_deloris_classification_async(input_vector, upt_metrics):
    """Async Deloris classification"""
    loop = asyncio.get_event_loop()
    with torch.no_grad():
        prediction = await loop.run_in_executor(
            None, lambda: deloris(input_vector, upt_metrics)
        )
        predicted_class = torch.argmax(prediction, dim=1).item()
    return predicted_class

async def process_inner_thought_async(text_input, upt_metrics, chat_history, global_state):
    """Async inner thought generation"""
    loop = asyncio.get_event_loop()
    inner_thought = await loop.run_in_executor(
        None, lambda: inner_monologue.generate_inner_thought(text_input, upt_metrics, chat_history, global_state)
    )
    return inner_thought

async def process_response_from_thought_async(inner_thought, text_input, upt_metrics, chat_history):
    """Async response generation from thought"""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: inner_monologue.generate_response_from_thought(inner_thought, text_input, upt_metrics, chat_history)
    )
    return response

# --- OPTIMIZED MAIN PROCESSING FUNCTION ---
async def process_user_input_optimized(text_input):
    """Optimized async processing with parallel execution"""
    global upt_calculator, vector_memory, rlhf_collector
    
    # Prepare inputs
    input_vector = torch.tensor(vectorizer.encode([text_input]), dtype=torch.float32)
    
    # Pad or truncate to match expected input dimension (774)
    if input_vector.shape[1] < 774:
        padding = torch.zeros(1, 774 - input_vector.shape[1])
        input_vector = torch.cat([input_vector, padding], dim=1)
    elif input_vector.shape[1] > 774:
        input_vector = input_vector[:, :774]

    # Use original 384-dim input for Deloris model
    deloris_input = torch.tensor(vectorizer.encode([text_input]), dtype=torch.float32)

    # PARALLEL PROCESSING: Run UPT prediction and sentiment prediction simultaneously
    print("🚀 Đang xử lý song song...")
    start_time = asyncio.get_event_loop().time()
    
    upt_task = process_upt_prediction_async(input_vector)
    sentiment_task = process_sentiment_prediction_async(text_input, {}, [])
    
    upt_results = await asyncio.gather(upt_task, sentiment_task)
    
    # Unpack the results
    upt_values = upt_results[0]  # (A_t, E_t, C_t)
    sentiment_values = upt_results[1]  # (predicted_sentiment, confidence)
    
    A_t, E_t, C_t = upt_values
    predicted_sentiment, confidence = sentiment_values
    
    processing_time = asyncio.get_event_loop().time() - start_time
    print(f"⚡ Xử lý song song hoàn tất trong {processing_time:.2f}s")
    
    print(f"UPT tự động đo được: A={A_t:.2f}, E={E_t:.2f}, C={C_t:.2f}")
    print(f"[Prediction] Dự đoán User sẽ: {predicted_sentiment} (confidence: {confidence:.2f})")

    upt_metrics = upt_calculator.update_metrics(A_t, E_t, C_t)
    
    # PARALLEL PROCESSING: Inner thought and Deloris classification
    inner_thought_task = process_inner_thought_async(text_input, upt_metrics, [], "")
    deloris_task = process_deloris_classification_async(deloris_input, upt_metrics)
    
    inner_results = await asyncio.gather(inner_thought_task, deloris_task)
    
    # Unpack the results
    inner_thought = inner_results[0]
    predicted_class = inner_results[1]
    
    print(f"[Inner Monologue] Suy nghĩ: '{inner_thought}'")
    
    # Generate response from thought
    response_from_thought = await process_response_from_thought_async(
        inner_thought, text_input, upt_metrics, []
    )
    
    # Retrieve current memory from vector system
    current_memory = await vector_memory.load_memory()
    
    # Retrieve relevant memories using vector search
    relevant_memories = await vector_memory.search_similar(text_input, k=3)
    
    # Generate final response with vector memories
    result = generate_final_response(
        predicted_class, 
        text_input, 
        relevant_memories,  # retrieved_docs (3rd parameter)
        [],  # chat_history (empty for web)
        0.5,  # entanglement_level
        "neutral",  # persona
        response_from_thought,  # global_state
        upt_metrics.get('CI', 0.5),  # CI_value
        None,  # proactive_report
        upt_metrics.get('Pulse', 0.0)  # pulse_value (last parameter)
    )
    
    # Handle the return value properly
    if isinstance(result, tuple) and len(result) == 2:
        final_output_message, clean_response_text = result
    else:
        final_output_message = str(result)
        clean_response_text = str(result)
    
    # RLHF Data Collection
    await rlhf_collector.collect_interaction(
        input_text=text_input,
        inner_thought=inner_thought,
        response=clean_response_text,
        predicted_class=predicted_class,
        upt_metrics=upt_metrics
    )
    
    return final_output_message, clean_response_text, upt_metrics, inner_thought, predicted_class

# --- REST OF THE FUNCTIONS (Keep all existing web functionality) ---
# [All the existing functions from app_web.py would go here]
# This is a placeholder - in a real implementation, you'd copy all the existing functions

# --- MAIN WEB ROUTES ---
@app.route('/chat', methods=['POST'])
def chat():
    """Optimized chat endpoint with async processing"""
    try:
        data = request.get_json()
        text_input = data.get('message', '').strip()
        
        if not text_input:
            return jsonify({'error': 'No message provided'})
        
        # Use optimized async processing
        loop = asyncio.new_event_loop()
        final_output_message, clean_response_text, upt_metrics, inner_thought, predicted_class = loop.run_until_complete(
            process_user_input_optimized(text_input)
        )
        
        response_data = {
            'response': final_output_message,
            'clean_response': clean_response_text,
            'upt_metrics': upt_metrics,
            'inner_thought': inner_thought,
            'predicted_class': predicted_class,
            'processing_time': time.time()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)})

# Keep all other existing routes from app_web.py...
@app.route('/')
def index():
    return render_template('index.html')

# [INITIALIZATION - Copy from original but with optimized modules]
def initialize_systems():
    """Initialize all systems with optimizations"""
    global vectorizer, deloris_model, predictor_model, upt_calculator
    global text_splitter, vector_store_docs, vector_store_chat, embeddings_model
    global clip_processor, clip_model, dummy_image_vector
    global superego, plasticity, dreaming, inner_monologue, prediction_error
    global heartbeat, vision, motor, coder, wallet, vector_memory, rlhf_collector
    
    try:
        print("Đang khởi tạo hệ thống Deloris (Web Optimized)...")
        
        # Configurations are now loaded from config.py
        print(f"Đang tải Bộ não Ngôn ngữ ({config.LANGUAGE_MODEL_NAME})...")
        vectorizer = SentenceTransformer(config.LANGUAGE_MODEL_NAME)
        print("Bộ não Ngôn ngữ: Sẵn sàng.")
        
        try:
            deloris_model = DelorisModel(config.INPUT_DIM, config.DELORIS_HIDDEN_DIM, config.DELORIS_OUTPUT_DIM)
            deloris_model.load_state_dict(torch.load(config.DELORIS_MODEL_PATH))
            deloris_model.eval()
            print(f"Mô hình Deloris ({config.DELORIS_MODEL_PATH}): Sẵn sàng.")
        except FileNotFoundError: 
            return f"LỖI: Không tìm thấy {config.DELORIS_MODEL_PATH}"
        
        try:
            predictor_model = UPTAutomatorModelCompat(
                config.INPUT_DIM, 
                config.IMAGE_VECTOR_DIM, 
                config.AUTOMATOR_HIDDEN_DIM
            )
            predictor_model.load_state_dict(torch.load(config.AUTOMATOR_MODEL_PATH))
            predictor_model.eval()
            print(f"Bộ dự đoán UPT ({config.AUTOMATOR_MODEL_PATH}): Sẵn sàng.")
        except FileNotFoundError: 
            return f"LỖI: Không tìm thấy {config.AUTOMATOR_MODEL_PATH}"
        
        upt_calculator = UPTCalculator(dt=1.0)
        print("Lõi UPT: Sẵn sàng.")
        
        # Initialize optimized systems
        inner_monologue = InnerMonologueSystemOptimized()
        prediction_error = PredictionErrorSystem()
        vector_memory = VectorMemorySystem()
        rlhf_collector = RLHFDataCollector()
        print("Hệ thống Inner Monologue (Tối ưu), Vector Memory, RLHF: Sẵn sàng.")
        
        # Initialize other AI modules (keep existing functionality)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embeddings_model = SentenceTransformerEmbeddings(model_name=config.LANGUAGE_MODEL_NAME)
        
        # Initialize vector stores
        vector_store_docs = FAISS(embeddings_model, "data/vector_memory_docs.faiss")
        vector_store_chat = FAISS(embeddings_model, "data/vector_memory_chat.faiss")
        
        # Load existing memories
        if os.path.exists("data/vector_memory_docs.faiss"):
            try:
                vector_store_docs.load_local("data/vector_memory_docs.faiss")
                vector_store_chat.load_local("data/vector_memory_chat.faiss")
                print("Vector Memory: Loaded existing memories")
            except Exception as e:
                print(f"Vector Memory loading error: {e}")
        
        clip_processor = CLIPProcessor()
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        dummy_image_vector = torch.zeros(512)
        
        # Initialize AI modules
        superego = SuperegoMonitor()
        plasticity = PlasticityLayer()
        dreaming = DreamWeaver()
        inner_monologue = InnerMonologueSystemOptimized()
        prediction_error = PredictionErrorSystem()
        heartbeat = HeartbeatSystem()
        vision = deloris_eye()
        motor = MotorSystem()
        coder = NeuralCoder()
        wallet = CryptoWallet()
        
        print("Tất cả module AI đã được khởi tạo.")
        
        return True
        
    except Exception as e:
        print(f"Lỗi khởi tạo hệ thống: {e}")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if initialize_systems():
        app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=True)

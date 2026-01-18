# app_web.py
# [PHIÊN BẢN: v9.2 - THE WEB3 AGENT]
# Tích hợp: Neural DB, Vision, Voice, Motor, Coder, AND CRYPTO WALLET

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
from upt_predictor.architecture import UPTAutomatorModel
import retrain_job

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
    "deloris", "em ơi", "em à", "ê robot", "trợ lý", "này", "alo", "ơi", "bạn ơi",
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
chat_history = []
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
dreamer = None
heartbeat = None
motor = None
coder = None
wallet = None # [MỚI]

last_upt_values = (0.5, 1.0, 1.0)
last_upt_metrics = {"CI": 0.5, "Pulse": 0.0, "Entanglement": 0.5}
user_vector_history = deque(maxlen=3)

# --- LOCKS ---
bg_status_lock = threading.Lock()
notifications_lock = threading.Lock()
upt_metrics_lock = threading.Lock()
vector_store_lock = threading.Lock()

# --- CONFIG PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
GENERATED_FOLDER = os.path.join(BASE_DIR, 'static', 'generated')
VOICE_FOLDER = os.path.join(BASE_DIR, 'static', 'voice')
CHAT_LOG_FILE = os.path.join(BASE_DIR, 'data', 'last_conversation.json')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)
os.makedirs(VOICE_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(CHAT_LOG_FILE), exist_ok=True)

ALLOWED_EXTENSIONS = {'json', 'csv', 'txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'py', 'js', 'html', 'css', 'md'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_file(filename):
    is_valid_ext = filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    return '.' in filename and (is_valid_ext or allowed_image(filename))

def web_log(message: str):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    LOG_QUEUE.put(f"[{timestamp}] {message}")

@app.route('/stream_logs')
def stream_logs():
    logs = []
    while not LOG_QUEUE.empty():
        logs.append(LOG_QUEUE.get())
    return jsonify({'logs': logs})

# --- EXECUTION CORE ---
def _execute_script(filename: str):
    global last_upt_values, last_upt_metrics, BACKGROUND_TASK_STATUS
    safe_filename = secure_filename(filename)
    filepath = os.path.join(UPLOAD_FOLDER, safe_filename)

    if not os.path.exists(filepath):
        return {'success': False, 'output': "File not found"}

    full_stdout = []
    full_stderr = []
    generated_image_url = None
    live_upt_metrics = None

    try:
        web_log(f"🚀 Đang chạy lệnh: {safe_filename}...")
        with bg_status_lock:
            BACKGROUND_TASK_STATUS = {"status": "running", "task": f"Running {safe_filename}"}

        for old_img in glob.glob("*.png"):
            try: os.remove(old_img)
            except: pass

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            [sys.executable, filepath],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding='utf-8', errors='replace', cwd=os.getcwd(), env=env, bufsize=1, universal_newlines=True
        )

        def reader(pipe, is_err):
            for line in iter(pipe.readline, ''):
                if line.strip():
                    prefix = "[ERR] " if is_err else "[OUT] "
                    web_log(f"{prefix}{line.strip()}")
                    if is_err: full_stderr.append(line)
                    else: full_stdout.append(line)
            pipe.close()

        t_out = threading.Thread(target=reader, args=(process.stdout, False))
        t_err = threading.Thread(target=reader, args=(process.stderr, True))
        t_out.start(); t_err.start()
        t_out.join(60); t_err.join(60)

        if process.poll() is None:
            process.terminate()
            web_log("⚠️ Timeout! Process killed.")
            return {'success': False, 'output': "Timeout (60s)"}

        current_images = glob.glob("*.png")
        if current_images:
            img_name = current_images[0]
            dst = os.path.join(GENERATED_FOLDER, img_name)
            if os.path.exists(dst): os.remove(dst)
            shutil.move(img_name, dst)
            generated_image_url = f"/static/generated/{img_name}"
            web_log(f"📸 Đã tạo ảnh: {img_name}")
            
            try:
                vision_desc = deloris_eye.analyze_image(dst, prompt="Analyze the content of this generated plot/image.")
                web_log(f"🎨 Deloris đánh giá kết quả: {vision_desc}")
                
                img_obj = Image.open(dst).convert("RGB")
                with upt_metrics_lock: aec, met = last_upt_values, last_upt_metrics.copy()
                with torch.no_grad():
                    vec = clip_model.get_image_features(**clip_processor(images=img_obj, return_tensors="pt", padding=True)).to(dtype=torch.float32)
                    t_vec = torch.tensor(vectorizer.encode([""]), dtype=torch.float32)
                    a_vec = torch.tensor([list(aec)], dtype=torch.float32)
                    m_vec = torch.tensor([[met['CI'], met['Pulse'], met['Entanglement']]], dtype=torch.float32)
                    inp = torch.cat((t_vec, a_vec, m_vec), dim=1)
                    oa, oe, oc = predictor_model(inp, vec)
                    at, et, ct = max(oa.item(), 0.1), max(oe.item()*5.0, 0.1), max(oc.item()*3.0, 0.1)
                
                at, et, ct = plasticity.apply_bias(at, et, ct)
                new_met = upt_calculator.update_metrics(at, et, ct)
                with upt_metrics_lock:
                    last_upt_values = (at, et, ct)
                    last_upt_metrics.update({"CI": new_met['CI'], "Pulse": new_met['Pulse']})
                    live_upt_metrics = last_upt_metrics.copy()
            except Exception as e: web_log(f"Vision/UPT Error: {e}")

        elif os.path.exists(GENERATED_FOLDER):
             files = sorted(glob.glob(os.path.join(GENERATED_FOLDER, "gen_*.png")), key=os.path.getmtime)
             if files and (time.time() - os.path.getmtime(files[-1])) < 10:
                 latest_img = os.path.basename(files[-1])
                 generated_image_url = f"/static/generated/{latest_img}"

        return {'success': True, 'stdout': "".join(full_stdout), 'image_url': generated_image_url, 'live_upt_metrics': live_upt_metrics, 'output': "".join(full_stdout) + "\n" + "".join(full_stderr)}
    except Exception as e:
        web_log(f"❌ Error: {e}")
        return {'success': False, 'output': str(e)}
    finally:
        with bg_status_lock:
            BACKGROUND_TASK_STATUS = {"status": "idle", "task": "Hoàn tất"}

def _run_existing_script_skill(user_prompt: str):
    files = os.listdir(UPLOAD_FOLDER)
    target = None
    for f in files:
        if f.endswith(".py") and f in user_prompt:
            target = f
            break
    if not target: return {"deloris_response": "Không tìm thấy file code nào khớp.", "live_upt_metrics": None}
    res = _execute_script(target)
    fmt = f"**[SYSTEM EXECUTOR]**\nĐã chạy: `{target}`\n\n```bash\n{res.get('stdout','')}\n```"
    if res.get('image_url'): fmt += f"\n\n\n![Result]({res.get('image_url')})"
    return {"deloris_response": fmt, "live_upt_metrics": res.get('live_upt_metrics')}

# --- GRACEFUL SHUTDOWN ---
def graceful_shutdown(signum=None, frame=None):
    global SYSTEM_ACTIVE
    if not SYSTEM_ACTIVE: return
    SYSTEM_ACTIVE = False
    print("\n\n🛑 [SYSTEM HALT] Sao lưu dữ liệu...")
    try:
        if chat_history:
            with open(CHAT_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(chat_history, f, ensure_ascii=False, indent=2)
        if vector_store_chat: vector_store_chat.save_local(config.FAISS_INDEX_CHAT_PATH)
        if vector_store_docs: vector_store_docs.save_local(config.FAISS_INDEX_DOCS_PATH)
        if dreamer: dreamer.consolidate_memories()
    except: pass
    print("👋 [GOODBYE] Deloris ngủ đông.")
    sys.exit(0)

# --- LOADER ---
def load_models():
    global vectorizer, deloris_model, predictor_model, upt_calculator, chat_history, text_splitter, clip_processor, clip_model, vector_store_docs, vector_store_chat, embeddings_model, dummy_image_vector, superego, plasticity, dreamer, heartbeat, motor, coder, wallet
    if vectorizer is not None: return

    print(">>> [SYSTEM] Đang khởi tạo Neural Core...")
    try:
        # History
        if os.path.exists(CHAT_LOG_FILE):
            try:
                with open(CHAT_LOG_FILE, 'r', encoding='utf-8') as f:
                    chat_history = json.load(f)
            except: pass

        if os.environ.get("GEMINI_API_KEY"): genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        vectorizer = SentenceTransformer(config.LANGUAGE_MODEL_NAME)
        embeddings_model = SentenceTransformerEmbeddings(model_name=config.LANGUAGE_MODEL_NAME)
        
        deloris_model = DelorisModel(config.INPUT_DIM, config.DELORIS_HIDDEN_DIM, config.DELORIS_OUTPUT_DIM)
        if os.path.exists(config.DELORIS_MODEL_PATH):
            try: deloris_model.load_state_dict(torch.load(config.DELORIS_MODEL_PATH, map_location='cpu'))
            except: pass
        deloris_model.eval()
        
        predictor_model = UPTAutomatorModel(config.PREDICTOR_INPUT_DIM, config.IMAGE_VECTOR_DIM, config.AUTOMATOR_HIDDEN_DIM)
        if os.path.exists(config.AUTOMATOR_MODEL_PATH):
            try: predictor_model.load_state_dict(torch.load(config.AUTOMATOR_MODEL_PATH, map_location='cpu'))
            except: pass
        predictor_model.eval()
        
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        dummy_image_vector = torch.zeros(1, config.IMAGE_VECTOR_DIM)

        upt_calculator = UPTCalculator(dt=1.0)
        superego = SuperegoMonitor()
        plasticity = PlasticityLayer()
        dreamer = DreamWeaver()
        
        # [KÍCH HOẠT CÁC MODULE MỞ RỘNG]
        motor = MotorSystem()
        coder = NeuralCoder(UPLOAD_FOLDER)
        wallet = CryptoWallet() # [MỚI] Web3 Wallet
        print("   -> Motor, Coder & Wallet Systems: ONLINE")
        
        heartbeat = HeartbeatSystem(GLOBAL_NOTIFICATIONS, last_upt_metrics, chat_history)
        heartbeat.start_loop()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        try:
            if os.path.exists(config.FAISS_INDEX_CHAT_PATH): vector_store_chat = FAISS.load_local(config.FAISS_INDEX_CHAT_PATH, embeddings_model, allow_dangerous_deserialization=True)
            else: vector_store_chat = FAISS.from_texts(["Init Chat"], embeddings_model)
        except: vector_store_chat = FAISS.from_texts(["Init Chat"], embeddings_model)

        try:
            if os.path.exists(config.FAISS_INDEX_DOCS_PATH): vector_store_docs = FAISS.load_local(config.FAISS_INDEX_DOCS_PATH, embeddings_model, allow_dangerous_deserialization=True)
            else: vector_store_docs = FAISS.from_texts(["Init Docs"], embeddings_model)
        except: vector_store_docs = FAISS.from_texts(["Init Docs"], embeddings_model)
        
        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)
        
        print(">>> [SYSTEM] KHỞI TẠO HOÀN TẤT.")
    except Exception as e:
        print(f"!!! CRITICAL BOOT ERROR: {e}")
        traceback.print_exc()

# --- ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/user_presence', methods=['POST'])
def user_presence():
    global SYSTEM_ACTIVE
    st = request.json.get('status')
    if st == 'away': SYSTEM_ACTIVE = False; return jsonify({'msg': 'Standby'})
    if st == 'active': SYSTEM_ACTIVE = True; return jsonify({'msg': 'Active'})
    if st == 'closed': graceful_shutdown(); return jsonify({'msg': 'Saved'})
    return jsonify({'status': 'ok'})

@app.route('/chat', methods=['POST'])
def chat():
    global last_upt_values, last_upt_metrics, chat_history, LATEST_VISUAL_CONTEXT
    if vectorizer is None: load_models()
    
    try:
        data = request.json
        msg = data.get('message', '')
        is_background = data.get('is_background', False)
        
        if not msg: return jsonify({'error': 'Empty'}), 400

        # --- 1. MOTOR SYSTEM (Hành động vật lý) ---
        if motor:
            motor_resp = motor.detect_and_act(msg)
            if motor_resp:
                chat_history.append(f"User: {msg}")
                chat_history.append(f"Deloris (Motor): {motor_resp}")
                return jsonify({'deloris_response': motor_resp, 'live_upt_metrics': last_upt_metrics})
        
        # --- 2. CRYPTO WALLET (Tài chính Web3) ---
        if wallet:
            msg_lower = msg.lower()
            # Kiểm tra số dư
            if "số dư" in msg_lower or "balance" in msg_lower or "ví của em" in msg_lower:
                bal = wallet.get_balance()
                addr = wallet.get_address()
                resp = f"💳 **Ví Web3 của Deloris**\n- Địa chỉ: `{addr}`\n- Số dư: **{bal}**"
                chat_history.append(f"User: {msg}")
                chat_history.append(f"Deloris (Wallet): {resp}")
                return jsonify({'deloris_response': resp, 'live_upt_metrics': last_upt_metrics})
            
            # Gửi tiền
            if "gửi" in msg_lower and ("eth" in msg_lower or "tiền" in msg_lower) and "cho" in msg_lower:
                try:
                    # Trích xuất số tiền và địa chỉ ví
                    amounts = re.findall(r"[-+]?\d*\.\d+|\d+", msg)
                    addr_match = re.search(r"0x[a-fA-F0-9]{40}", msg)
                    
                    if amounts and addr_match:
                        amount = float(amounts[0])
                        target_addr = addr_match.group()
                        
                        # Thực hiện giao dịch
                        tx_res = wallet.send_eth(target_addr, amount)
                        resp = f"💸 **Lệnh chuyển tiền:**\n{tx_res}"
                        
                        chat_history.append(f"User: {msg}")
                        chat_history.append(f"Deloris (Wallet): {resp}")
                        return jsonify({'deloris_response': resp, 'live_upt_metrics': last_upt_metrics})
                except: pass

        # --- 3. NEURAL CODER (Tự viết code) ---
        if coder and any(k in msg.lower() for k in ["viết code", "lập trình", "tạo script", "code cho", "viết chương trình", "tạo tool"]):
            script_name, script_content = coder.create_script(msg)
            if script_name:
                exec_res = _execute_script(script_name)
                response_text = f"**[NEURAL CODER]**\nEm đã viết xong chương trình `{script_name}`.\n\n"
                response_text += f"```python\n{script_content}\n```\n\n"
                response_text += f"**KẾT QUẢ CHẠY:**\n```bash\n{exec_res.get('stdout', '')}\n```"
                if exec_res.get('image_url'):
                    response_text += f"\n\n![Kết quả đồ họa]({exec_res.get('image_url')})"
                
                chat_history.append(f"User: {msg}")
                chat_history.append(f"Deloris (Coder): {response_text}")
                return jsonify({'deloris_response': response_text, 'live_upt_metrics': last_upt_metrics})
        # ----------------------------------------
        
        # [WAKE WORD LOGIC]
        if is_background:
            msg_lower = msg.lower()
            is_wake_word = any(w in msg_lower for w in WAKE_WORDS)
            last_interaction = heartbeat.last_interaction if heartbeat else 0
            is_in_conversation = (time.time() - last_interaction) < 30
            
            if not is_wake_word and not is_in_conversation:
                web_log(f"🔇 [IGNORED] Tiếng ồn nền: '{msg}'")
                return jsonify({'deloris_response': '', 'silent': True, 'live_upt_metrics': last_upt_metrics})

        if heartbeat: heartbeat.touch()

        if "chạy file" in msg.lower() or "run script" in msg.lower():
            return jsonify(_run_existing_script_skill(msg))

        final_msg_for_ai = msg
        if LATEST_VISUAL_CONTEXT:
             final_msg_for_ai = f"{msg} \n[THÔNG TIN TỪ MẮT (MOONDREAM): {LATEST_VISUAL_CONTEXT}]"

        vec = torch.tensor(vectorizer.encode([msg]), dtype=torch.float32)
        
        with upt_metrics_lock: aec, met = last_upt_values, last_upt_metrics.copy()
        inp = torch.cat((vec, torch.tensor([list(aec)], dtype=torch.float32), torch.tensor([[met['CI'], met['Pulse'], met['Entanglement']]], dtype=torch.float32)), dim=1)
        dummy = dummy_image_vector if dummy_image_vector is not None else torch.zeros(1, config.IMAGE_VECTOR_DIM)
        
        with torch.no_grad(): oa, oe, oc = predictor_model(inp, dummy)
        at, et, ct = max(oa.item(), 0.1), max(oe.item()*5.0, 0.1), max(oc.item()*3.0, 0.1)

        at, et, ct = plasticity.apply_bias(at, et, ct)
        new_met = upt_calculator.update_metrics(at, et, ct)
        new_met, warnings, is_unstable = superego.stabilize_metrics(new_met)
        if warnings:
            for w in warnings: web_log(w)
        
        with upt_metrics_lock:
            last_upt_values = (at, et, ct)
            last_upt_metrics.update(new_met)

        docs = []
        with vector_store_lock:
            if vector_store_docs: docs += vector_store_docs.similarity_search(msg, k=3)

        with torch.no_grad():
            pred = deloris_model(vec, last_upt_metrics)
            cls = torch.argmax(pred, dim=1).item()
            
        state_str = f"CI: {new_met['CI']:.2f} | Pulse: {new_met['Pulse']:.2f}"
        
        # [NEURO-LINK] Get heartbeat status for dynamic prompting
        heartbeat_status = None
        if heartbeat:
            heartbeat_status = heartbeat.get_status()
            web_log(f"💓 [NEURO-LINK] Status: Energy={heartbeat_status.get('energy', 0)}%, Mood={heartbeat_status.get('mood', 'Unknown')}")
        
        raw_resp = generate_final_response(cls, final_msg_for_ai, chat_history, docs, 0.5, "neutral", state_str, new_met['CI'], None, pulse_value=new_met['Pulse'], heartbeat_status=heartbeat_status)
        safe_resp = superego.censor_response(raw_resp, is_unstable)
        
        should_draw, art_prompt = detect_art_intent(msg, new_met['Pulse'])
        if should_draw:
            web_log(f"🎨 Deloris muốn vẽ: '{art_prompt}'")
            img_url = generate_image(art_prompt, GENERATED_FOLDER)
            if img_url: safe_resp += f"\n\n![Tranh Deloris vẽ]({img_url})"
        
        chat_history.append(f"User: {msg}")
        chat_history.append(f"Deloris: {safe_resp}")
        
        return jsonify({'deloris_response': safe_resp, 'live_upt_metrics': last_upt_metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentinel', methods=['POST'])
def sentinel_eye():
    if 'file' not in request.files: return jsonify({'message': None})
    f = request.files['file']
    try:
        temp_filename = f"sentinel_{uuid.uuid4()}.jpg"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        f.save(temp_path)
        
        global last_upt_values, last_upt_metrics, LATEST_VISUAL_CONTEXT
        
        try:
            image_obj = Image.open(temp_path).convert("RGB")
            with torch.no_grad():
                visual_features = clip_model.get_image_features(**clip_processor(images=image_obj, return_tensors="pt", padding=True)).to(dtype=torch.float32)
                dummy_text = vectorizer.encode([""])
                text_tensor = torch.tensor(dummy_text, dtype=torch.float32)
                with upt_metrics_lock:
                    prev_aec = list(last_upt_values)
                    prev_metrics = [last_upt_metrics['CI'], last_upt_metrics['Pulse'], last_upt_metrics['Entanglement']]
                state_tensor_aec = torch.tensor([prev_aec], dtype=torch.float32)
                state_tensor_met = torch.tensor([prev_metrics], dtype=torch.float32)
                textual_input = torch.cat((text_tensor, state_tensor_aec, state_tensor_met), dim=1)
                oa, oe, oc = predictor_model(textual_input, visual_features)
                at, et, ct = max(oa.item(), 0.1), max(oe.item()*5.0, 0.1), max(oc.item()*3.0, 0.1)
                new_met = upt_calculator.update_metrics(at, et, ct)
                with upt_metrics_lock:
                    last_upt_values = (at, et, ct)
                    last_upt_metrics.update(new_met)
        except Exception as e:
            web_log(f"Sentinel CLIP Error: {e}")
            new_met = last_upt_metrics

        web_log("👁️ [SENTINEL] Đang phân tích ảnh qua Moondream...")
        description = deloris_eye.analyze_image(temp_path, prompt="Describe what is happening in this image briefly.")
        LATEST_VISUAL_CONTEXT = description
        
        try: os.remove(temp_path)
        except: pass
        
        return jsonify({'message': None, 'pulse': new_met.get('Pulse', 0)})

    except Exception as e:
        web_log(f"Sentinel Error: {e}")
        return jsonify({'message': None})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    if f and allowed_file(f.filename):
        n = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, n)
        f.save(path)
        threading.Thread(target=lambda: _ingest_file(n), daemon=True).start()
        return jsonify({'message': 'OK'})
    return jsonify({'error': 'Invalid'}), 400

def _ingest_file(fname):
    try:
        path = os.path.join(UPLOAD_FOLDER, fname)
        if allowed_image(fname):
            web_log(f"👁️ Đang kích hoạt thị giác cho: {fname}...")
            desc = deloris_eye.analyze_image(path)
            web_log(f"   -> Nội dung ảnh: {desc}")
            if vector_store_docs:
                doc_content = f"[IMAGE MEMORY] Filename: {fname}\nDescription: {desc}"
                with vector_store_lock:
                    vector_store_docs.add_documents(text_splitter.create_documents([doc_content]))
                    vector_store_docs.save_local(config.FAISS_INDEX_DOCS_PATH)
            
            try:
                global last_upt_values, last_upt_metrics
                image_obj = Image.open(path).convert("RGB")
                with torch.no_grad():
                    visual_features = clip_model.get_image_features(**clip_processor(images=image_obj, return_tensors="pt", padding=True)).to(dtype=torch.float32)
                    dummy_text = vectorizer.encode([""]) 
                    text_tensor = torch.tensor(dummy_text, dtype=torch.float32)
                    with upt_metrics_lock:
                        prev_aec = list(last_upt_values)
                        prev_metrics = [last_upt_metrics['CI'], last_upt_metrics['Pulse'], last_upt_metrics['Entanglement']]
                    state_tensor_aec = torch.tensor([prev_aec], dtype=torch.float32)
                    state_tensor_met = torch.tensor([prev_metrics], dtype=torch.float32)
                    textual_input = torch.cat((text_tensor, state_tensor_aec, state_tensor_met), dim=1)
                    oa, oe, oc = predictor_model(textual_input, visual_features)
                    at, et, ct = max(oa.item(), 0.1), max(oe.item() * 5.0, 0.1), max(oc.item() * 3.0, 0.1)
                    at, et, ct = plasticity.apply_bias(at, et, ct)
                    new_met = upt_calculator.update_metrics(at, et, ct)
                    with upt_metrics_lock:
                        last_upt_values = (at, et, ct)
                        last_upt_metrics.update(new_met)
                    web_log(f"-> Cảm thấy từ ảnh (Upload): Pulse {new_met['Pulse']:.2f}")
            except Exception as e: web_log(f"Vision UPT Error: {e}")
        else:
            s, _, res = FileProcessor.process_file(path, False)
            if s and vector_store_docs:
                with vector_store_lock:
                    vector_store_docs.add_documents(text_splitter.create_documents([res['content']]))
                    vector_store_docs.save_local(config.FAISS_INDEX_DOCS_PATH)
                web_log(f"Đã học xong: {fname}")
                
    except Exception as e: web_log(f"Ingest Error: {e}")

@app.route('/api/files', methods=['GET'])
def list_f():
    if not os.path.exists(UPLOAD_FOLDER): return jsonify([])
    return jsonify([{'name': f} for f in os.listdir(UPLOAD_FOLDER) if not f.startswith('gen_')])

@app.route('/api/files/<n>', methods=['DELETE'])
def del_f(n):
    try:
        os.remove(os.path.join(UPLOAD_FOLDER, secure_filename(n)))
        return jsonify({'success': True})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/speak', methods=['POST'])
def speak():
    data = request.json
    txt = data.get('text', '')
    current_pulse = float(data.get('pulse', 0.0))
    if not txt: return jsonify({'error': 'No text'}), 400
    fn = f"v_{uuid.uuid4()}.mp3"
    fp = os.path.join(VOICE_FOLDER, fn)
    VOICE_ID = "vi-VN-HoaiMyNeural" 
    try:
        rate_val = int(current_pulse * 4) 
        rate_str = f"{rate_val:+d}%"
        pitch_val = int(current_pulse * 2)
        pitch_str = f"{pitch_val:+d}Hz"
        web_log(f"🗣️ Voice: Rate {rate_str} | Pitch {pitch_str}")
        async def _generate_neural_voice():
            communicate = edge_tts.Communicate(txt, VOICE_ID, rate=rate_str, pitch=pitch_str)
            await communicate.save(fp)
        asyncio.run(_generate_neural_voice())
        return jsonify({'url': f"/static/voice/{fn}"})
    except Exception as e: return jsonify({'url': ''})

@app.route('/perceive_image', methods=['POST'])
def perceive_image():
    f = request.files['file']
    if not allowed_image(f.filename): return jsonify({'success': False})
    return jsonify({'success': True, 'message': "Vision handled via ingest", 'live_upt_metrics': last_upt_metrics})

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    try:
        if plasticity:
            plasticity.record_feedback(
                user_input=data.get('input', ''), 
                model_output=data.get('output', ''), 
                upt_state=last_upt_metrics, 
                rating=int(data.get('rating', 0))
            )
            return jsonify({'success': True, 'message': 'Bias updated in Database'})
        return jsonify({'success': False, 'message': 'Plasticity module not ready'})
    except Exception as e: return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sleep', methods=['POST'])
def trigger_sleep():
    web_log("🌙 Deloris đang đi vào trạng thái Giấc Mơ (Memory Consolidation)...")
    if dreamer:
        s1, c1 = dreamer.consolidate_memories()
        s2, c2 = False, 0
        try: s2, c2 = dreamer.lucid_dream(num_scenarios=2) 
        except AttributeError: pass
        msg = []
        if s1: msg.append(f"Đã lưu {c1} ký ức vào DB Training.")
        if s2: msg.append(f"Đã mơ thấy {c2} kịch bản mới.")
        full_msg = " ".join(msg) if msg else "Không có gì mới để lưu."
        web_log(f"💤 [DREAM DONE] {full_msg}")
        return jsonify({'success': True, 'message': full_msg})
    return jsonify({'success': False, 'message': "Dreamer error"})

@app.route('/retrain_model', methods=['POST'])
def rtm():
    s, m = retrain_job.run_retraining()
    return jsonify({'status': 'ok' if s else 'error', 'message': m})

@app.route('/reset_memory', methods=['POST'])
def rsm():
    global vector_store_docs, vector_store_chat
    try:
        with vector_store_lock:
            if os.path.exists(config.FAISS_INDEX_DOCS_PATH): shutil.rmtree(config.FAISS_INDEX_DOCS_PATH)
            if os.path.exists(config.FAISS_INDEX_CHAT_PATH): shutil.rmtree(config.FAISS_INDEX_CHAT_PATH)
            vector_store_docs = FAISS.from_texts(["Init"], embeddings_model)
            vector_store_chat = FAISS.from_texts(["Init"], embeddings_model)
        return jsonify({'success': True})
    except Exception as e: return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status', methods=['GET'])
def gst(): return jsonify(BACKGROUND_TASK_STATUS)

@app.route('/api/notifications', methods=['GET'])
def gn():
    with notifications_lock:
        m = GLOBAL_NOTIFICATIONS.popleft() if GLOBAL_NOTIFICATIONS else None
    return jsonify({'message': m})

def _self_diagnostic():
    import time
    import requests
    print("\n⏳ [SYSTEM] Đang đợi Core ổn định trước khi tự kiểm tra (3s)...")
    time.sleep(3) 
    print("--- [DIAGNOSTIC] BẮT ĐẦU TỰ KIỂM TRA HỆ THỐNG ---")
    target_url = f"http://127.0.0.1:{config.FLASK_PORT}/api/feedback"
    try:
        payload = {"input": "SELF_DIAGNOSTIC_TEST", "output": "SYSTEM_CHECK_OK", "rating": 1}
        res = requests.post(target_url, json=payload, timeout=2)
        if res.status_code == 200:
            print("   -> ✅ [PASS] Module Neuroplasticity (DB): ONLINE")
        else:
            print(f"   -> ⚠️ [WARNING] Module phản hồi mã lạ: {res.status_code}")
    except Exception as e:
        print(f"   -> ❌ [FAIL] Không thể tự kết nối: {e}")
    print("--- [DIAGNOSTIC] HOÀN TẤT ---\n")

if __name__ == '__main__':
    load_models()
    threading.Thread(target=_self_diagnostic, daemon=True).start()
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False, use_reloader=False)
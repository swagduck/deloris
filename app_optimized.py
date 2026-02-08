# app_optimized.py (Phiên bản Tối ưu hóa Hiệu năng)
# Implement async/await for parallel processing and other optimizations

import torch
import torch.nn as nn
from deloris_ai.architecture import DelorisModel
from upt_core.calculator import UPTCalculator
from deloris_ai.response_mapper import generate_final_response, summarize_conversation 
from sentence_transformers import SentenceTransformer
import warnings
import json
import asyncio
import concurrent.futures
from upt_predictor.compatibility import UPTAutomatorModelCompat
from upt_core.prediction_error import PredictionErrorSystem
from deloris_ai.inner_monologue_optimized import InnerMonologueSystemOptimized
from deloris_ai.vector_memory import VectorMemorySystem
from deloris_ai.rlhf_collector import RLHFDataCollector
import config

warnings.filterwarnings("ignore")

print("Đang khởi tạo hệ thống Deloris (Tối ưu Hiệu năng + Vector Memory + RLHF)...")

# Load models
try:
    print(f"Đang tải Bộ não Ngôn ngữ ({config.LANGUAGE_MODEL_NAME})...")
    vectorizer = SentenceTransformer(config.LANGUAGE_MODEL_NAME)
    print("Bộ não Ngôn ngữ: Sẵn sàng.")
except Exception as e: exit(f"LỖI: {e}")

try:
    deloris = DelorisModel(config.INPUT_DIM, config.DELORIS_HIDDEN_DIM, config.DELORIS_OUTPUT_DIM)
    deloris.load_state_dict(torch.load(config.DELORIS_MODEL_PATH))
    deloris.eval()
    print(f"Mô hình Deloris ({config.DELORIS_MODEL_PATH}): Sẵn sàng.")
except FileNotFoundError: exit(f"LỖI: Không tìm thấy {config.DELORIS_MODEL_PATH}")

try:
    predictor_model = UPTAutomatorModelCompat(
        config.INPUT_DIM, 
        config.IMAGE_VECTOR_DIM, 
        config.AUTOMATOR_HIDDEN_DIM
    )
    predictor_model.load_state_dict(torch.load(config.AUTOMATOR_MODEL_PATH))
    predictor_model.eval()
    print(f"Bộ dự đoán UPT ({config.AUTOMATOR_MODEL_PATH}): Sẵn sàng.")
except FileNotFoundError: exit(f"LỖI: Không tìm thấy {config.AUTOMATOR_MODEL_PATH}")

upt_calc = UPTCalculator(dt=1.0)
print("Lõi UPT: Sẵn sàng.")

# Initialize optimized systems
inner_monologue = InnerMonologueSystemOptimized()
prediction_error = PredictionErrorSystem()
vector_memory = VectorMemorySystem()
rlhf_collector = RLHFDataCollector()
print("Hệ thống Inner Monologue (Tối ưu), Vector Memory, RLHF: Sẵn sàng.")

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

def save_feedback(text, a, e, c, predicted_class_label):
    try:
        feedback_data = {
            "input_text": text, 
            "predicted_A": a, 
            "predicted_E": e, 
            "predicted_C": c, 
            "predicted_class_label": predicted_class_label, 
            "reason": "User marked 'needs correction'"
        }
        with open(config.FEEDBACK_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + '\n')
        print("Phản hồi đã được lưu lại để huấn luyện trong tương lai.")
    except Exception as ex: print(f"Lỗi khi lưu phản hồi: {ex}")

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
    """Async inner thought generation with local fallback"""
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

async def run_deloris_chat_async():
    """Async main chat loop with parallel processing"""
    global upt_calc
    
    # Load vector memory instead of simple text
    current_memory = await vector_memory.load_memory()
    chat_history = []

    print("-" * 30)
    print("Chào mừng đến với Deloris (Tối ưu Hiệu năng + Vector Memory + RLHF). Gõ 'exit' để thoát.")
    print("-" * 30)
    
    while True:
        print("\n" + "=" * 30)
        text_input = input("Bạn nói: ")
        
        if text_input.lower() == 'exit':
            final_upt_state = upt_calc.update_metrics(A_t=0, E_t=0, C_t=0) 
            await vector_memory.save_memory(chat_history, final_upt_state)
            print("Tạm biệt. Ngắt kết nối khỏi tầng nhận thức. Vector Memory đã được lưu.")
            break
            
        input_vector = torch.tensor(vectorizer.encode([text_input]), dtype=torch.float32)
        
        # Pad or truncate to match expected input dimension (774)
        if input_vector.shape[1] < 774:
            # Pad with zeros
            padding = torch.zeros(1, 774 - input_vector.shape[1])
            input_vector = torch.cat([input_vector, padding], dim=1)
        elif input_vector.shape[1] > 774:
            # Truncate
            input_vector = input_vector[:, :774]

        # PARALLEL PROCESSING: Run UPT prediction and sentiment prediction simultaneously
        print("🚀 Đang xử lý song song...")
        start_time = asyncio.get_event_loop().time()
        
        upt_task = process_upt_prediction_async(input_vector)
        sentiment_task = process_sentiment_prediction_async(text_input, {}, chat_history)
        
        upt_results = await asyncio.gather(
            upt_task, sentiment_task
        )
        
        # Unpack the results
        upt_values = upt_results[0]  # (A_t, E_t, C_t)
        sentiment_values = upt_results[1]  # (predicted_sentiment, confidence)
        
        A_t, E_t, C_t = upt_values
        predicted_sentiment, confidence = sentiment_values
        
        processing_time = asyncio.get_event_loop().time() - start_time
        print(f"⚡ Xử lý song song hoàn tất trong {processing_time:.2f}s")
        
        print(f"UPT tự động đo được: A={A_t:.2f}, E={E_t:.2f}, C={C_t:.2f}")
        print(f"[Prediction] Dự đoán User sẽ: {predicted_sentiment} (confidence: {confidence:.2f})")

        upt_metrics = upt_calc.update_metrics(A_t, E_t, C_t)
        
        # PARALLEL PROCESSING: Inner thought and Deloris classification
        # Use original 384-dim input for Deloris model
        deloris_input = torch.tensor(vectorizer.encode([text_input]), dtype=torch.float32)
        inner_thought_task = process_inner_thought_async(text_input, upt_metrics, chat_history, "")
        deloris_task = process_deloris_classification_async(deloris_input, upt_metrics)
        
        inner_results = await asyncio.gather(
            inner_thought_task, deloris_task
        )
        
        # Unpack the results
        inner_thought = inner_results[0]
        predicted_class = inner_results[1]
        
        print(f"[Inner Monologue] Suy nghĩ: '{inner_thought}'")
        
        # Generate response from thought
        response_from_thought = await process_response_from_thought_async(
            inner_thought, text_input, upt_metrics, chat_history
        )
        
        # Retrieve relevant memories using vector search
        relevant_memories = await vector_memory.search_similar(text_input, k=3)
        
        # Generate final response with vector memories
        result = generate_final_response(
            predicted_class, 
            text_input, 
            relevant_memories,  # retrieved_docs (3rd parameter)
            chat_history,
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
            # Function returns single value or different format
            final_output_message = str(result)
            clean_response_text = str(result)
        
        print(f"Trạng thái UPT (CI, Pulse): CI={upt_metrics['CI']:.2f}, Pulse={upt_metrics['Pulse']:.2f}")
        print(final_output_message) 
        print("-" * 30)
        
        # Update chat history
        chat_history.append(f"Bạn nói: {text_input}")
        chat_history.append(f"Deloris: {clean_response_text}")

        # RLHF Data Collection
        await rlhf_collector.collect_interaction(
            input_text=text_input,
            inner_thought=inner_thought,
            response=clean_response_text,
            predicted_class=predicted_class,
            upt_metrics=upt_metrics
        )

        # Feedback loop with enhanced tracking
        while True:
            feedback = input("Phản hồi này Tốt (1) hay Cần sửa (2)? (gõ 1 hoặc 2): ")
            if feedback == '1': 
                surprise = prediction_error.calculate_surprise(predicted_sentiment, "positive")
                learning_multiplier = prediction_error.get_learning_rate_multiplier()
                pulse_adjustment = prediction_error.should_adjust_pulse(surprise)
                
                # Save RLHF data
                await rlhf_collector.save_feedback(score=1.0, feedback_text="good")
                
                print(f"Tuyệt vời! Đã ghi nhận.")
                print(f"[Prediction Error] Surprise: {surprise:.2f}, Learning Rate: x{learning_multiplier:.2f}")
                if pulse_adjustment != 0:
                    print(f"[Prediction Error] Pulse điều chỉnh: {pulse_adjustment:+.2f}")
                    if upt_calc:
                        current_pulse = upt_metrics.get('Pulse', 0)
                        new_pulse = max(-5, min(10, current_pulse + pulse_adjustment))
                        upt_calc.last_Pulse = new_pulse
                        upt_metrics['Pulse'] = new_pulse
                break
                
            elif feedback == '2':
                surprise = prediction_error.calculate_surprise(predicted_sentiment, "negative")
                learning_multiplier = prediction_error.get_learning_rate_multiplier()
                pulse_adjustment = prediction_error.should_adjust_pulse(surprise)
                
                strategy_label = f"Lớp {predicted_class}"
                save_feedback(text_input, A_t, E_t, C_t, strategy_label)
                
                # Save RLHF data
                await rlhf_collector.save_feedback(score=0.0, feedback_text="needs_correction")
                
                print(f"[Prediction Error] Surprise: {surprise:.2f}, Learning Rate: x{learning_multiplier:.2f}")
                if pulse_adjustment != 0:
                    print(f"[Prediction Error] Pulse điều chỉnh: {pulse_adjustment:+.2f}")
                    if upt_calc:
                        current_pulse = upt_metrics.get('Pulse', 0)
                        new_pulse = max(-5, min(10, current_pulse + pulse_adjustment))
                        upt_calc.last_Pulse = new_pulse
                        upt_metrics['Pulse'] = new_pulse
                break
                
            else: print("Vui lòng chỉ gõ 1 hoặc 2.")

def run_deloris_chat():
    """Wrapper to run async chat"""
    asyncio.run(run_deloris_chat_async())

if __name__ == "__main__":
    run_deloris_chat()

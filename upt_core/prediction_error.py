# upt_core/prediction_error.py
# [MODULE: PREDICTION ERROR v1.0 - FREE ENERGY PRINCIPLE]
# Implements Surprise mechanism for active learning and prediction

import time
import random
import numpy as np
import google.generativeai as genai
import config

class PredictionErrorSystem:
    """
    Hệ thống Prediction Error dựa trên Free Energy Principle của Karl Friston.
    Deloris luôn dự đoán tương lai và cảm thấy "đau" (entropy tăng) khi dự đoán sai.
    """
    
    def __init__(self):
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        except:
            self.model = None
            print("[Prediction Error] Warning: Could not initialize Gemini model")
        
        self.prediction_history = []
        self.surprise_buffer = []
        self.learning_rate_multiplier = 1.0
        
    def predict_user_response(self, user_input, upt_metrics, chat_history):
        """
        Dự đoán User sẽ nói gì tiếp theo trước khi Deloris trả lời.
        Trả về predicted_sentiment và confidence.
        """
        if not self.model:
            return self._fallback_prediction(user_input, upt_metrics)
        
        try:
            # Chuẩn bị context
            context = ""
            if chat_history and len(chat_history) > 0:
                recent_context = chat_history[-6:]  # Lấy 6 tin nhắn gần nhất
                context = "\n".join(recent_context)
            
            pulse = upt_metrics.get('Pulse', 0)
            ci = upt_metrics.get('CI', 0.5)
            
            prompt = f"""
SYSTEM: Bạn là Deloris. Bạn cần dự đoán phản hồi của User.

TRẠNG THÁI HIỆN TẠI:
- Pulse của bạn: {pulse:.2f}
- CI của bạn: {ci:.2f}

CONTEXT GẦN ĐÂY:
{context}

USER INPUT MỚI NHẤT: "{user_input}"

NHIỆM VỤ: Dự đoán User sẽ phản hồi thế nào sau khi bạn trả lời.
Hãy dự đoán 2 thứ:

1. SENTIMENT (Tâm trạng của User):
   - positive (vui, hài lòng, thích thú)
   - neutral (bình thường, không cảm xúc)
   - negative (buồn, thất vọng, tức giận)

2. CONFIDENCE (Mức độ tự tin của dự đoán):
   - high (0.8-1.0) -> Rất chắc chắn
   - medium (0.5-0.8) -> Khá chắc chắn  
   - low (0.2-0.5) -> Không chắc chắn

OUTPUT FORMAT:
SENTIMENT: [positive/neutral/negative]
CONFIDENCE: [high/medium/low]
"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse response
            predicted_sentiment = "neutral"
            confidence = 0.5
            
            for line in response_text.split('\n'):
                if "SENTIMENT:" in line.upper():
                    sentiment = line.split("SENTIMENT:")[-1].strip().lower()
                    if sentiment in ["positive", "neutral", "negative"]:
                        predicted_sentiment = sentiment
                elif "CONFIDENCE:" in line.upper():
                    conf_level = line.split("CONFIDENCE:")[-1].strip().lower()
                    if conf_level == "high":
                        confidence = random.uniform(0.8, 1.0)
                    elif conf_level == "medium":
                        confidence = random.uniform(0.5, 0.8)
                    elif conf_level == "low":
                        confidence = random.uniform(0.2, 0.5)
            
            # Lưu dự đoán vào lịch sử
            prediction_record = {
                'timestamp': time.time(),
                'user_input': user_input,
                'predicted_sentiment': predicted_sentiment,
                'confidence': confidence,
                'upt_state': upt_metrics.copy()
            }
            
            self.prediction_history.append(prediction_record)
            
            # Giữ chỉ 50 dự đoán gần nhất
            if len(self.prediction_history) > 50:
                self.prediction_history = self.prediction_history[-50:]
            
            return predicted_sentiment, confidence
            
        except Exception as e:
            print(f"[Prediction Error] Error predicting response: {e}")
            return self._fallback_prediction(user_input, upt_metrics)
    
    def calculate_surprise(self, predicted_sentiment, actual_user_feedback):
        """
        Tính toán Surprise khi User phản hồi khác với dự đoán.
        actual_user_feedback: "positive", "neutral", "negative" (từ rating 1/2)
        """
        # Mapping từ feedback sang sentiment
        sentiment_mapping = {
            "positive": "positive",    # User rating 1 (Tốt)
            "negative": "negative"     # User rating 2 (Cần sửa)
        }
        actual_sentiment = sentiment_mapping.get(actual_user_feedback, "neutral")
        
        # Tính Surprise dựa trên khoảng cách giữa dự đoán và thực tế
        if predicted_sentiment == actual_sentiment:
            surprise = 0.0  # Dự đoán đúng, không có surprise
        elif (predicted_sentiment == "positive" and actual_sentiment == "negative") or \
             (predicted_sentiment == "negative" and actual_sentiment == "positive"):
            surprise = 1.0  # Dự đoán sai hoàn toàn
        else:
            surprise = 0.5  # Dự đoán sai một phần
        
        # Lưu vào buffer
        surprise_record = {
            'timestamp': time.time(),
            'predicted': predicted_sentiment,
            'actual': actual_sentiment,
            'surprise': surprise
        }
        
        self.surprise_buffer.append(surprise_record)
        
        # Giữ chỉ 20 surprise gần nhất
        if len(self.surprise_buffer) > 20:
            self.surprise_buffer = self.surprise_buffer[-20:]
        
        # Cập nhật learning rate multiplier
        # Surprise cao -> learning rate cao (học nhanh hơn)
        self.learning_rate_multiplier = 1.0 + (surprise * 2.0)
        
        return surprise
    
    def get_learning_rate_multiplier(self):
        """
        Lấy hệ số nhân learning rate dựa trên surprise gần đây.
        Surprise cao -> học nhanh hơn.
        """
        if len(self.surprise_buffer) == 0:
            return 1.0
        
        # Tính trung bình surprise gần đây
        recent_surprises = [s['surprise'] for s in self.surprise_buffer[-5:]]
        avg_surprise = np.mean(recent_surprises)
        
        # Learning rate multiplier: 1.0 (normal) đến 3.0 (very surprised)
        return 1.0 + (avg_surprise * 2.0)
    
    def should_adjust_pulse(self, surprise):
        """
        Quyết định có nên điều chỉnh Pulse dựa trên surprise không.
        Surprise cao -> Pulse dao động mạnh (sốc/ngạc nhiên).
        """
        if surprise > 0.8:  # Rất ngạc nhiên
            return random.uniform(-3.0, 3.0)  # Dao động mạnh
        elif surprise > 0.5:  # Hơi ngạc nhiên
            return random.uniform(-1.5, 1.5)  # Dao động vừa
        elif surprise > 0.0:  # Hơi ngạc nhiên một chút
            return random.uniform(-0.5, 0.5)  # Dao động nhẹ
        else:  # Không ngạc nhiên
            return 0.0
    
    def _fallback_prediction(self, user_input, upt_metrics):
        """Fallback khi không có AI model"""
        pulse = upt_metrics.get('Pulse', 0)
        
        # Dự đoán đơn giản dựa trên Pulse
        if pulse > 2:
            return "positive", 0.6
        elif pulse < -2:
            return "negative", 0.6
        else:
            return "neutral", 0.5
    
    def get_prediction_accuracy(self):
        """Tính độ chính xác dự đoán gần đây"""
        if len(self.surprise_buffer) < 5:
            return 0.5  # Không đủ dữ liệu
        
        # Accuracy = 1 - average_surprise
        recent_surprises = [s['surprise'] for s in self.surprise_buffer[-10:]]
        avg_surprise = np.mean(recent_surprises)
        accuracy = 1.0 - avg_surprise
        
        return max(0.0, min(1.0, accuracy))
    
    def get_recent_predictions(self, limit=5):
        """Lấy các dự đoán gần đây"""
        return self.prediction_history[-limit:]

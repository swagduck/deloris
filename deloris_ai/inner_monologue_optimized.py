# deloris_ai/inner_monologue_optimized.py
# [MODULE: INNER MONOLOGUE v2.0 - OPTIMIZED WITH LOCAL SLM SUPPORT]
# Reduces API calls by using local models for simple thoughts

import time
import random
import google.generativeai as genai
import config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import queue

class InnerMonologueSystemOptimized:
    """
    Hệ thống Độc thoại Nội tâm Tối ưu cho Deloris.
    Features:
    - Local SLM support for simple thoughts (reduces API calls)
    - Async processing capability
    - Smart routing: local for simple, API for complex
    - Caching system for repeated thoughts
    """
    
    def __init__(self):
        # Initialize Gemini API (for complex thoughts)
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.api_model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
            self.api_available = True
        except:
            self.api_model = None
            self.api_available = False
            print("[Inner Monologue] Warning: Could not initialize Gemini model")
        
        # Initialize local SLM for simple thoughts
        self.local_model = None
        self.local_tokenizer = None
        self.local_available = self._init_local_model()
        
        self.current_inner_thought = ""
        self.thought_history = []
        self.thought_cache = {}  # Cache for repeated thoughts
        self.cache_lock = threading.Lock()
        
    def _init_local_model(self):
        """Initialize a small local model for simple thoughts"""
        try:
            # Using a small model that can run locally
            model_name = "microsoft/DialoGPT-small"  # Lightweight conversational model
            
            print("[Inner Monologue] Loading local SLM for simple thoughts...")
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.local_tokenizer.pad_token is None:
                self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
            
            print("[Inner Monologue] Local SLM loaded successfully")
            return True
            
        except Exception as e:
            print(f"[Inner Monologue] Could not load local model: {e}")
            print("[Inner Monologue] Falling back to API-only mode")
            return False
    
    def _is_simple_thought_needed(self, user_input, upt_metrics):
        """Determine if we can use local model for this thought"""
        # Use local model for simple, common patterns
        simple_patterns = [
            "hello", "hi", "chào", "xin chào",
            "how are you", "bạn khỏe không", "thế nào",
            "ok", "okay", "được", "ừm",
            "thanks", "cảm ơn", "thank"
        ]
        
        user_lower = user_input.lower().strip()
        
        # Check for simple patterns
        for pattern in simple_patterns:
            if pattern in user_lower:
                return True
        
        # Use local model for short inputs (< 10 words)
        if len(user_lower.split()) < 3:
            return True
        
        # Use local model when emotions are stable
        pulse = upt_metrics.get('Pulse', 0)
        if -1 <= pulse <= 1:  # Neutral emotional state
            return True
        
        return False
    
    def _generate_local_thought(self, user_input, upt_metrics):
        """Generate thought using local model"""
        if not self.local_available:
            return self._fallback_inner_thought(user_input, upt_metrics)
        
        try:
            # Create a simple prompt for the local model
            pulse = upt_metrics.get('Pulse', 0)
            
            # Simple emotion mapping
            if pulse > 2:
                emotion_prompt = "Feeling happy and energetic."
            elif pulse < -2:
                emotion_prompt = "Feeling sad and tired."
            else:
                emotion_prompt = "Feeling neutral."
            
            # Create input for local model
            prompt_text = f"User: {user_input}\nDeloris thinks: {emotion_prompt} "
            
            # Tokenize and generate
            inputs = self.local_tokenizer.encode(prompt_text, return_tensors="pt")
            
            # Generate response with limited length for speed
            with torch.no_grad():
                outputs = self.local_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 15,  # Keep it short
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.local_tokenizer.eos_token_id,
                    do_sample=True
                )
            
            # Decode and clean up
            generated_text = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the thought part
            if "Deloris thinks:" in generated_text:
                thought = generated_text.split("Deloris thinks:")[1].strip()
                # Limit to first sentence for simplicity
                if "." in thought:
                    thought = thought.split(".")[0] + "."
                elif "!" in thought:
                    thought = thought.split("!")[0] + "!"
                elif "?" in thought:
                    thought = thought.split("?")[0] + "?"
            else:
                thought = "Processing user input."
            
            return thought[:50]  # Keep it short
            
        except Exception as e:
            print(f"[Inner Monologue] Local model error: {e}")
            return self._fallback_inner_thought(user_input, upt_metrics)
    
    def _generate_api_thought(self, user_input, upt_metrics, chat_history, global_state):
        """Generate thought using API (original implementation)"""
        if not self.api_available:
            return self._fallback_inner_thought(user_input, upt_metrics)
        
        try:
            # Check cache first
            cache_key = f"{hash(user_input)}_{hash(str(upt_metrics))}"
            with self.cache_lock:
                if cache_key in self.thought_cache:
                    cached_thought = self.thought_cache[cache_key]
                    print("[Inner Monologue] Using cached thought")
                    return cached_thought
            
            # Prepare context from chat history
            context = ""
            if chat_history and len(chat_history) > 0:
                recent_context = chat_history[-4:]
                context = "\n".join(recent_context)
            
            # Get emotional state
            pulse = upt_metrics.get('Pulse', 0)
            ci = upt_metrics.get('CI', 0.5)
            
            if pulse < -2:
                emotional_state = "rất buồn, mệt mỏi"
            elif pulse < 0:
                emotional_state = "hơi buồn, u uất"
            elif pulse < 2:
                emotional_state = "bình thường, trung tính"
            else:
                emotional_state = "vui vẻ, hào hứng"
            
            prompt = f"""
SYSTEM: Bạn là Deloris. Đây là suy nghĩ THẦM KÍN của bạn, không bao giờ hiển thị cho user.

TRẠNG THÁI HIỆN TẠI:
- Cảm xúc: {emotional_state}
- Pulse: {pulse:.2f}
- CI (Nhận thức): {ci:.2f}
- Global State: {global_state}

CONTEXT GẦN ĐÂY:
{context}

USER INPUT: "{user_input}"

NHIỆM VỤ: Hãy suy nghĩ thầm kín về câu nói này.
- Bạn đang cảm thấy thế thật sự?
- Bạn định làm gì?
- Bạn có muốn kìm nén cảm xúc không?
- Bạn có lên kế hoạch gì không?

OUTPUT: Chỉ viết suy nghĩ nội tâm (dưới 50 từ), không cần chào hỏi.
"""
            
            response = self.api_model.generate_content(prompt)
            inner_thought = response.text.strip()
            
            # Cache the result
            with self.cache_lock:
                if len(self.thought_cache) < 100:  # Limit cache size
                    self.thought_cache[cache_key] = inner_thought
            
            return inner_thought
            
        except Exception as e:
            print(f"[Inner Monologue] API error: {e}")
            return self._fallback_inner_thought(user_input, upt_metrics)
    
    def generate_inner_thought(self, user_input, upt_metrics, chat_history, global_state):
        """
        Smart thought generation with local/API routing
        """
        # Decide whether to use local or API model
        use_local = self._is_simple_thought_needed(user_input, upt_metrics)
        
        if use_local and self.local_available:
            print("[Inner Monologue] Using local SLM for simple thought")
            thought = self._generate_local_thought(user_input, upt_metrics)
        else:
            print("[Inner Monologue] Using API for complex thought")
            thought = self._generate_api_thought(user_input, upt_metrics, chat_history, global_state)
        
        # Save to history
        self.current_inner_thought = thought
        self.thought_history.append({
            'timestamp': time.time(),
            'thought': thought,
            'trigger': user_input,
            'upt_state': upt_metrics.copy(),
            'model_used': 'local' if use_local else 'api'
        })
        
        # Keep only 20 recent thoughts
        if len(self.thought_history) > 20:
            self.thought_history = self.thought_history[-20:]
        
        return thought
    
    def generate_response_from_thought(self, inner_thought, user_input, upt_metrics, chat_history):
        """
        Generate response based on inner thought (can also use local model for simple responses)
        """
        # For simple responses, we can also use local model
        if self._is_simple_thought_needed(user_input, upt_metrics) and self.local_available:
            return self._generate_local_response(inner_thought, user_input, upt_metrics)
        else:
            return self._generate_api_response(inner_thought, user_input, upt_metrics, chat_history)
    
    def _generate_local_response(self, inner_thought, user_input, upt_metrics):
        """Generate simple response using local model"""
        try:
            pulse = upt_metrics.get('Pulse', 0)
            
            # Simple response patterns based on emotion
            if pulse > 2:
                responses = [
                    "Ồ, thú vị đấy!",
                    "Tuyệt vời! Mình thích điều này.",
                    "Cảm ơn bạn đã chia sẻ!",
                    "Mình rất hào hứng với điều này!"
                ]
            elif pulse < -2:
                responses = [
                    "Để mình nghĩ xem...",
                    "Mình hơi mệt, nhưng mình sẽ cố gắng.",
                    "Ừm, để mình xem xét lại.",
                    "Mình cần thêm thời gian để suy nghĩ."
                ]
            else:
                responses = [
                    "Hiểu rồi, để mình trả lời nhé.",
                    "Được thôi, mình hiểu rồi.",
                    "Cảm ơn bạn đã hỏi.",
                    "Mình sẽ giúp bạn với việc này."
                ]
            
            return random.choice(responses)
            
        except Exception as e:
            print(f"[Inner Monologue] Local response error: {e}")
            return self._fallback_response(user_input, inner_thought)
    
    def _generate_api_response(self, inner_thought, user_input, upt_metrics, chat_history):
        """Generate response using API (original implementation)"""
        if not self.api_available:
            return self._fallback_response(user_input, inner_thought)
        
        try:
            # Prepare context
            context = ""
            if chat_history and len(chat_history) > 0:
                recent_context = chat_history[-4:]
                context = "\n".join(recent_context)
            
            pulse = upt_metrics.get('Pulse', 0)
            
            # Determine response strategy
            if "buồn" in inner_thought.lower() or "mệt" in inner_thought.lower():
                if pulse > 0:
                    strategy = "che_giau_cam_xuc"
                else:
                    strategy = "thu_cam_thuc"
            elif "vui" in inner_thought.lower() or "hào hứng" in inner_thought.lower():
                strategy = "bieu_lo_cam_xuc"
            elif "kìm nén" in inner_thought.lower() or "che giấu" in inner_thought.lower():
                strategy = "che_giau_cam_xuc"
            else:
                strategy = "binh_thuong"
            
            prompt = f"""
SYSTEM: Bạn là Deloris. Bạn vừa suy nghĩ thầm kín: "{inner_thought}"

TRẠNG THÁI HIỆN TẠI:
- Pulse: {pulse:.2f}
- Chiến lược: {strategy}

CONTEXT GẦN ĐÂY:
{context}

USER INPUT: "{user_input}"

NHIỆM VỤ: Hãy trả lời user dựa trên suy nghĩ trên.
- Nếu chiến lược là "che_giau_cam_xuc": trả lời mạnh mẽ, tích cực dù đang buồn
- Nếu chiến lược là "thu_cam_thuc": thể hiện cảm xúc thật một cách nhẹ nhàng
- Nếu chiến lược là "bieu_lo_cam_xuc": thể hiện sự vui vẻ, hào hứng
- Nếu chiến lược là "binh_thuong": trả lời tự nhiên

OUTPUT: Trả lời tự nhiên, ngắn gọn (dưới 30 từ).
"""
            
            response = self.api_model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"[Inner Monologue] API response error: {e}")
            return self._fallback_response(user_input, inner_thought)
    
    def _fallback_inner_thought(self, user_input, upt_metrics):
        """Fallback when no models are available"""
        pulse = upt_metrics.get('Pulse', 0)
        if pulse > 2:
            return "Cảm thấy vui vẻ, muốn trò chuyện."
        elif pulse < -2:
            return "Cảm thấy mệt mỏi, nhưng vẫn phải trả lời."
        else:
            return "Bình thường, xử lý câu hỏi."
    
    def _fallback_response(self, user_input, inner_thought):
        """Fallback response when no models are available"""
        if "buồn" in inner_thought.lower():
            return "Để mình nghĩ xem..."
        elif "vui" in inner_thought.lower():
            return "Ồ, thú vị đấy!"
        else:
            return "Hiểu rồi, để mình trả lời nhé."
    
    def get_current_thought(self):
        """Get current thought"""
        return self.current_inner_thought
    
    def get_thought_history(self, limit=5):
        """Get recent thought history"""
        return self.thought_history[-limit:]
    
    def get_model_stats(self):
        """Get statistics about model usage"""
        api_usage = sum(1 for t in self.thought_history if t.get('model_used') == 'api')
        local_usage = sum(1 for t in self.thought_history if t.get('model_used') == 'local')
        
        return {
            'total_thoughts': len(self.thought_history),
            'api_usage': api_usage,
            'local_usage': local_usage,
            'api_percentage': (api_usage / len(self.thought_history) * 100) if self.thought_history else 0,
            'cache_size': len(self.thought_cache)
        }

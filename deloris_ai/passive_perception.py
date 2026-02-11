# deloris_ai/passive_perception.py
# [MODULE: PASSIVE PERCEPTION]
# Automatic screen monitoring and context awareness

import time
import threading
import random
import pyautogui
import os
from PIL import Image
from datetime import datetime
import json
from typing import Optional, Dict, Any
from .vision import deloris_eye

class PassivePerceptionSystem:
    def __init__(self, upload_dir: str, check_interval: int = 300):
        """
        Initialize passive perception system
        
        Args:
            upload_dir: Directory to save temporary screenshots
            check_interval: Seconds between automatic screen checks (default: 5 minutes)
        """
        self.upload_dir = upload_dir
        self.check_interval = check_interval
        self.is_running = False
        self.thread = None
        self.last_context = ""
        self.context_history = []
        self.user_activity_detected = False
        self.last_activity_time = time.time()
        self.lock = threading.Lock()
        
        # Activity thresholds
        self.activity_timeout = 60  # seconds of inactivity before passive mode
        self.max_context_history = 10
        
        # Ensure upload directory exists
        os.makedirs(upload_dir, exist_ok=True)
        
        print(f"👁️ [PASSIVE PERCEPTION] Initialized with {check_interval}s interval")
    
    def start_monitoring(self):
        """Start passive monitoring in background thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        print("🔍 [PASSIVE PERCEPTION] Started background monitoring")
    
    def stop_monitoring(self):
        """Stop passive monitoring"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("🛑 [PASSIVE PERCEPTION] Stopped monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background"""
        while self.is_running:
            try:
                # Check if user is inactive
                if self._is_user_inactive():
                    self._perform_passive_glance()
                
                # Sleep for check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"⚠️ [PASSIVE PERCEPTION] Monitoring error: {e}")
                time.sleep(30)  # Wait before retry
    
    def _is_user_inactive(self) -> bool:
        """Check if user has been inactive for threshold time"""
        current_time = time.time()
        inactive_time = current_time - self.last_activity_time
        return inactive_time > self.activity_timeout
    
    def _perform_passive_glance(self):
        """Perform a passive screen glance"""
        try:
            print("👀 [PASSIVE PERCEPTION] Taking a glance at the screen...")
            
            # Take screenshot
            screenshot_path = self._take_screenshot()
            if not screenshot_path:
                return
            
            # Analyze screenshot with vision system
            if deloris_eye.is_ready:
                analysis = self._analyze_screen_context(screenshot_path)
                if analysis:
                    self._process_context(analysis, screenshot_path)
            
            # Clean up temporary screenshot
            try:
                os.remove(screenshot_path)
            except:
                pass
                
        except Exception as e:
            print(f"⚠️ [PASSIVE PERCEPTION] Glance error: {e}")
    
    def _take_screenshot(self) -> Optional[str]:
        """Take a screenshot and save to temporary file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"passive_glance_{timestamp}.png"
            filepath = os.path.join(self.upload_dir, filename)
            
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot.save(filepath)
            
            print(f"📸 [PASSIVE PERCEPTION] Screenshot saved: {filename}")
            return filepath
            
        except Exception as e:
            print(f"⚠️ [PASSIVE PERCEPTION] Screenshot failed: {e}")
            return None
    
    def _analyze_screen_context(self, image_path: str) -> Optional[str]:
        """Analyze screenshot for context"""
        try:
            # Use vision system to analyze what's on screen
            prompt = """Analyze this screen image and provide a brief, helpful context about what the user is doing. 
            Focus on:
            1. What applications are visible?
            2. What kind of work is being done?
            3. Any visible problems, errors, or challenges?
            4. Overall activity type (coding, browsing, document work, etc.)
            
            Keep the analysis concise and actionable."""
            
            analysis = deloris_eye.analyze_image(image_path, prompt)
            return analysis
            
        except Exception as e:
            print(f"⚠️ [PASSIVE PERCEPTION] Analysis failed: {e}")
            return None
    
    def _process_context(self, analysis: str, screenshot_path: str):
        """Process analyzed context and determine if action is needed"""
        with self.lock:
            # Store context in history
            context_entry = {
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'screenshot': screenshot_path
            }
            
            self.context_history.append(context_entry)
            
            # Limit history size
            if len(self.context_history) > self.max_context_history:
                self.context_history.pop(0)
            
            self.last_context = analysis
        
        print(f"🧠 [PASSIVE PERCEPTION] Context: {analysis}")
        
        # Determine if proactive assistance is needed
        assistance_needed = self._evaluate_assistance_need(analysis)
        if assistance_needed:
            self._trigger_proactive_assistance(analysis, assistance_needed)
    
    def _evaluate_assistance_need(self, analysis: str) -> Optional[str]:
        """Evaluate if user needs proactive assistance"""
        analysis_lower = analysis.lower()
        
        # Check for common problems
        if any(keyword in analysis_lower for keyword in ['error', 'exception', 'failed', 'bug', 'crash']):
            return "error_detected"
        
        if any(keyword in analysis_lower for keyword in ['debug', 'stuck', 'confused', 'problem']):
            return "debug_help"
        
        if any(keyword in analysis_lower for keyword in ['code', 'programming', 'development']):
            return "coding_context"
        
        if any(keyword in analysis_lower for keyword in ['document', 'writing', 'text']):
            return "writing_context"
        
        return None
    
    def _trigger_proactive_assistance(self, analysis: str, assistance_type: str):
        """Trigger proactive assistance based on context"""
        assistance_messages = {
            "error_detected": "Tôi thấy bạn đang gặp lỗi trên màn hình. Có cần tôi giúp phân tích và sửa lỗi không?",
            "debug_help": "Có vẻ như bạn đang gặp khó khăn trong việc debug. Tôi có thể gợi ý một vài cách tiếp cận nếu bạn muốn.",
            "coding_context": "Tôi thấy bạn đang lập trình. Nếu cần bất kỳ trợ giúp nào về code, đừng ngần ngại hỏi nhé!",
            "writing_context": "Bạn đang làm việc với tài liệu. Nếu cần giúp đỡ về viết lách hoặc chỉnh sửa, tôi ở đây."
        }
        
        message = assistance_messages.get(assistance_type, "Tôi thấy bạn đang bận rộn. Có cần tôi giúp gì không?")
        
        # Store proactive message for delivery
        self._store_proactive_message(message, analysis, assistance_type)
    
    def _store_proactive_message(self, message: str, context: str, assistance_type: str):
        """Store proactive message for delivery to user"""
        proactive_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'context': context,
            'type': assistance_type,
            'delivered': False
        }
        
        # This would be integrated with the main chat system
        print(f"💬 [PASSIVE PERCEPTION] Proactive message ready: {message}")
        
        # For now, just log it - integration with chat system would be needed
        # to actually deliver this message to the user
    
    def update_user_activity(self):
        """Call this when user activity is detected"""
        self.last_activity_time = time.time()
        self.user_activity_detected = True
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context and history"""
        with self.lock:
            return {
                'last_context': self.last_context,
                'context_history': self.context_history[-5:],  # Last 5 entries
                'is_monitoring': self.is_running,
                'last_activity': datetime.fromtimestamp(self.last_activity_time).isoformat(),
                'user_inactive': self._is_user_inactive()
            }
    
    def force_glance(self):
        """Force an immediate screen glance"""
        if self.is_running:
            threading.Thread(target=self._perform_passive_glance, daemon=True).start()

# Global instance
passive_perception = None

def init_passive_perception(upload_dir: str, check_interval: int = 300):
    """Initialize passive perception system"""
    global passive_perception
    passive_perception = PassivePerceptionSystem(upload_dir, check_interval)
    return passive_perception

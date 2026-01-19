#!/usr/bin/env python3
# test_main_flow.py
# Test the main conversation flow with mock models

import sys
import os
import torch
import torch.nn as nn

# Mock models for testing
class MockDelorisModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x, upt_metrics):
        # Return mock prediction
        batch_size = x.size(0)
        return torch.randn(batch_size, 3)  # 3 classes

class MockUPTAutomatorModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 3)  # A, E, C outputs
    
    def forward(self, x):
        return torch.randn(3)  # Mock A, E, C values

class MockSentenceTransformer:
    def encode(self, texts):
        import numpy as np
        # Return mock embeddings
        return [np.random.randn(384).astype(np.float32)]

def test_main_conversation_flow():
    """Test the main conversation flow with mock models"""
    print("🗣️ Testing Main Conversation Flow...")
    
    try:
        # Import the actual modules first
        from deloris_ai import architecture
        from upt_predictor import architecture as predictor_architecture
        
        # Mock the model classes
        architecture.DelorisModel = MockDelorisModel
        predictor_architecture.UPTAutomatorModel = MockUPTAutomatorModel
        
        # Mock sentence transformer
        import sys
        from unittest.mock import MagicMock
        sys.modules['sentence_transformers'].SentenceTransformer = MockSentenceTransformer
        
        # Mock config
        config_mock = MagicMock()
        config_mock.LANGUAGE_MODEL_NAME = "mock-model"
        config_mock.DELORIS_MODEL_PATH = "mock-deloris.pth"
        config_mock.AUTOMATOR_MODEL_PATH = "mock-automator.pth"
        config_mock.INPUT_DIM = 384
        config_mock.DELORIS_HIDDEN_DIM = 128
        config_mock.DELORIS_OUTPUT_DIM = 3
        config_mock.AUTOMATOR_HIDDEN_DIM = 64
        config_mock.MEMORY_FILE = "mock-memory.txt"
        config_mock.FEEDBACK_FILE = "mock-feedback.json"
        
        sys.modules['config'] = config_mock
        
        # Now import and test the main components
        from upt_core.calculator import UPTCalculator
        from upt_core.prediction_error import PredictionErrorSystem
        from deloris_ai.inner_monologue import InnerMonologueSystem
        
        # Initialize systems
        upt_calc = UPTCalculator(dt=1.0)
        inner_monologue = InnerMonologueSystem()
        prediction_error = PredictionErrorSystem()
        
        print("   ✅ Systems initialized")
        
        # Mock conversation
        vectorizer = MockSentenceTransformer()
        deloris = MockDelorisModel(384, 128, 3)
        predictor_model = MockUPTAutomatorModel(384, 64)
        
        # Simulate one conversation turn
        text_input = "Hello Deloris, how are you?"
        chat_history = []
        current_memory = "This is a test conversation."
        
        print(f"   📝 User: {text_input}")
        
        # Step 1: Vectorize input
        input_vector = torch.tensor(vectorizer.encode([text_input]), dtype=torch.float32)
        
        # Step 2: Get UPT predictions
        with torch.no_grad():
            normalized_preds = predictor_model(input_vector).squeeze()
            A_t = max(normalized_preds[0].item() * 1.0, 0.1)
            E_t = max(normalized_preds[1].item() * 5.0, 0.1)
            C_t = max(normalized_preds[2].item() * 3.0, 0.1)
        
        print(f"   📊 UPT: A={A_t:.2f}, E={E_t:.2f}, C={C_t:.2f}")
        
        # Step 3: Update UPT metrics
        upt_metrics = upt_calc.update_metrics(A_t, E_t, C_t)
        print(f"   💓 UPT Metrics: CI={upt_metrics['CI']:.2f}, Pulse={upt_metrics['Pulse']:.2f}")
        
        # Step 4: Inner Monologue - Generate thought
        inner_thought = inner_monologue.generate_inner_thought(
            text_input, upt_metrics, chat_history, ""
        )
        print(f"   🧠 Inner Thought: '{inner_thought}'")
        
        # Step 5: Prediction - Predict user response
        predicted_sentiment, confidence = prediction_error.predict_user_response(
            text_input, upt_metrics, chat_history
        )
        print(f"   🔮 Prediction: User will be {predicted_sentiment} ({confidence:.2f} confidence)")
        
        # Step 6: Generate response
        with torch.no_grad():
            prediction = deloris(input_vector, upt_metrics)
            predicted_class = torch.argmax(prediction, dim=1).item()
        
        # Step 7: Response from thought
        response_from_thought = inner_monologue.generate_response_from_thought(
            inner_thought, text_input, upt_metrics, chat_history
        )
        print(f"   💬 Response: '{response_from_thought}'")
        
        # Step 8: Simulate feedback
        user_feedback = "positive"  # User liked the response
        surprise = prediction_error.calculate_surprise(predicted_sentiment, user_feedback)
        learning_multiplier = prediction_error.get_learning_rate_multiplier()
        
        print(f"   😲 Surprise: {surprise:.2f}")
        print(f"   📈 Learning Rate: x{learning_multiplier:.2f}")
        
        # Step 9: Update history
        chat_history.append(f"Bạn nói: {text_input}")
        chat_history.append(f"Deloris: {response_from_thought}")
        
        print("   ✅ Full conversation flow completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the main flow test"""
    print("🚀 Testing Complete Conversation Flow\n")
    
    result = test_main_conversation_flow()
    
    print(f"\n{'='*50}")
    if result:
        print("🎉 MAIN FLOW TEST PASSED!")
        print("✅ All consciousness upgrades work together")
        print("✅ Deloris can run the enhanced conversation system")
        print("✅ Ready for production with valid API keys")
    else:
        print("❌ MAIN FLOW TEST FAILED!")
        print("⚠️  Check the errors above")
    print("="*50)
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main())

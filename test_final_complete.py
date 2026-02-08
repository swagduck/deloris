#!/usr/bin/env python3

print("Testing final complete fix...")

import asyncio
import torch
from sentence_transformers import SentenceTransformer
from upt_predictor.compatibility import UPTAutomatorModelCompat
from deloris_ai.architecture import DelorisModel
from upt_core.prediction_error import PredictionErrorSystem
import config

async def test_complete_processing():
    print("Loading vectorizer...")
    vectorizer = SentenceTransformer(config.LANGUAGE_MODEL_NAME)
    
    print("Creating models...")
    predictor_model = UPTAutomatorModelCompat(
        config.INPUT_DIM, 
        config.IMAGE_VECTOR_DIM, 
        config.AUTOMATOR_HIDDEN_DIM
    )
    
    deloris = DelorisModel(config.INPUT_DIM, config.DELORIS_HIDDEN_DIM, config.DELORIS_OUTPUT_DIM)
    
    print("Loading weights...")
    predictor_model.load_state_dict(torch.load(config.AUTOMATOR_MODEL_PATH))
    predictor_model.eval()
    
    deloris.load_state_dict(torch.load(config.DELORIS_MODEL_PATH))
    deloris.eval()
    
    print("Creating prediction error system...")
    prediction_error = PredictionErrorSystem()
    
    print("Testing complete processing...")
    text_input = "hello"
    
    # Test inputs
    test_input_774 = torch.tensor(vectorizer.encode([text_input]), dtype=torch.float32)
    test_input_384 = torch.tensor(vectorizer.encode([text_input]), dtype=torch.float32)
    
    # Pad UPT input to 774 dimensions
    if test_input_774.shape[1] < 774:
        padding = torch.zeros(1, 774 - test_input_774.shape[1])
        test_input_774 = torch.cat([test_input_774, padding], dim=1)
    
    print(f"UPT input shape: {test_input_774.shape}")
    print(f"Deloris input shape: {test_input_384.shape}")
    
    # Test async functions
    async def process_upt_prediction_async(input_vector):
        """Async UPT prediction"""
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            normalized_preds = await loop.run_in_executor(
                None, lambda: predictor_model(input_vector).squeeze()
            )
            if normalized_preds.dim() == 2 and normalized_preds.shape[1] == 3:
                A_norm, E_norm, C_norm = normalized_preds[0, 0], normalized_preds[0, 1], normalized_preds[0, 2]
            else:
                A_norm, E_norm, C_norm = normalized_preds[0], normalized_preds[1], normalized_preds[2]
            
            A_t = max(A_norm.item() * 1.0, 0.1)
            E_t = max(E_norm.item() * 5.0, 0.1)
            C_t = max(C_norm.item() * 3.0, 0.1)
            return A_t, E_t, C_t

    async def process_deloris_classification_async(input_vector, upt_metrics):
        """Async Deloris classification"""
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            prediction = await loop.run_in_executor(
                None, lambda: deloris(input_vector, upt_metrics)
            )
            predicted_class = torch.argmax(prediction, dim=1).item()
        return predicted_class

    async def process_sentiment_prediction_async(text_input, upt_metrics, chat_history):
        """Async sentiment prediction"""
        loop = asyncio.get_event_loop()
        predicted_sentiment, confidence = await loop.run_in_executor(
            None, lambda: prediction_error.predict_user_response(text_input, upt_metrics, chat_history)
        )
        return predicted_sentiment, confidence
    
    # Test all async processing
    upt_task = process_upt_prediction_async(test_input_774)
    sentiment_task = process_sentiment_prediction_async(text_input, {}, [])
    deloris_task = process_deloris_classification_async(test_input_384, {})
    
    # First gather
    upt_results = await asyncio.gather(upt_task, sentiment_task)
    upt_values = upt_results[0]
    sentiment_values = upt_results[1]
    A_t, E_t, C_t = upt_values
    predicted_sentiment, confidence = sentiment_values
    
    # Second gather
    async def mock_inner_thought():
        return "test thought"
    
    inner_results = await asyncio.gather(
        mock_inner_thought(),
        deloris_task
    )
    inner_thought = inner_results[0]
    predicted_class = inner_results[1]
    
    print(f"✅ UPT values: A={A_t:.2f}, E={E_t:.2f}, C={C_t:.2f}")
    print(f"✅ Sentiment: {predicted_sentiment} (confidence: {confidence:.2f})")
    print(f"✅ Inner thought: {inner_thought}")
    print(f"✅ Predicted class: {predicted_class}")
    print("✅ Complete test successful!")

if __name__ == "__main__":
    asyncio.run(test_complete_processing())

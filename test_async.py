#!/usr/bin/env python3

print("Testing async fix...")

import asyncio
import torch
from sentence_transformers import SentenceTransformer
from upt_predictor.compatibility import UPTAutomatorModelCompat
from upt_core.prediction_error import PredictionErrorSystem
import config

async def test_async_processing():
    print("Loading vectorizer...")
    vectorizer = SentenceTransformer(config.LANGUAGE_MODEL_NAME)
    
    print("Creating legacy model...")
    predictor_model = UPTAutomatorModelCompat(
        config.INPUT_DIM, 
        config.IMAGE_VECTOR_DIM, 
        config.AUTOMATOR_HIDDEN_DIM
    )
    
    print("Loading weights...")
    predictor_model.load_state_dict(torch.load(config.AUTOMATOR_MODEL_PATH))
    predictor_model.eval()
    
    print("Creating prediction error system...")
    prediction_error = PredictionErrorSystem()
    
    print("Testing async processing...")
    text_input = "hello"
    test_input = torch.tensor(vectorizer.encode([text_input]), dtype=torch.float32)
    
    # Pad to 774 dimensions
    if test_input.shape[1] < 774:
        padding = torch.zeros(1, 774 - test_input.shape[1])
        test_input = torch.cat([test_input, padding], dim=1)
    
    # Test the async functions
    async def process_upt_prediction_async(input_vector):
        """Async UPT prediction"""
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            normalized_preds = await loop.run_in_executor(
                None, lambda: predictor_model(input_vector).squeeze()
            )
            # Simple denormalization for test
            if normalized_preds.dim() == 2 and normalized_preds.shape[1] == 3:
                A_norm, E_norm, C_norm = normalized_preds[0, 0], normalized_preds[0, 1], normalized_preds[0, 2]
            else:
                A_norm, E_norm, C_norm = normalized_preds[0], normalized_preds[1], normalized_preds[2]
            
            A_t = max(A_norm.item() * 1.0, 0.1)
            E_t = max(E_norm.item() * 5.0, 0.1)
            C_t = max(C_norm.item() * 3.0, 0.1)
            return A_t, E_t, C_t

    async def process_sentiment_prediction_async(text_input, upt_metrics, chat_history):
        """Async sentiment prediction"""
        loop = asyncio.get_event_loop()
        predicted_sentiment, confidence = await loop.run_in_executor(
            None, lambda: prediction_error.predict_user_response(text_input, upt_metrics, chat_history)
        )
        return predicted_sentiment, confidence
    
    # Test the gather
    upt_task = process_upt_prediction_async(test_input)
    sentiment_task = process_sentiment_prediction_async(text_input, {}, [])
    
    upt_results = await asyncio.gather(upt_task, sentiment_task)
    
    # Unpack the results
    upt_values = upt_results[0]  # (A_t, E_t, C_t)
    sentiment_values = upt_results[1]  # (predicted_sentiment, confidence)
    
    A_t, E_t, C_t = upt_values
    predicted_sentiment, confidence = sentiment_values
    
    print(f"✅ UPT values: A={A_t:.2f}, E={E_t:.2f}, C={C_t:.2f}")
    print(f"✅ Sentiment: {predicted_sentiment} (confidence: {confidence:.2f})")
    print("✅ Async test successful!")

if __name__ == "__main__":
    asyncio.run(test_async_processing())

#!/usr/bin/env python3

print("Testing final fix...")

import torch
from sentence_transformers import SentenceTransformer
from upt_predictor.compatibility import UPTAutomatorModelCompat
import config

try:
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
    
    print("Testing forward pass...")
    test_input = torch.tensor(vectorizer.encode(["hello"]), dtype=torch.float32)
    
    # Pad to 774 dimensions
    if test_input.shape[1] < 774:
        padding = torch.zeros(1, 774 - test_input.shape[1])
        test_input = torch.cat([test_input, padding], dim=1)
    
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = predictor_model(test_input)
        print(f"✅ Model output shape: {output.shape}")
        print(f"✅ Model output values: {output}")
        
        # Test denormalization with the fixed function
        if output.shape[1] == 3:
            A_norm, E_norm, C_norm = output[0, 0], output[0, 1], output[0, 2]
            A_t = max(A_norm.item() * 1.0, 0.1)
            E_t = max(E_norm.item() * 5.0, 0.1)
            C_t = max(C_norm.item() * 3.0, 0.1)
            print(f"✅ Denormalized values: A={A_t:.2f}, E={E_t:.2f}, C={C_t:.2f}")
    
    print("✅ Final fix test successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

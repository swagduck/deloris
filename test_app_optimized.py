#!/usr/bin/env python3

print("Testing app_optimized fix...")

import asyncio
import torch
from sentence_transformers import SentenceTransformer
from deloris_ai.response_mapper import generate_final_response

async def test_app_optimized():
    print("Testing generate_final_response call...")
    
    # Test parameters
    predicted_class = 2
    text_input = "hello"
    current_memory = "Test memory"
    chat_history = ["Previous conversation"]
    response_from_thought = "Test thought"
    upt_metrics = {'CI': 0.5, 'Pulse': 0.0}
    relevant_memories = ["Memory 1", "Memory 2", "Memory 3"]
    
    try:
        # Test the function call with correct parameters (11 total)
        result = generate_final_response(
            predicted_class,        # 1st: strategy_class
            text_input,            # 2nd: user_message  
            relevant_memories,     # 3rd: retrieved_docs
            chat_history,           # 4th: chat_history
            0.5,                   # 5th: entanglement_level
            "neutral",              # 6th: persona
            response_from_thought,   # 7th: global_state
            upt_metrics.get('CI', 0.5),  # 8th: CI_value
            None,                  # 9th: proactive_report
            upt_metrics.get('Pulse', 0.0)   # 10th: pulse_value
            # No heartbeat_status parameter (11th - has default)
        )
        
        # Handle the return value properly
        if isinstance(result, tuple) and len(result) == 2:
            final_output_message, clean_response_text = result
        else:
            # Function returns single value or different format
            final_output_message = str(result)
            clean_response_text = str(result)
        
        print(f"✅ Final response: {clean_response_text}")
        print("✅ Function call test successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_app_optimized())

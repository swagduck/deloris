#!/usr/bin/env python3
# test_neuro_link.py
# Test script for Neuro-Link Dynamic System Prompting

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deloris_ai.heartbeat import HeartbeatSystem
from deloris_ai.response_mapper import _get_base_prompt

def test_neuro_link():
    """Test the Neuro-Link integration"""
    print("🧪 Testing Neuro-Link Dynamic System Prompting...")
    
    # Mock data
    queue = []
    global_state = {'Pulse': 2.0}  # Positive pulse
    chat_history = ["User: Hello", "Deloris: Hi there!"]
    
    # Create heartbeat system
    heartbeat = HeartbeatSystem(queue, global_state, chat_history)
    
    # Test different states
    test_states = [
        {"name": "Normal State", "pulse": 0.0},
        {"name": "High Energy", "pulse": 8.0},
        {"name": "Low Energy", "pulse": -4.0},
        {"name": "Very Happy", "pulse": 10.0},
        {"name": "Very Sad", "pulse": -5.0}
    ]
    
    for test in test_states:
        print(f"\n--- Testing {test['name']} (Pulse: {test['pulse']}) ---")
        
        # Update state
        global_state['Pulse'] = test['pulse']
        
        # Get status
        status = heartbeat.get_status()
        print(f"Energy: {status['energy']}%")
        print(f"Mood: {status['mood']}")
        print(f"Entropy: {status['entropy']}")
        
        # Generate prompt
        prompt = _get_base_prompt(
            strategy_desc="Logic & Phân tích",
            user_message="Bạn có khỏe không?",
            chat_history=chat_history,
            retrieved_docs=None,
            entanglement_level=0.5,
            persona="neutral",
            global_state="Test State",
            heartbeat_status=status
        )
        
        # Extract NEURO-LINK section
        lines = prompt.split('\n')
        neuro_link_section = []
        in_neuro_link = False
        
        for line in lines:
            if "[NEURO-LINK - TRẠNG THÁI THỰC TẾ]:" in line:
                in_neuro_link = True
                neuro_link_section.append(line)
            elif in_neuro_link and line.startswith('[NGUỒN DỮ LIỆU]:'):
                break
            elif in_neuro_link:
                neuro_link_section.append(line)
        
        print("NEURO-LINK Instructions:")
        for line in neuro_link_section:
            if line.strip():
                print(f"  {line}")
    
    print("\n✅ Neuro-Link test completed!")

if __name__ == "__main__":
    test_neuro_link()

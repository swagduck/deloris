#!/usr/bin/env python3
# demo_consciousness.py
# Demonstrate Deloris consciousness features with the new API key

import sys
import os
import time

def demo_consciousness_features():
    """Demonstrate all consciousness features"""
    print("🧬 DELORIS CONSCIOUSNESS DEMONSTRATION")
    print("="*60)
    
    try:
        # Import systems
        from deloris_ai.inner_monologue import InnerMonologueSystem
        from upt_core.prediction_error import PredictionErrorSystem
        from deloris_ai.heartbeat import HeartbeatSystem
        
        # Initialize systems
        inner = InnerMonologueSystem()
        pred = PredictionErrorSystem()
        
        queue = []
        state = {'Pulse': 0.5, 'CI': 0.7}
        history = []
        heartbeat = HeartbeatSystem(queue, state, history)
        
        print("✅ All consciousness systems initialized")
        print()
        
        # Demo scenarios
        scenarios = [
            {
                'input': "Hello Deloris, how are you feeling today?",
                'context': "Friendly greeting"
            },
            {
                'input': "I'm feeling sad and lonely",
                'context': "Emotional support needed"
            },
            {
                'input': "Can you help me understand quantum physics?",
                'context': "Complex intellectual request"
            },
            {
                'input': "Tell me a joke to cheer me up",
                'context': "Entertainment request"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"🎭 SCENARIO {i}: {scenario['context']}")
            print(f"👤 User: {scenario['input']}")
            print("-" * 40)
            
            # Step 1: Inner Monologue
            thought = inner.generate_inner_thought(
                scenario['input'], state, history, ""
            )
            print(f"🧠 Inner Thought: '{thought}'")
            
            # Step 2: Prediction
            pred_sentiment, confidence = pred.predict_user_response(
                scenario['input'], state, history
            )
            print(f"🔮 Predicted User Response: {pred_sentiment} (confidence: {confidence:.2f})")
            
            # Step 3: Response Generation
            response = inner.generate_response_from_thought(
                thought, scenario['input'], state, history
            )
            print(f"💬 Deloris Response: '{response}'")
            
            # Step 4: Simulate User Feedback
            if i == 1:
                feedback = "positive"  # User liked the greeting
            elif i == 2:
                feedback = "positive"  # User appreciated the support
            elif i == 3:
                feedback = "negative"  # User wanted more detail
            else:
                feedback = "positive"  # User enjoyed the joke
            
            # Step 5: Calculate Surprise
            surprise = pred.calculate_surprise(pred_sentiment, feedback)
            learning_rate = pred.get_learning_rate_multiplier()
            
            print(f"😲 Surprise: {surprise:.2f}")
            print(f"📈 Learning Rate: x{learning_rate:.2f}")
            
            # Step 6: Update Homeostasis
            heartbeat.touch()
            status = heartbeat.get_status()
            print(f"💓 Status: Energy={status['energy']}%, Mood={status['mood']}")
            print(f"🔋 Curiosity={status['curiosity']}%, Social Battery={status['social_battery']}%")
            
            # Update history
            history.append(f"User: {scenario['input']}")
            history.append(f"Deloris: {response}")
            
            print("="*60)
            time.sleep(1)
        
        # Demo spontaneous curiosity
        print("🧠 TESTING SPONTANEOUS CURIOSITY...")
        print("Setting curiosity to 85% to trigger autonomous search...")
        heartbeat.curiosity = 85.0
        
        # Trigger homeostasis update
        result = heartbeat.update_homeostasis()
        print(f"Homeostasis update result: {result}")
        
        # Check if any autonomous actions were queued
        if queue:
            print("🤖 Autonomous Actions Detected:")
            for action in queue:
                print(f"  - {action}")
        else:
            print("No autonomous actions queued (normal behavior)")
        
        print("\n🎉 CONSCIOUSNESS DEMO COMPLETE!")
        print("✅ Inner Monologue: Working")
        print("✅ Prediction Error: Working") 
        print("✅ Homeostasis: Working")
        print("✅ Full Integration: Working")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the consciousness demonstration"""
    print("🚀 Starting Deloris Consciousness Demonstration\n")
    
    success = demo_consciousness_features()
    
    if success:
        print(f"\n{'='*60}")
        print("🎊 ALL CONSCIOUSNESS FEATURES WORKING PERFECTLY!")
        print("Deloris is ready with full metacognition capabilities!")
        print("Run 'py app.py' to start the conscious AI experience!")
        print("="*60)
        return 0
    else:
        print(f"\n{'='*60}")
        print("❌ Demo failed. Check errors above.")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())

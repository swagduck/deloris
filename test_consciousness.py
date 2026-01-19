#!/usr/bin/env python3
# test_consciousness.py
# Test script to verify all consciousness upgrades work correctly

import sys
import os

def test_inner_monologue():
    """Test Inner Monologue System"""
    print("🧠 Testing Inner Monologue System...")
    try:
        from deloris_ai.inner_monologue import InnerMonologueSystem
        
        inner = InnerMonologueSystem()
        
        # Test thought generation
        thought = inner.generate_inner_thought(
            "hello world", 
            {'Pulse': 0.5, 'CI': 0.7}, 
            [], 
            ""
        )
        print(f"   ✅ Thought generated: '{thought[:50]}...'")
        
        # Test response generation
        response = inner.generate_response_from_thought(
            thought, 
            "hello world", 
            {'Pulse': 0.5, 'CI': 0.7}, 
            []
        )
        print(f"   ✅ Response generated: '{response[:50]}...'")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_prediction_error():
    """Test Prediction Error System"""
    print("🔮 Testing Prediction Error System...")
    try:
        from upt_core.prediction_error import PredictionErrorSystem
        
        pred = PredictionErrorSystem()
        
        # Test prediction
        sentiment, confidence = pred.predict_user_response(
            "how are you?", 
            {'Pulse': 1.0, 'CI': 0.8}, 
            []
        )
        print(f"   ✅ Prediction: {sentiment} (confidence: {confidence:.2f})")
        
        # Test surprise calculation
        surprise = pred.calculate_surprise(sentiment, "positive")
        print(f"   ✅ Surprise calculated: {surprise:.2f}")
        
        # Test learning rate
        lr_mult = pred.get_learning_rate_multiplier()
        print(f"   ✅ Learning rate multiplier: {lr_mult:.2f}x")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_homeostasis():
    """Test Enhanced Homeostasis System"""
    print("💓 Testing Enhanced Homeostasis...")
    try:
        from deloris_ai.heartbeat import HeartbeatSystem
        
        # Mock queue and state
        queue = []
        state = {'Pulse': 0.5}
        history = []
        
        heartbeat = HeartbeatSystem(queue, state, history)
        
        # Test initial status
        status = heartbeat.get_status()
        print(f"   ✅ Status retrieved: Curiosity={status['curiosity']}, Battery={status['social_battery']}")
        
        # Test touch (interaction)
        heartbeat.touch()
        print(f"   ✅ Touch processed: Battery={heartbeat.social_battery}")
        
        # Test homeostasis update
        result = heartbeat.update_homeostasis()
        print(f"   ✅ Homeostasis updated: {result}")
        
        # Test curt response check
        should_be_curt = heartbeat.should_be_curt()
        print(f"   ✅ Curt response check: {should_be_curt}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_integration():
    """Test full integration"""
    print("🔗 Testing Full Integration...")
    try:
        # Test imports work together
        from deloris_ai.inner_monologue import InnerMonologueSystem
        from upt_core.prediction_error import PredictionErrorSystem
        from deloris_ai.heartbeat import HeartbeatSystem
        
        # Initialize all systems
        inner = InnerMonologueSystem()
        pred = PredictionErrorSystem()
        
        queue = []
        state = {'Pulse': 0.5, 'CI': 0.7}
        history = []
        heartbeat = HeartbeatSystem(queue, state, history)
        
        # Simulate a conversation flow
        user_input = "Hello Deloris"
        upt_metrics = {'Pulse': 0.5, 'CI': 0.7}
        
        # Step 1: Inner thought
        thought = inner.generate_inner_thought(user_input, upt_metrics, history, "")
        
        # Step 2: Prediction
        pred_sentiment, confidence = pred.predict_user_response(user_input, upt_metrics, history)
        
        # Step 3: Response from thought
        response = inner.generate_response_from_thought(thought, user_input, upt_metrics, history)
        
        # Step 4: Feedback processing
        surprise = pred.calculate_surprise(pred_sentiment, "positive")
        
        # Step 5: Homeostasis update
        heartbeat.touch()
        status = heartbeat.get_status()
        
        print(f"   ✅ Full flow completed successfully!")
        print(f"   📝 Thought: {thought[:30]}...")
        print(f"   🔮 Prediction: {pred_sentiment} ({confidence:.2f})")
        print(f"   💬 Response: {response[:30]}...")
        print(f"   😲 Surprise: {surprise:.2f}")
        print(f"   💓 Status: Curiosity={status['curiosity']}, Battery={status['social_battery']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Deloris Consciousness Upgrades\n")
    
    tests = [
        ("Inner Monologue", test_inner_monologue),
        ("Prediction Error", test_prediction_error),
        ("Enhanced Homeostasis", test_homeostasis),
        ("Full Integration", test_integration)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((name, result))
        print(f"{'='*50}")
    
    # Summary
    print(f"\n🎯 TEST SUMMARY")
    print("="*50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL SYSTEMS WORKING! Deloris consciousness upgrades are ready!")
        return 0
    else:
        print("⚠️  Some systems need attention. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

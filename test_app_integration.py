#!/usr/bin/env python3
# test_app_integration.py
# Test that the main app.py can import and initialize all systems

import sys
import os

def test_app_imports():
    """Test that app.py can import all consciousness systems"""
    print("🔧 Testing App Integration...")
    try:
        # Test importing app module components
        import torch
        from deloris_ai.architecture import DelorisModel
        from upt_core.calculator import UPTCalculator
        from upt_core.prediction_error import PredictionErrorSystem
        from deloris_ai.inner_monologue import InnerMonologueSystem
        
        print("   ✅ All imports successful")
        
        # Test system initialization
        inner_monologue = InnerMonologueSystem()
        prediction_error = PredictionErrorSystem()
        
        print("   ✅ Systems initialize correctly")
        
        # Test basic functionality without models
        test_input = "hello world"
        test_upt = {'Pulse': 0.5, 'CI': 0.7}
        test_history = []
        
        # Test inner monologue fallback
        thought = inner_monologue.generate_inner_thought(test_input, test_upt, test_history, "")
        print(f"   ✅ Inner thought: '{thought}'")
        
        response = inner_monologue.generate_response_from_thought(thought, test_input, test_upt, test_history)
        print(f"   ✅ Response: '{response}'")
        
        # Test prediction error fallback
        pred_sentiment, confidence = prediction_error.predict_user_response(test_input, test_upt, test_history)
        print(f"   ✅ Prediction: {pred_sentiment} ({confidence:.2f})")
        
        surprise = prediction_error.calculate_surprise(pred_sentiment, "positive")
        print(f"   ✅ Surprise: {surprise:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_structure():
    """Test that app.py has the required structure"""
    print("📋 Testing App Structure...")
    try:
        # Read app.py content
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = [
            'from upt_core.prediction_error import PredictionErrorSystem',
            'from deloris_ai.inner_monologue import InnerMonologueSystem'
        ]
        
        for import_stmt in required_imports:
            if import_stmt in content:
                print(f"   ✅ Found: {import_stmt}")
            else:
                print(f"   ❌ Missing: {import_stmt}")
                return False
        
        # Check for system initialization
        if 'inner_monologue = InnerMonologueSystem()' in content:
            print("   ✅ Inner monologue initialization found")
        else:
            print("   ❌ Inner monologue initialization missing")
            return False
            
        if 'prediction_error = PredictionErrorSystem()' in content:
            print("   ✅ Prediction error initialization found")
        else:
            print("   ❌ Prediction error initialization missing")
            return False
        
        # Check for integration in main loop
        if 'generate_inner_thought' in content:
            print("   ✅ Inner thought generation found in main loop")
        else:
            print("   ❌ Inner thought generation missing from main loop")
            return False
            
        if 'predict_user_response' in content:
            print("   ✅ Prediction system found in main loop")
        else:
            print("   ❌ Prediction system missing from main loop")
            return False
            
        if 'calculate_surprise' in content:
            print("   ✅ Surprise calculation found in feedback loop")
        else:
            print("   ❌ Surprise calculation missing from feedback loop")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run integration tests"""
    print("🚀 Testing Deloris App Integration\n")
    
    tests = [
        ("App Imports & Functionality", test_app_imports),
        ("App Structure Check", test_app_structure)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((name, result))
        print(f"{'='*50}")
    
    # Summary
    print(f"\n🎯 INTEGRATION TEST SUMMARY")
    print("="*50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 APP INTEGRATION READY! Deloris can run with consciousness upgrades!")
        return 0
    else:
        print("⚠️  Integration issues found. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

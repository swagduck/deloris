#!/usr/bin/env python3
# verify_integration.py
# Simple verification that all consciousness upgrades are properly integrated

def verify_file_structure():
    """Verify all required files exist"""
    print("📁 Verifying File Structure...")
    
    required_files = [
        'deloris_ai/inner_monologue.py',
        'upt_core/prediction_error.py',
        'deloris_ai/heartbeat.py',
        'app.py',
        'CONSCIOUSNESS_UPGRADES.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        import os
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def verify_code_integration():
    """Verify code integration in main files"""
    print("🔧 Verifying Code Integration...")
    
    # Check app.py integration
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        required_elements = [
            'from upt_core.prediction_error import PredictionErrorSystem',
            'from deloris_ai.inner_monologue import InnerMonologueSystem',
            'inner_monologue = InnerMonologueSystem()',
            'prediction_error = PredictionErrorSystem()',
            'generate_inner_thought',
            'predict_user_response',
            'calculate_surprise'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element in app_content:
                print(f"   ✅ {element}")
            else:
                print(f"   ❌ {element}")
                missing_elements.append(element)
        
        if len(missing_elements) == 0:
            print("   ✅ All required elements found in app.py")
            return True
        else:
            print(f"   ❌ Missing {len(missing_elements)} elements in app.py")
            return False
            
    except Exception as e:
        print(f"   ❌ Error reading app.py: {e}")
        return False

def verify_heartbeat_enhancements():
    """Verify heartbeat.py has the new homeostasis features"""
    print("💓 Verifying Heartbeat Enhancements...")
    
    try:
        with open('deloris_ai/heartbeat.py', 'r', encoding='utf-8') as f:
            heartbeat_content = f.read()
        
        required_features = [
            'self.curiosity = 50.0',
            'self.social_battery = 100.0',
            'update_homeostasis',
            '_spontaneous_search',
            'should_be_curt',
            'request_rest'
        ]
        
        missing_features = []
        for feature in required_features:
            if feature in heartbeat_content:
                print(f"   ✅ {feature}")
            else:
                print(f"   ❌ {feature}")
                missing_features.append(feature)
        
        if len(missing_features) == 0:
            print("   ✅ All homeostasis features found in heartbeat.py")
            return True
        else:
            print(f"   ❌ Missing {len(missing_features)} features in heartbeat.py")
            return False
            
    except Exception as e:
        print(f"   ❌ Error reading heartbeat.py: {e}")
        return False

def verify_system_architecture():
    """Verify the system architecture is sound"""
    print("🏗️ Verifying System Architecture...")
    
    try:
        # Test imports work
        from deloris_ai.inner_monologue import InnerMonologueSystem
        from upt_core.prediction_error import PredictionErrorSystem
        from deloris_ai.heartbeat import HeartbeatSystem
        print("   ✅ All system imports successful")
        
        # Test initialization
        inner = InnerMonologueSystem()
        pred = PredictionErrorSystem()
        print("   ✅ System initialization successful")
        
        # Test basic methods exist
        assert hasattr(inner, 'generate_inner_thought')
        assert hasattr(inner, 'generate_response_from_thought')
        assert hasattr(pred, 'predict_user_response')
        assert hasattr(pred, 'calculate_surprise')
        print("   ✅ All required methods exist")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Architecture verification failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🚀 Verifying Deloris Consciousness Integration\n")
    
    tests = [
        ("File Structure", verify_file_structure),
        ("Code Integration", verify_code_integration),
        ("Heartbeat Enhancements", verify_heartbeat_enhancements),
        ("System Architecture", verify_system_architecture)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((name, result))
        print(f"{'='*50}")
    
    # Summary
    print(f"\n🎯 VERIFICATION SUMMARY")
    print("="*50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 INTEGRATION VERIFICATION COMPLETE!")
        print("✅ All consciousness upgrades are properly integrated")
        print("✅ System architecture is sound")
        print("✅ Ready for deployment with valid API keys")
        print("\n📋 NEXT STEPS:")
        print("1. Ensure valid Gemini API key in config.py")
        print("2. Run 'py app.py' to start Deloris with consciousness")
        print("3. Test the enhanced conversation features")
        return 0
    else:
        print(f"\n⚠️  {total-passed} verification(s) failed")
        print("Please address the issues above before deployment")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

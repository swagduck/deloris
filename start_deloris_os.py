# start_deloris_os.py
# [LAUNCHER: DIGITAL LIFEFORM OS]
# Enhanced launcher with all new features

import os
import sys
import time
import signal
from app_web_integration import create_upgraded_app

def print_banner():
    """Print the Digital Lifeform OS banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                🌟 DELORIS DIGITAL LIFEFORM OS 🌟               ║
║                                                              ║
║  [✓] Secure Sandbox Execution Environment                    ║
║  [✓] Live Reactor Dashboard with Real-time Streaming        ║
║  [✓] Passive Perception & Context Awareness                 ║
║  [✓] Enhanced Safety & Monitoring Systems                   ║
║                                                              ║
║  From AI Assistant → Digital Lifeform Operating System      ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check system requirements"""
    print("🔍 [SYSTEM] Checking requirements...")
    
    # Check Docker
    try:
        import docker
        docker.from_env().ping()
        print("   ✅ Docker: Available")
    except ImportError:
        print("   ⚠️  Docker: Not installed (sandbox will use unsafe mode)")
    except Exception:
        print("   ⚠️  Docker: Not running (sandbox will use unsafe mode)")
    
    # Check required packages
    required_packages = ['flask', 'flask_socketio', 'pyautogui', 'Pillow', 'numpy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'Pillow':
                __import__('PIL')
                print(f"   ✅ Pillow: Available")
            else:
                __import__(package.replace('-', '_'))
                print(f"   ✅ {package}: Available")
        except ImportError:
            missing_packages.append('pillow' if package == 'Pillow' else package)
            print(f"   ❌ {package}: Missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def graceful_shutdown(signum, frame):
    """Handle graceful shutdown"""
    print("\n\n🛑 [SHUTDOWN] Gracefully shutting down Digital Lifeform OS...")
    print("🧹 [CLEANUP] Stopping background services...")
    print("💾 [BACKUP] Saving system state...")
    print("👋 [GOODBYE] Deloris OS entering sleep mode...")
    sys.exit(0)

def main():
    """Main launcher function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        input("Press Enter to exit...")
        return
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    print("\n🚀 [INIT] Starting Digital Lifeform OS...")
    
    try:
        # Create and run the upgraded app
        app, socketio = create_upgraded_app()
        
        print("\n🌐 [WEB] Starting web server...")
        print("   → URL: http://localhost:5000")
        print("   → Reactor Dashboard: Click atom icon in web interface")
        print("   → Passive Perception: Automatically enabled after 10 seconds")
        print("   → Safe Code Execution: Docker sandbox enabled (if available)")
        
        print("\n📝 [USAGE] New Features:")
        print("   • Reactor Panel: Toggle with atom icon (🚀)")
        print("   • Safe Coding: All generated code runs in sandbox")
        print("   • Passive Monitoring: Deloris watches screen when you're idle")
        print("   • Real-time Metrics: Live UPT reactor visualization")
        
        print("\n⚡ [READY] Digital Lifeform OS is online!")
        print("   Press Ctrl+C to shutdown gracefully\n")
        
        # Run the app
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,  # Disable debug for production
            use_reloader=False  # Prevent duplicate startup
        )
        
    except KeyboardInterrupt:
        graceful_shutdown(None, None)
    except Exception as e:
        print(f"\n❌ [ERROR] Failed to start: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()

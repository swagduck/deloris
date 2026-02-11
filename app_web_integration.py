# app_web_integration.py
# [MODULE: WEB INTEGRATION UPGRADES]
# Integration of all new systems into the main Flask app

import os
import sys
from flask import Flask
from flask_socketio import SocketIO

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import new modules
from deloris_ai.sandbox import init_sandbox
from deloris_ai.passive_perception import init_passive_perception
from upt_core.reactor_service import reactor_service
from app_web_reactor import socketio

def upgrade_deloris_app(app):
    """
    Upgrade the main Deloris Flask app with new Digital Lifeform OS features
    """
    
    # 1. Initialize Sandbox System
    upload_folder = os.path.join(app.root_path, 'uploads')
    sandbox_executor = init_sandbox(upload_folder)
    app.sandbox_executor = sandbox_executor
    
    # 2. Initialize Passive Perception
    passive_perception = init_passive_perception(
        upload_folder, 
        check_interval=300  # 5 minutes
    )
    app.passive_perception = passive_perception
    
    # 3. Wrap app with Socket.IO
    socketio.init_app(app, cors_allowed_origins="*")
    
    # 4. Add new API routes
    add_upgrade_routes(app)
    
    # 5. Start background services
    start_background_services(app)
    
    print("🚀 [UPGRADE] Digital Lifeform OS features integrated")
    return app

def add_upgrade_routes(app):
    """Add new API routes for upgraded features"""
    
    @app.route('/api/reactor/start', methods=['POST'])
    def start_reactor():
        """Start reactor simulation"""
        data = request.get_json() or {}
        c_geo = data.get('c_geo', 0.911)
        tau_ion = data.get('tau_ion', 0.080)
        
        reactor_service.start_reactor(c_geo, tau_ion)
        return jsonify({'status': 'started', 'metrics': reactor_service.current_metrics})
    
    @app.route('/api/reactor/stop', methods=['POST'])
    def stop_reactor():
        """Stop reactor simulation"""
        reactor_service.stop_reactor()
        return jsonify({'status': 'stopped'})
    
    @app.route('/api/reactor/status', methods=['GET'])
    def reactor_status():
        """Get reactor status"""
        return jsonify(reactor_service.get_status())
    
    @app.route('/api/perception/start', methods=['POST'])
    def start_perception():
        """Start passive perception"""
        if hasattr(app, 'passive_perception'):
            app.passive_perception.start_monitoring()
            return jsonify({'status': 'started'})
        return jsonify({'error': 'Perception system not available'}), 500
    
    @app.route('/api/perception/stop', methods=['POST'])
    def stop_perception():
        """Stop passive perception"""
        if hasattr(app, 'passive_perception'):
            app.passive_perception.stop_monitoring()
            return jsonify({'status': 'stopped'})
        return jsonify({'error': 'Perception system not available'}), 500
    
    @app.route('/api/perception/context', methods=['GET'])
    def perception_context():
        """Get current perception context"""
        if hasattr(app, 'passive_perception'):
            return jsonify(app.passive_perception.get_context_summary())
        return jsonify({'error': 'Perception system not available'}), 500
    
    @app.route('/api/perception/glance', methods=['POST'])
    def force_glance():
        """Force immediate screen glance"""
        if hasattr(app, 'passive_perception'):
            app.passive_perception.force_glance()
            return jsonify({'status': 'glance_triggered'})
        return jsonify({'error': 'Perception system not available'}), 500

def start_background_services(app):
    """Start background services"""
    
    # Start passive perception if available
    if hasattr(app, 'passive_perception'):
        # Start after a delay to ensure system is fully loaded
        import threading
        import time
        
        def delayed_start():
            time.sleep(10)  # Wait 10 seconds after app start
            try:
                app.passive_perception.start_monitoring()
                print("👁️ [BACKGROUND] Passive perception started")
            except Exception as e:
                print(f"⚠️ [BACKGROUND] Failed to start perception: {e}")
        
        threading.Thread(target=delayed_start, daemon=True).start()
    
    # Reactor service is started on-demand via WebSocket

def create_upgraded_app():
    """Create the upgraded Deloris app"""
    from app_web import app
    
    # Apply upgrades
    upgraded_app = upgrade_deloris_app(app)
    
    return upgraded_app, socketio

if __name__ == '__main__':
    # Run the upgraded app
    app, socketio = create_upgraded_app()
    
    print("🌟 [DELORIS OS] Starting Digital Lifeform OS...")
    print("🐳 [SANDBOX] Secure code execution enabled")
    print("🔗 [WEBSOCKET] Reactor dashboard available")
    print("👁️ [PERCEPTION] Passive monitoring active")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)

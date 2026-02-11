# app_web_reactor.py
# [MODULE: REACTOR WEB INTEGRATION]
# Socket.IO integration for live reactor dashboard

from flask_socketio import SocketIO, emit, join_room, leave_room
from upt_core.reactor_service import reactor_service
import threading
import time

# Initialize Socket.IO
socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection to reactor dashboard"""
    print("🔗 [WEBSOCKET] Client connected to reactor")
    # Register client with reactor service
    client_id = id(threading.current_thread())
    reactor_service.register_client(client_id)
    
    # Send current reactor status
    emit('reactor_status', reactor_service.get_status())
    
    # Send latest visualization
    image_data = reactor_service.generate_reactor_image()
    if image_data:
        emit('reactor_image', {'image': image_data})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = id(threading.current_thread())
    reactor_service.unregister_client(client_id)
    print("🔌 [WEBSOCKET] Client disconnected from reactor")

@socketio.on('start_reactor')
def handle_start_reactor(data):
    """Start reactor simulation"""
    c_geo = data.get('c_geo', 0.911)
    tau_ion = data.get('tau_ion', 0.080)
    
    reactor_service.start_reactor(c_geo, tau_ion)
    emit('reactor_status', reactor_service.get_status(), broadcast=True)

@socketio.on('stop_reactor')
def handle_stop_reactor():
    """Stop reactor simulation"""
    reactor_service.stop_reactor()
    emit('reactor_status', reactor_service.get_status(), broadcast=True)

@socketio.on('get_reactor_image')
def handle_get_reactor_image():
    """Request current reactor visualization"""
    image_data = reactor_service.generate_reactor_image()
    if image_data:
        emit('reactor_image', {'image': image_data})

# Override reactor service client communication
def send_to_clients_override(data):
    """Send reactor updates to WebSocket clients"""
    socketio.emit('reactor_update', data, broadcast=True)

# Monkey patch the reactor service to use WebSocket
reactor_service._send_to_clients = send_to_clients_override

# Background thread for periodic image updates
def reactor_image_updater():
    """Periodically update reactor visualization"""
    while True:
        try:
            if reactor_service.clients:
                image_data = reactor_service.generate_reactor_image()
                if image_data:
                    socketio.emit('reactor_image', {'image': image_data}, broadcast=True)
            time.sleep(2.0)  # Update every 2 seconds
        except Exception as e:
            print(f"⚠️ [REACTOR] Image updater error: {e}")
            time.sleep(5.0)

# Start background image updater
image_thread = threading.Thread(target=reactor_image_updater, daemon=True)
image_thread.start()

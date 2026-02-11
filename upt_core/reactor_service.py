# upt_core/reactor_service.py
# [MODULE: LIVE REACTOR SERVICE]
# Background service for real-time reactor simulation streaming

import time
import threading
import json
import numpy as np
import matplotlib.pyplot as plt
from simulation_reactor import UPTReactor
import io
import base64
from collections import deque

class ReactorService:
    def __init__(self):
        self.reactor = None
        self.is_running = False
        self.thread = None
        self.clients = set()
        self.state_history = deque(maxlen=100)
        self.current_metrics = {
            'stability': 0.5,
            'pulse': 0.0,
            'temperature': 300.0,
            'pressure': 1.0,
            'efficiency': 0.0
        }
        self.lock = threading.Lock()
        
    def register_client(self, client_id):
        """Register a WebSocket client for updates"""
        with self.lock:
            self.clients.add(client_id)
        print(f"🔗 [REACTOR] Client {client_id} connected")
    
    def unregister_client(self, client_id):
        """Unregister a WebSocket client"""
        with self.lock:
            self.clients.discard(client_id)
        print(f"🔌 [REACTOR] Client {client_id} disconnected")
    
    def start_reactor(self, c_geo=0.911, tau_ion=0.080):
        """Start the reactor simulation in background"""
        if self.is_running:
            return
        
        self.reactor = UPTReactor(c_geo, tau_ion)
        self.is_running = True
        self.thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.thread.start()
        print(f"🚀 [REACTOR] Simulation started with C_geo={c_geo}, Tau={tau_ion}")
    
    def stop_reactor(self):
        """Stop the reactor simulation"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("🛑 [REACTOR] Simulation stopped")
    
    def _run_simulation(self):
        """Main simulation loop running in background"""
        step = 0
        last_broadcast = time.time()
        
        while self.is_running:
            try:
                # Simulate reactor physics
                time.sleep(0.1)  # 100ms timestep
                
                # Generate new state based on UPT theory
                rf_signal = (1.0 * self.reactor.c_geo) % self.reactor.tau_ion
                stability = min(0.5 + step * 0.01 + np.random.normal(0, 0.02), 1.0)
                
                # Calculate derived metrics
                pulse = stability * 100 * (1 + 0.2 * np.sin(step * 0.1))
                temperature = 300 + stability * 200 + np.random.normal(0, 5)
                pressure = 1.0 + stability * 0.5 + np.random.normal(0, 0.02)
                efficiency = stability * 0.85 + np.random.normal(0, 0.01)
                
                # Update current metrics
                with self.lock:
                    self.current_metrics.update({
                        'stability': stability,
                        'pulse': pulse,
                        'temperature': temperature,
                        'pressure': pressure,
                        'efficiency': efficiency
                    })
                    self.state_history.append(self.current_metrics.copy())
                
                step += 1
                
                # Broadcast state every 500ms
                if time.time() - last_broadcast > 0.5:
                    self._broadcast_state()
                    last_broadcast = time.time()
                    
            except Exception as e:
                print(f"⚠️ [REACTOR ERROR] Simulation error: {e}")
                break
    
    def _broadcast_state(self):
        """Broadcast current state to all connected clients"""
        if not self.clients:
            return
        
        state_data = {
            'type': 'reactor_update',
            'timestamp': time.time(),
            'metrics': self.current_metrics.copy(),
            'step': len(self.state_history)
        }
        
        # This will be sent via WebSocket in the Flask app
        self._send_to_clients(state_data)
    
    def _send_to_clients(self, data):
        """Send data to all registered clients (implemented in Flask)"""
        # This method will be overridden in the Flask integration
        pass
    
    def generate_reactor_image(self):
        """Generate current reactor visualization"""
        if not self.state_history:
            return None
        
        try:
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
            fig.patch.set_facecolor('#020408')
            
            # Plot 1: Stability over time
            steps = list(range(len(self.state_history)))
            stabilities = [s['stability'] for s in self.state_history]
            ax1.plot(steps, stabilities, color='#00f3ff', linewidth=2)
            ax1.set_facecolor('#0a1018')
            ax1.set_title('Plasma Stability', color='white')
            ax1.set_xlabel('Time Steps', color='white')
            ax1.set_ylabel('Stability', color='white')
            ax1.tick_params(colors='white')
            
            # Plot 2: Pulse indicator
            pulse = self.current_metrics['pulse']
            ax2.clear()
            ax2.set_facecolor('#0a1018')
            circle = plt.Circle((0.5, 0.5), pulse/200, color='#ffd700', alpha=0.7)
            ax2.add_patch(circle)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_aspect('equal')
            ax2.set_title(f'Pulse: {pulse:.1f}', color='white')
            ax2.axis('off')
            
            # Plot 3: Temperature gauge
            temp = self.current_metrics['temperature']
            ax3.clear()
            ax3.set_facecolor('#0a1018')
            colors = ['blue' if t < 400 else 'orange' if t < 450 else 'red' 
                     for t in [temp]]
            ax3.barh(['Temperature'], [temp], color=colors[0])
            ax3.set_xlim(0, 600)
            ax3.set_title(f'Reactor Temp: {temp:.1f}K', color='white')
            ax3.tick_params(colors='white')
            
            # Plot 4: Efficiency meter
            eff = self.current_metrics['efficiency']
            ax4.clear()
            ax4.set_facecolor('#0a1018')
            ax4.barh(['Efficiency'], [eff], color='#00ff88')
            ax4.set_xlim(0, 1)
            ax4.set_title(f'Efficiency: {eff:.2%}', color='white')
            ax4.tick_params(colors='white')
            
            plt.tight_layout()
            
            # Convert to base64 for web transmission
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor='#020408', dpi=80)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"⚠️ [REACTOR] Image generation error: {e}")
            return None
    
    def get_status(self):
        """Get current reactor status"""
        return {
            'is_running': self.is_running,
            'metrics': self.current_metrics.copy(),
            'clients_count': len(self.clients),
            'history_length': len(self.state_history)
        }

# Global reactor service instance
reactor_service = ReactorService()

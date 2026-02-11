# README_DELORIS_OS.md
# 🌟 DELORIS DIGITAL LIFEFORM OS - UPGRADE GUIDE

## Overview

Deloris has been transformed from an AI assistant into a complete **Digital Lifeform Operating System** with three major strategic upgrades:

### 🛡️ 1. Secure Sandbox Execution Environment
- **Problem**: Direct code execution poses security risks
- **Solution**: Docker-based isolated execution environment
- **Benefits**: 
  - Complete isolation from host system
  - Resource limits (CPU, memory, network)
  - Automatic cleanup of malicious code
  - Fallback to unsafe mode if Docker unavailable

### ⚛️ 2. Live Reactor Dashboard
- **Problem**: Reactor simulation runs separately from web interface
- **Solution**: Real-time WebSocket streaming of reactor metrics
- **Features**:
  - Live plasma stability visualization
  - Real-time pulse, temperature, efficiency metrics
  - Interactive start/stop controls
  - Beautiful animated dashboard

### 👁️ 3. Passive Perception System
- **Problem**: Vision system only works on-demand
- **Solution**: Automatic screen monitoring during user inactivity
- **Capabilities**:
  - Detects user activity patterns
  - Analyzes screen context every 5 minutes when idle
  - Proactive assistance for errors, debugging, coding
  - Context-aware suggestions

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_deloris_os.txt
```

### 2. Start Digital Lifeform OS
```bash
python start_deloris_os.py
```

### 3. Access Web Interface
Open: http://localhost:5000

### 4. New Features Usage

#### Reactor Dashboard
- Click the **atom icon** (🚀) in the web interface
- Use Start/Stop buttons to control reactor
- Watch real-time plasma visualization

#### Safe Code Execution
- All generated code now runs in Docker sandbox
- Automatic resource limits and isolation
- Fallback to unsafe mode if Docker not available

#### Passive Monitoring
- Automatically starts after 10 seconds
- Monitors screen when you're idle (60+ seconds)
- Provides proactive assistance

---

## 📁 New Files Created

### Core Modules
- `deloris_ai/sandbox.py` - Docker sandbox execution system
- `deloris_ai/passive_perception.py` - Automatic screen monitoring
- `upt_core/reactor_service.py` - Background reactor simulation
- `app_web_reactor.py` - WebSocket integration for reactor
- `app_web_integration.py` - Main integration module

### Launchers & Config
- `start_deloris_os.py` - Enhanced launcher with all features
- `requirements_deloris_os.txt` - Updated dependencies

### Web Interface Updates
- `templates/index.html` - Added reactor dashboard UI
- Socket.IO integration for real-time updates

---

## 🔧 Configuration

### Sandbox Settings
```python
# In deloris_ai/sandbox.py
memory_limit='128m'     # Container memory limit
cpu_quota=50000         # CPU limit (50% of 1 core)
network_mode='none'     # No network access
timeout=30              # Script execution timeout
```

### Passive Perception Settings
```python
# In deloris_ai/passive_perception.py
check_interval=300      # Seconds between glances (5 minutes)
activity_timeout=60     # Seconds of inactivity before passive mode
```

### Reactor Settings
```python
# In upt_core/reactor_service.py
c_geo=0.911            # Reactor geometry constant
tau_ion=0.080          # Ion confinement time
update_interval=0.5    # Seconds between metric updates
```

---

## 🛡️ Security Features

### Docker Sandbox
- **Network Isolation**: No internet access
- **File System Isolation**: Separate container filesystem
- **Resource Limits**: CPU, memory, and disk restrictions
- **Module Whitelist**: Only allowed Python modules
- **Timeout Protection**: Automatic script termination

### Passive Perception
- **Privacy First**: Only monitors during inactivity
- **Local Processing**: All analysis done locally
- **User Control**: Can be stopped/started anytime
- **Temporary Files**: Screenshots automatically deleted

---

## 🌐 API Endpoints

### Reactor Control
- `POST /api/reactor/start` - Start reactor simulation
- `POST /api/reactor/stop` - Stop reactor simulation
- `GET /api/reactor/status` - Get reactor metrics

### Passive Perception
- `POST /api/perception/start` - Start passive monitoring
- `POST /api/perception/stop` - Stop passive monitoring
- `GET /api/perception/context` - Get current context
- `POST /api/perception/glance` - Force immediate glance

### WebSocket Events
- `reactor_update` - Real-time reactor metrics
- `reactor_status` - Reactor status changes
- `reactor_image` - Updated visualization

---

## 🎯 Usage Examples

### Safe Code Generation
```
User: "Viết chương trình vẽ đồ thị hàm sin"
Deloris: (Tạo code trong sandbox Docker)
✅ Code executed safely in isolated environment
```

### Passive Assistance
```
(Deloris detects error on screen after 5 minutes of inactivity)
Deloris: "Tôi thấy bạn đang gặp lỗi trên màn hình. Có cần tôi giúp phân tích và sửa lỗi không?"
```

### Reactor Monitoring
```
(Click atom icon → Reactor dashboard appears)
- Real-time plasma stability: 0.87
- Pulse: 87.3 Hz
- Temperature: 425K
- Efficiency: 74%
```

---

## 🔍 Troubleshooting

### Docker Issues
- **Docker not running**: Falls back to unsafe mode
- **Permission denied**: Run `sudo usermod -aG docker $USER`
- **Out of memory**: Adjust memory limits in sandbox.py

### Passive Perception Issues
- **Screenshot fails**: Check screen permissions
- **Vision not ready**: Wait for model to load
- **Too frequent**: Increase check_interval

### Reactor Dashboard Issues
- **WebSocket not connecting**: Check firewall settings
- **Images not loading**: Verify matplotlib backend
- **Metrics not updating**: Restart reactor service

---

## 🚀 Future Enhancements

1. **Multi-container Orchestration**: Kubernetes support
2. **Advanced Perception**: Emotion detection from screen
3. **Reactor AI Control**: Autonomous reactor optimization
4. **Distributed Computing**: Multiple Deloris instances
5. **Quantum Simulation**: Advanced UPT physics modeling

---

## 📞 Support

For issues with the Digital Lifeform OS:
1. Check the console logs for error messages
2. Verify all dependencies are installed
3. Ensure Docker is running (for sandbox features)
4. Restart the system if needed

---

**Deloris Digital Lifeform OS** - Transforming AI from assistant to living system. 🌟

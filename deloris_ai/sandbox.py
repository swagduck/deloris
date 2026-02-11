# deloris_ai/sandbox.py
# [MODULE: SECURE SANDBOX EXECUTION]
# Isolate Deloris' code execution in Docker containers for safety

import docker
import os
import uuid
import time
import json
import tempfile
import shutil
from typing import Optional, Dict, Any

class SecureSandbox:
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir
        self.client = None
        self.active_containers = {}
        self._init_docker()
    
    def _init_docker(self):
        """Initialize Docker client"""
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
            print("🐳 [SANDBOX] Docker connected successfully")
        except Exception as e:
            print(f"⚠️ [SANDBOX ERROR] Docker not available: {e}")
            print("   -> Falling back to unsafe execution mode")
            self.client = None
    
    def create_sandbox_script(self, script_content: str, script_name: str) -> str:
        """Create a secure wrapper script for execution"""
        wrapper_content = f'''#!/usr/bin/env python3
import sys
import os
import importlib.util
import traceback
import signal

# Security: Limit available modules
ALLOWED_MODULES = {{
    'numpy', 'matplotlib', 'pandas', 'PIL', 'pillow', 'pyautogui', 
    'requests', 'random', 'math', 'time', 'json', 'csv', 'datetime'
}}

def secure_import(name, *args, **kwargs):
    if name not in ALLOWED_MODULES:
        raise ImportError(f"Module '{{name}}' is not allowed in sandbox")
    return __import__(name, *args, **kwargs)

# Override import mechanism
original_import = __builtins__['__import__']
__builtins__['__import__'] = secure_import

# Set up secure environment
os.chdir('/workspace')
os.environ['PYTHONPATH'] = '/workspace'

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Script execution timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    # Execute the user script
    exec(open('{script_name}').read())
    print("✅ Script completed successfully")
except Exception as e:
    print(f"❌ Script error: {{e}}")
    traceback.print_exc()
finally:
    signal.alarm(0)  # Cancel timeout
'''
        return wrapper_content
    
    def execute_script(self, script_name: str, script_content: str) -> Dict[str, Any]:
        """Execute script in isolated Docker container"""
        if not self.client:
            # Fallback to unsafe execution
            return self._unsafe_fallback(script_name, script_content)
        
        container_id = f"deloris_sandbox_{uuid.uuid4().hex[:8]}"
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Write script files
            script_path = os.path.join(temp_dir, script_name)
            wrapper_path = os.path.join(temp_dir, "sandbox_wrapper.py")
            
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            wrapper_content = self.create_sandbox_script(script_content, script_name)
            with open(wrapper_path, 'w', encoding='utf-8') as f:
                f.write(wrapper_content)
            
            # Create and run container
            print(f"🐳 [SANDBOX] Starting container: {container_id}")
            
            container = self.client.containers.run(
                "python:3.9-slim",
                command=f"python /workspace/sandbox_wrapper.py",
                volumes={
                    temp_dir: {'bind': '/workspace', 'mode': 'rw'}
                },
                working_dir='/workspace',
                network_mode='none',  # No network access
                mem_limit='128m',     # Memory limit
                cpu_quota=50000,      # CPU limit
                detach=True,
                remove=True,
                name=container_id
            )
            
            self.active_containers[container_id] = container
            
            # Wait for completion with timeout
            result = container.wait(timeout=60)
            logs = container.logs.decode('utf-8')
            
            # Clean up generated files
            generated_files = []
            for file in os.listdir(temp_dir):
                if file.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    # Move generated images to static folder
                    src = os.path.join(temp_dir, file)
                    dst = os.path.join(self.upload_dir, '..', 'static', 'generated', file)
                    shutil.move(src, dst)
                    generated_files.append(f"/static/generated/{file}")
            
            success = result['StatusCode'] == 0
            
            return {
                'success': success,
                'output': logs,
                'generated_files': generated_files,
                'container_id': container_id
            }
            
        except Exception as e:
            print(f"⚠️ [SANDBOX ERROR] {e}")
            return {'success': False, 'output': str(e)}
        finally:
            # Cleanup
            if container_id in self.active_containers:
                try:
                    container = self.active_containers[container_id]
                    container.remove(force=True)
                    del self.active_containers[container_id]
                except:
                    pass
            
            # Clean temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def _unsafe_fallback(self, script_name: str, script_content: str) -> Dict[str, Any]:
        """Fallback execution when Docker is not available"""
        print("⚠️ [SANDBOX] Using unsafe execution mode")
        try:
            # Write script to uploads folder
            script_path = os.path.join(self.upload_dir, script_name)
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Execute directly (original method)
            import subprocess
            import sys
            
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.upload_dir
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout + result.stderr,
                'generated_files': [],
                'container_id': 'unsafe_mode'
            }
            
        except Exception as e:
            return {'success': False, 'output': str(e)}
    
    def cleanup_all_containers(self):
        """Force cleanup of all sandbox containers"""
        if not self.client:
            return
        
        for container_id in list(self.active_containers.keys()):
            try:
                container = self.active_containers[container_id]
                container.stop()
                container.remove(force=True)
                del self.active_containers[container_id]
                print(f"🧹 [SANDBOX] Cleaned up container: {container_id}")
            except Exception as e:
                print(f"⚠️ [SANDBOX] Cleanup error for {container_id}: {e}")

# Global instance
sandbox_executor = None

def init_sandbox(upload_dir: str):
    """Initialize the sandbox system"""
    global sandbox_executor
    sandbox_executor = SecureSandbox(upload_dir)
    return sandbox_executor

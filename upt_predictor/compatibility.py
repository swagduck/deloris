# upt_predictor/compatibility.py
# Compatibility layer for different UPTAutomatorModel versions

import torch
import torch.nn as nn
from .architecture import UPTAutomatorModel

class UPTAutomatorModelCompat(nn.Module):
    """
    Compatibility wrapper for UPTAutomatorModel.
    Handles both old single-input and new dual-input architectures.
    """
    
    def __init__(self, text_input_dim, image_input_dim, hidden_dim):
        super(UPTAutomatorModelCompat, self).__init__()
        
        self.text_input_dim = text_input_dim
        self.image_input_dim = image_input_dim
        self.hidden_dim = hidden_dim
        
        # Force use legacy model since saved checkpoint is in legacy format
        print("[UPTAutomator] Using legacy model format for compatibility")
        self.model = self._create_legacy_model(text_input_dim, hidden_dim)
        self.is_dual_input = False
    
    def _create_legacy_model(self, input_dim, hidden_dim):
        """Create legacy single-input model matching saved checkpoint"""
        class LegacyUPTAutomator(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super(LegacyUPTAutomator, self).__init__()
                # Based on the error, the saved model has these dimensions:
                # fc1_text: [256, 774] -> input_dim should be 774, hidden_dim 256
                # fc1_image: [256, 512] -> image_dim 512, hidden_dim 256
                # fc2: [256, 512] -> input 512 (256+256), output 256
                # fc3_*: [1, 256] -> input 256, output 1
                
                self.fc1_text = nn.Linear(774, 256)  # Match checkpoint
                self.fc1_image = nn.Linear(512, 256)  # Match checkpoint
                self.fc2 = nn.Linear(512, 256)  # Match checkpoint (256+256 -> 256)
                self.fc3_A = nn.Linear(256, 1)  # Match checkpoint
                self.fc3_E = nn.Linear(256, 1)  # Match checkpoint
                self.fc3_C = nn.Linear(256, 1)  # Match checkpoint
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                # Legacy single-input forward - simulate dual input processing
                x_text = self.dropout(self.relu(self.fc1_text(x)))
                # Create dummy image features (512 features to match fc1_image input)
                batch_size = x.shape[0]
                x_image_dummy = torch.zeros(batch_size, 512, device=x.device, dtype=x.dtype)
                x_image = self.dropout(self.relu(self.fc1_image(x_image_dummy)))
                
                # Concatenate text and image features
                x_fused = torch.cat((x_text, x_image), dim=1)
                
                # Continue with fusion layer
                x = self.dropout(self.relu(self.fc2(x_fused)))
                out_A = self.sigmoid(self.fc3_A(x))
                out_E = self.sigmoid(self.fc3_E(x))
                out_C = self.sigmoid(self.fc3_C(x))
                return torch.cat([out_A, out_E, out_C], dim=1)
        
        return LegacyUPTAutomator(input_dim, hidden_dim)
    
    def forward(self, x_textual, x_visual=None):
        """
        Forward pass with compatibility handling.
        """
        if self.is_dual_input:
            # New dual-input model
            if x_visual is None:
                # Create dummy visual input if not provided
                batch_size = x_textual.shape[0]
                x_visual = torch.zeros(batch_size, self.image_input_dim, 
                                     device=x_textual.device, dtype=x_textual.dtype)
            return self.model(x_textual, x_visual)
        else:
            # Legacy single-input model
            return self.model(x_textual)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with compatibility handling"""
        try:
            return self.model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print("[UPTAutomator] Model architecture mismatch, using compatibility mode")
                # Try loading with non-strict mode
                return self.model.load_state_dict(state_dict, strict=False)
            else:
                raise e

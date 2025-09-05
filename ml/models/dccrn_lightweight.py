"""
Lightweight DCCRN Model for Low-Resource Systems
Optimized for Intel i3-7020U CPU with 8GB RAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class LightweightComplexConv2d(nn.Module):
    """Memory-efficient complex convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], 
                 stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0)):
        super().__init__()
        # Reduced parameter count by using depthwise separable convolution
        self.depthwise = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size, 
                                 stride, padding, groups=in_channels * 2, bias=False)
        self.pointwise = nn.Conv2d(in_channels * 2, out_channels * 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, F, T] where C represents [real, imag] stacked
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return x

class LightweightComplexLSTM(nn.Module):
    """Simplified LSTM for complex spectrograms"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Single LSTM layer to reduce memory usage
        self.lstm = nn.LSTM(input_size * 2, hidden_size, num_layers, 
                           batch_first=True, dropout=0.0, bidirectional=False)
        self.projection = nn.Linear(hidden_size, input_size * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, F, T]
        B, C, F, T = x.shape
        
        # Reshape for LSTM: [B, T, F*C]
        x = x.permute(0, 3, 2, 1).contiguous()  # [B, T, F, C]
        x = x.view(B, T, F * C)  # [B, T, F*C]
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [B, T, hidden_size]
        
        # Project back to original size
        output = self.projection(lstm_out)  # [B, T, F*C]
        
        # Reshape back: [B, C, F, T]
        output = output.view(B, T, F, C)
        output = output.permute(0, 3, 2, 1).contiguous()
        
        return output

class LightweightDCCRN(nn.Module):
    """
    Lightweight Deep Complex Convolution Recurrent Network
    Optimized for low-resource systems
    """
    
    def __init__(self, 
                 rnn_layers: int = 1,
                 rnn_units: int = 64,
                 kernel_size: Tuple[int, int] = (5, 3),
                 kernel_num: Tuple[int, int, int] = (16, 32, 64),
                 dropout: float = 0.2):
        super().__init__()
        
        self.kernel_num = kernel_num
        
        # Encoder - Lightweight version
        self.encoder = nn.ModuleList([
            LightweightComplexConv2d(1, kernel_num[0], kernel_size, (2, 1), (2, 1)),
            LightweightComplexConv2d(kernel_num[0], kernel_num[1], kernel_size, (2, 1), (2, 1)),
            LightweightComplexConv2d(kernel_num[1], kernel_num[2], kernel_size, (2, 1), (2, 1)),
        ])
        
        # Calculate LSTM input size (frequency dimension after downsampling)
        # Assuming input frequency bins = 257 (for 512 FFT)
        freq_bins = 257
        for _ in range(len(self.encoder)):
            freq_bins = (freq_bins + 2 * 2 - kernel_size[0]) // 2 + 1
        
        # LSTM - Single layer for efficiency
        self.lstm = LightweightComplexLSTM(
            input_size=freq_bins * kernel_num[2] // 2,  # Divide by 2 for real/imag
            hidden_size=rnn_units,
            num_layers=rnn_layers
        )
        
        # Decoder - Lightweight version
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(kernel_num[2] * 2, kernel_num[1] * 2, kernel_size, (2, 1), (2, 1)),
            nn.ConvTranspose2d(kernel_num[1] * 2, kernel_num[0] * 2, kernel_size, (2, 1), (2, 1)),
            nn.ConvTranspose2d(kernel_num[0] * 2, 2, kernel_size, (2, 1), (2, 1)),
        ])
        
        # Normalization layers
        self.decoder_norms = nn.ModuleList([
            nn.BatchNorm2d(kernel_num[1] * 2),
            nn.BatchNorm2d(kernel_num[0] * 2),
            nn.BatchNorm2d(2),
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout)
        
        # Output activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Complex spectrogram [B, 2, F, T] where dim 1 = [real, imag]
        Returns:
            Enhanced complex spectrogram [B, 2, F, T]
        """
        # Store original shape and skip connections
        orig_shape = x.shape
        skip_connections = []
        
        # Encoder path
        for i, enc_layer in enumerate(self.encoder):
            skip_connections.append(x)
            x = enc_layer(x)
            x = F.leaky_relu(x, 0.1)
            if i < len(self.encoder) - 1:  # Don't apply dropout to last encoder layer
                x = self.dropout(x)
        
        # LSTM processing
        x = self.lstm(x)
        
        # Decoder path
        for i, (dec_layer, norm) in enumerate(zip(self.decoder, self.decoder_norms)):
            # Skip connection
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                # Resize skip connection if needed
                if x.shape[-2:] != skip.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='nearest')
                x = x + skip
            
            x = dec_layer(x)
            x = norm(x)
            
            if i < len(self.decoder) - 1:
                x = F.leaky_relu(x, 0.1)
                x = self.dropout(x)
        
        # Final resize to match input
        if x.shape != orig_shape:
            x = F.interpolate(x, size=orig_shape[-2:], mode='nearest')
        
        # Apply magnitude mask
        input_mag = torch.sqrt(orig_shape[0]**2 + orig_shape[1]**2 + 1e-8)
        mask = self.sigmoid(x[:, :1])  # Use only first channel for mask
        
        # Apply mask to input magnitude, preserve phase
        output_real = mask * orig_shape[0]
        output_imag = mask * orig_shape[1]
        
        return torch.stack([output_real, output_imag], dim=1).squeeze(2)
    
    def get_model_size(self) -> int:
        """Get model size in parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_usage(self, batch_size: int = 1, freq_bins: int = 257, time_frames: int = 100) -> float:
        """Estimate memory usage in MB"""
        # Rough estimation
        param_memory = self.get_model_size() * 4 / (1024 * 1024)  # 4 bytes per float32
        
        # Activation memory (rough estimate)
        activation_memory = batch_size * freq_bins * time_frames * 64 * 4 / (1024 * 1024)
        
        return param_memory + activation_memory

def create_lightweight_dccrn(**kwargs) -> LightweightDCCRN:
    """Create a lightweight DCCRN model with default parameters optimized for low resources"""
    return LightweightDCCRN(**kwargs)

if __name__ == "__main__":
    # Test the lightweight model
    model = create_lightweight_dccrn()
    
    # Test input (batch_size=1, channels=2, freq_bins=257, time_frames=100)
    test_input = torch.randn(1, 2, 257, 100)
    
    print(f"Model parameters: {model.get_model_size():,}")
    print(f"Estimated memory usage: {model.get_memory_usage():.2f} MB")
    
    # Test forward pass
    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print("Lightweight DCCRN model test successful!")

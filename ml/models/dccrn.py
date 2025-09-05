import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange
import numpy as np


class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    
    def forward(self, x_real, x_imag):
        # Complex multiplication: (a + jb)(c + jd) = (ac - bd) + j(ad + bc)
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag


class ComplexConvTranspose2d(nn.Module):
    """Complex-valued 2D transposed convolution layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
        self.conv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
    
    def forward(self, x_real, x_imag):
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag


class ComplexBatchNorm2d(nn.Module):
    """Complex-valued batch normalization"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(ComplexBatchNorm2d, self).__init__()
        self.bn_real = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.bn_imag = nn.BatchNorm2d(num_features, eps, momentum, affine)
    
    def forward(self, x_real, x_imag):
        return self.bn_real(x_real), self.bn_imag(x_imag)


class ComplexLSTM(nn.Module):
    """Complex-valued LSTM layer"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=False):
        super(ComplexLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        
        # Ensure dropout is valid for LSTM (only applied if num_layers > 1)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        # Real and imaginary LSTM layers
        self.lstm_real = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=batch_first, 
            dropout=lstm_dropout, 
            bidirectional=bidirectional
        )
        self.lstm_imag = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=batch_first, 
            dropout=lstm_dropout, 
            bidirectional=bidirectional
        )
        
    def forward(self, x_real, x_imag, hidden=None):
        if hidden is not None:
            h_real, c_real = hidden[0]
            h_imag, c_imag = hidden[1]
        else:
            h_real = h_imag = c_real = c_imag = None
        
        # Complex LSTM forward pass
        out_real_r, (h_real_r, c_real_r) = self.lstm_real(x_real, (h_real, c_real) if h_real is not None else None)
        out_real_i, _ = self.lstm_real(x_imag, (h_imag, c_imag) if h_imag is not None else None)
        
        out_imag_r, (h_imag_r, c_imag_r) = self.lstm_imag(x_real, (h_real, c_real) if h_real is not None else None)
        out_imag_i, _ = self.lstm_imag(x_imag, (h_imag, c_imag) if h_imag is not None else None)
        
        # Complex multiplication
        out_real = out_real_r - out_imag_i
        out_imag = out_real_i + out_imag_r
        
        return (out_real, out_imag), ((h_real_r, c_real_r), (h_imag_r, c_imag_r))


class DCCRN(nn.Module):
    """Deep Complex Convolutional Recurrent Network for Speech Enhancement"""
    
    def __init__(
        self,
        n_fft=512,
        hop_length=256,
        win_length=512,
        encoder_layers=5,
        hidden_dim=128,
        lstm_layers=2,
        use_clstm=True,
        kernel_size=(5, 2),
        stride=(2, 1),
        use_cbn=True,
        masking_mode='E',  # 'E' for magnitude estimation, 'C' for complex mask, 'R' for real mask
        causal=False
    ):
        super(DCCRN, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.masking_mode = masking_mode
        self.causal = causal
        
        # Calculate frequency bins
        self.freq_bins = n_fft // 2 + 1
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()
        
        in_channels = 1
        for i in range(encoder_layers):
            out_channels = hidden_dim * (2 ** min(i, 3))  # Cap at 1024 channels
            
            self.encoder_layers.append(
                ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding=(2, 0))
            )
            
            if use_cbn:
                self.encoder_bns.append(ComplexBatchNorm2d(out_channels))
            else:
                self.encoder_bns.append(None)
                
            in_channels = out_channels
        
        # LSTM layers
        self.use_clstm = use_clstm
        if use_clstm:
            # Calculate LSTM input size based on encoder output
            lstm_input_size = self._calculate_lstm_input_size()
            self.lstm = ComplexLSTM(
                lstm_input_size, 
                hidden_dim, 
                lstm_layers, 
                batch_first=True,
                dropout=0.0,  # Explicitly set dropout to 0.0
                bidirectional=not causal
            )
            lstm_output_size = hidden_dim * (2 if not causal else 1)
        else:
            # Use real-valued LSTM on concatenated real and imaginary parts
            lstm_input_size = self._calculate_lstm_input_size() * 2
            lstm_dropout = 0.0 if lstm_layers == 1 else 0.1  # Only use dropout if multiple layers
            self.lstm = nn.LSTM(
                lstm_input_size, 
                hidden_dim, 
                lstm_layers, 
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=not causal
            )
            lstm_output_size = hidden_dim * (2 if not causal else 1)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        self.decoder_bns = nn.ModuleList()
        
        # First decoder layer transforms LSTM output back to conv format
        self.lstm_to_conv = ComplexConv2d(lstm_output_size, in_channels, (1, 1))
        
        for i in range(encoder_layers):
            layer_idx = encoder_layers - 1 - i
            if layer_idx == 0:
                out_channels = 1  # Final layer outputs single channel
            else:
                out_channels = hidden_dim * (2 ** min(layer_idx - 1, 3))
            
            self.decoder_layers.append(
                ComplexConvTranspose2d(
                    in_channels * 2,  # Skip connection doubles channels
                    out_channels, 
                    kernel_size, 
                    stride, 
                    padding=(2, 0),
                    output_padding=(1, 0) if stride[0] > 1 else (0, 0)
                )
            )
            
            if use_cbn and i < encoder_layers - 1:  # No BN on final layer
                self.decoder_bns.append(ComplexBatchNorm2d(out_channels))
            else:
                self.decoder_bns.append(None)
                
            in_channels = out_channels
        
        # Output layer for mask estimation
        if masking_mode == 'E':
            # Magnitude estimation
            self.output_layer = nn.Sequential(
                nn.Conv2d(2, 1, 1),  # Combine real and imaginary parts
                nn.Sigmoid()
            )
        elif masking_mode == 'C':
            # Complex mask
            self.output_layer = ComplexConv2d(1, 1, 1)
        else:  # masking_mode == 'R'
            # Real mask
            self.output_layer = nn.Sequential(
                nn.Conv2d(2, 1, 1),
                nn.Tanh()
            )
    
    def _calculate_lstm_input_size(self):
        """Calculate the input size for LSTM based on encoder output"""
        # Simulate forward pass through encoder to get dimensions
        dummy_input = torch.randn(1, 1, self.freq_bins, 100)  # Dummy time dimension
        
        x_real, x_imag = dummy_input, torch.zeros_like(dummy_input)
        
        for layer, bn in zip(self.encoder_layers, self.encoder_bns):
            x_real, x_imag = layer(x_real, x_imag)
            if bn is not None:
                x_real, x_imag = bn(x_real, x_imag)
            x_real = F.elu(x_real)
            x_imag = F.elu(x_imag)
        
        return x_real.size(1) * x_real.size(2)  # channels * frequency_bins
    
    def forward(self, noisy_spec):
        """
        Forward pass of DCCRN
        
        Args:
            noisy_spec: Complex spectrogram [B, F, T, 2] where last dim is [real, imag]
        
        Returns:
            enhanced_spec: Enhanced complex spectrogram [B, F, T, 2]
        """
        batch_size, freq_bins, time_steps, _ = noisy_spec.shape
        
        # Split real and imaginary parts
        noisy_real = noisy_spec[..., 0].unsqueeze(1)  # [B, 1, F, T]
        noisy_imag = noisy_spec[..., 1].unsqueeze(1)  # [B, 1, F, T]
        
        # Encoder
        encoder_outputs = []
        x_real, x_imag = noisy_real, noisy_imag
        
        for layer, bn in zip(self.encoder_layers, self.encoder_bns):
            x_real, x_imag = layer(x_real, x_imag)
            if bn is not None:
                x_real, x_imag = bn(x_real, x_imag)
            x_real = F.elu(x_real)
            x_imag = F.elu(x_imag)
            encoder_outputs.append((x_real, x_imag))
        
        # Reshape for LSTM
        B, C, Freq, T = x_real.shape
        
        if self.use_clstm:
            # Complex LSTM
            B, C, Freq_dim, T = x_real.shape
            x_real_lstm = rearrange(x_real, 'b c f t -> b t (c f)')
            x_imag_lstm = rearrange(x_imag, 'b c f t -> b t (c f)')
            
            (lstm_real, lstm_imag), _ = self.lstm(x_real_lstm, x_imag_lstm)
            
            # Reshape back to [B, hidden_dim, F, T] format for conv layers
            lstm_real = rearrange(lstm_real, 'b t h -> b h 1 t')  # [B, H, 1, T]
            lstm_imag = rearrange(lstm_imag, 'b t h -> b h 1 t')  # [B, H, 1, T]
            
            # Expand frequency dimension to match original
            lstm_real = lstm_real.expand(B, -1, Freq_dim, T)  # [B, H, F, T]
            lstm_imag = lstm_imag.expand(B, -1, Freq_dim, T)  # [B, H, F, T]
            
        else:
            # Real LSTM on concatenated features  
            B, C, Freq_dim, T = x_real.shape
            x_lstm = rearrange(torch.cat([x_real, x_imag], dim=1), 'b c f t -> b t (c f)')
            lstm_out, _ = self.lstm(x_lstm)
            
            # Reshape back to [B, hidden_dim, F, T] format for conv layers
            lstm_out = rearrange(lstm_out, 'b t h -> b h 1 t')  # [B, H, 1, T]
            lstm_out = lstm_out.expand(B, -1, Freq_dim, T)  # [B, H, F, T]
            
            # Split back to real and imaginary
            mid_channels = lstm_out.size(1) // 2
            lstm_real = lstm_out[:, :mid_channels]
            lstm_imag = lstm_out[:, mid_channels:]
        
        # Transform LSTM output back to conv format
        # LSTM output is now [B, H, F, T] which matches conv format
        x_real, x_imag = self.lstm_to_conv(lstm_real, lstm_imag)
        
        # Decoder with skip connections
        for i, (layer, bn) in enumerate(zip(self.decoder_layers, self.decoder_bns)):
            # Skip connection with size matching
            skip_real, skip_imag = encoder_outputs[-(i+1)]
            
            # Match spatial dimensions for concatenation
            if x_real.shape[2:] != skip_real.shape[2:]:
                # Interpolate to match dimensions
                target_size = skip_real.shape[2:]
                x_real = F.interpolate(x_real, size=target_size, mode='nearest')
                x_imag = F.interpolate(x_imag, size=target_size, mode='nearest')
            
            x_real = torch.cat([x_real, skip_real], dim=1)
            x_imag = torch.cat([x_imag, skip_imag], dim=1)
            
            x_real, x_imag = layer(x_real, x_imag)
            
            if bn is not None:
                x_real, x_imag = bn(x_real, x_imag)
                x_real = F.elu(x_real)
                x_imag = F.elu(x_imag)
        
        # Output layer - ensure frequency dimension matches input
        if x_real.shape[2] != freq_bins:
            # Crop or pad to match input frequency dimension
            if x_real.shape[2] > freq_bins:
                x_real = x_real[:, :, :freq_bins, :]
                x_imag = x_imag[:, :, :freq_bins, :]
            else:
                padding = freq_bins - x_real.shape[2]
                x_real = F.pad(x_real, (0, 0, 0, padding))
                x_imag = F.pad(x_imag, (0, 0, 0, padding))
        
        if self.masking_mode == 'E':
            # Magnitude estimation
            magnitude_mask = self.output_layer(torch.cat([x_real, x_imag], dim=1))  # [B, 1, F, T]
            magnitude_mask = magnitude_mask.squeeze(1)  # [B, F, T]
            
            # Apply mask to noisy magnitude while preserving phase
            noisy_magnitude = torch.sqrt(noisy_real.squeeze(1)**2 + noisy_imag.squeeze(1)**2 + 1e-8)
            noisy_phase_real = noisy_real.squeeze(1) / (noisy_magnitude + 1e-8)
            noisy_phase_imag = noisy_imag.squeeze(1) / (noisy_magnitude + 1e-8)
            
            enhanced_magnitude = magnitude_mask * noisy_magnitude
            enhanced_real = enhanced_magnitude * noisy_phase_real
            enhanced_imag = enhanced_magnitude * noisy_phase_imag
            
        elif self.masking_mode == 'C':
            # Complex mask
            mask_real, mask_imag = self.output_layer(x_real, x_imag)
            mask_real, mask_imag = mask_real.squeeze(1), mask_imag.squeeze(1)
            
            # Apply complex mask
            noisy_r, noisy_i = noisy_real.squeeze(1), noisy_imag.squeeze(1)
            enhanced_real = mask_real * noisy_r - mask_imag * noisy_i
            enhanced_imag = mask_real * noisy_i + mask_imag * noisy_r
            
        else:  # masking_mode == 'R'
            # Real mask
            real_mask = self.output_layer(torch.cat([x_real, x_imag], dim=1))  # [B, 1, F, T]
            real_mask = real_mask.squeeze(1)  # [B, F, T]
            
            enhanced_real = real_mask * noisy_real.squeeze(1)
            enhanced_imag = real_mask * noisy_imag.squeeze(1)
        
        # Combine real and imaginary parts
        enhanced_spec = torch.stack([enhanced_real, enhanced_imag], dim=-1)  # [B, F, T, 2]
        
        return enhanced_spec


def test_dccrn():
    """Test DCCRN model"""
    # Model parameters
    model = DCCRN(
        n_fft=512,
        hop_length=256,
        encoder_layers=5,
        hidden_dim=64,
        lstm_layers=2,
        masking_mode='E'
    )
    
    # Test input
    batch_size, freq_bins, time_steps = 2, 257, 100
    noisy_spec = torch.randn(batch_size, freq_bins, time_steps, 2)
    
    # Forward pass
    enhanced_spec = model(noisy_spec)
    
    print(f"Input shape: {noisy_spec.shape}")
    print(f"Output shape: {enhanced_spec.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


if __name__ == "__main__":
    test_dccrn()

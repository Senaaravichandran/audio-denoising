import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SISDRLoss(nn.Module):
    """Scale-Invariant Signal-to-Distortion Ratio Loss"""
    
    def __init__(self, zero_mean: bool = True, epsilon: float = 1e-8):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SI-SDR loss
        
        Args:
            pred: Predicted audio [B, T] or [B, F, T, 2] (complex spec)
            target: Target audio [B, T] or [B, F, T, 2] (complex spec)
        
        Returns:
            loss: SI-SDR loss (negative SI-SDR)
        """
        # Convert complex spectrograms to audio if needed
        if pred.dim() == 4 and pred.shape[-1] == 2:
            # Assuming we have a way to convert spec to audio
            # For now, compute loss on magnitude
            pred = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2 + self.epsilon)
            target = torch.sqrt(target[..., 0]**2 + target[..., 1]**2 + self.epsilon)
            pred = pred.flatten(1)  # [B, F*T]
            target = target.flatten(1)  # [B, F*T]
        
        # Zero mean
        if self.zero_mean:
            pred = pred - torch.mean(pred, dim=-1, keepdim=True)
            target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # Compute SI-SDR
        # s_target = <pred, target> * target / ||target||^2
        dot_product = torch.sum(pred * target, dim=-1, keepdim=True)
        target_energy = torch.sum(target**2, dim=-1, keepdim=True) + self.epsilon
        s_target = dot_product * target / target_energy
        
        # e_noise = pred - s_target
        e_noise = pred - s_target
        
        # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        s_target_energy = torch.sum(s_target**2, dim=-1) + self.epsilon
        e_noise_energy = torch.sum(e_noise**2, dim=-1) + self.epsilon
        
        si_sdr = 10 * torch.log10(s_target_energy / e_noise_energy)
        
        # Return negative SI-SDR as loss (we want to maximize SI-SDR)
        return -torch.mean(si_sdr)


class ComplexMSELoss(nn.Module):
    """Mean Squared Error loss for complex spectrograms"""
    
    def __init__(self, 
                 real_weight: float = 1.0, 
                 imag_weight: float = 1.0,
                 magnitude_weight: float = 0.0,
                 phase_weight: float = 0.0):
        super(ComplexMSELoss, self).__init__()
        self.real_weight = real_weight
        self.imag_weight = imag_weight
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute complex MSE loss
        
        Args:
            pred: Predicted complex spectrogram [B, F, T, 2]
            target: Target complex spectrogram [B, F, T, 2]
        
        Returns:
            loss: Complex MSE loss
        """
        pred_real, pred_imag = pred[..., 0], pred[..., 1]
        target_real, target_imag = target[..., 0], target[..., 1]
        
        # Real and imaginary MSE
        real_loss = F.mse_loss(pred_real, target_real)
        imag_loss = F.mse_loss(pred_imag, target_imag)
        
        loss = self.real_weight * real_loss + self.imag_weight * imag_loss
        
        # Optional magnitude and phase losses
        if self.magnitude_weight > 0:
            pred_mag = torch.sqrt(pred_real**2 + pred_imag**2 + 1e-8)
            target_mag = torch.sqrt(target_real**2 + target_imag**2 + 1e-8)
            mag_loss = F.mse_loss(pred_mag, target_mag)
            loss += self.magnitude_weight * mag_loss
        
        if self.phase_weight > 0:
            pred_phase = torch.atan2(pred_imag, pred_real + 1e-8)
            target_phase = torch.atan2(target_imag, target_real + 1e-8)
            # Wrap phase difference to [-π, π]
            phase_diff = pred_phase - target_phase
            phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
            phase_loss = F.mse_loss(phase_diff, torch.zeros_like(phase_diff))
            loss += self.phase_weight * phase_loss
        
        return loss


class SpectralLoss(nn.Module):
    """Multi-scale spectral loss"""
    
    def __init__(self, 
                 fft_sizes: list = [512, 1024, 2048],
                 hop_lengths: Optional[list] = None,
                 win_lengths: Optional[list] = None,
                 magnitude_weight: float = 1.0,
                 log_magnitude_weight: float = 1.0):
        super(SpectralLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_lengths = hop_lengths or [size // 4 for size in fft_sizes]
        self.win_lengths = win_lengths or fft_sizes
        self.magnitude_weight = magnitude_weight
        self.log_magnitude_weight = log_magnitude_weight
        
        # Create windows
        self.windows = {}
        for size in fft_sizes:
            self.windows[size] = torch.hann_window(size)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale spectral loss
        
        Args:
            pred: Predicted audio [B, T]
            target: Target audio [B, T]
        
        Returns:
            loss: Multi-scale spectral loss
        """
        loss = 0.0
        
        for fft_size, hop_length, win_length in zip(self.fft_sizes, self.hop_lengths, self.win_lengths):
            # Move window to same device as input
            window = self.windows[fft_size].to(pred.device)
            
            # Compute STFT
            pred_spec = torch.stft(
                pred, 
                n_fft=fft_size, 
                hop_length=hop_length, 
                win_length=win_length,
                window=window,
                return_complex=True
            )
            target_spec = torch.stft(
                target, 
                n_fft=fft_size, 
                hop_length=hop_length, 
                win_length=win_length,
                window=window,
                return_complex=True
            )
            
            # Compute magnitudes
            pred_mag = torch.abs(pred_spec)
            target_mag = torch.abs(target_spec)
            
            # Magnitude loss
            if self.magnitude_weight > 0:
                mag_loss = F.l1_loss(pred_mag, target_mag)
                loss += self.magnitude_weight * mag_loss
            
            # Log magnitude loss
            if self.log_magnitude_weight > 0:
                log_pred_mag = torch.log(pred_mag + 1e-5)
                log_target_mag = torch.log(target_mag + 1e-5)
                log_mag_loss = F.l1_loss(log_pred_mag, log_target_mag)
                loss += self.log_magnitude_weight * log_mag_loss
        
        return loss / len(self.fft_sizes)


class PerceptualLoss(nn.Module):
    """Perceptual loss using mel-scale spectrograms"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 80,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None):
        super(PerceptualLoss, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        
        # Create mel filterbank
        self.mel_scale = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using mel spectrograms
        
        Args:
            pred: Predicted audio [B, T]
            target: Target audio [B, T]
        
        Returns:
            loss: Perceptual loss
        """
        # Move mel transform to same device
        self.mel_scale = self.mel_scale.to(pred.device)
        
        # Compute mel spectrograms
        pred_mel = self.mel_scale(pred)
        target_mel = self.mel_scale(target)
        
        # Log mel spectrograms
        pred_log_mel = torch.log(pred_mel + 1e-5)
        target_log_mel = torch.log(target_mel + 1e-5)
        
        # L1 loss on log mel spectrograms
        return F.l1_loss(pred_log_mel, target_log_mel)


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components"""
    
    def __init__(self,
                 si_sdr_weight: float = 1.0,
                 complex_mse_weight: float = 0.5,
                 spectral_weight: float = 0.1,
                 perceptual_weight: float = 0.1,
                 **loss_kwargs):
        super(CombinedLoss, self).__init__()
        
        self.si_sdr_weight = si_sdr_weight
        self.complex_mse_weight = complex_mse_weight
        self.spectral_weight = spectral_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize loss functions
        if si_sdr_weight > 0:
            self.si_sdr_loss = SISDRLoss()
        
        if complex_mse_weight > 0:
            self.complex_mse_loss = ComplexMSELoss(**loss_kwargs.get('complex_mse', {}))
        
        if spectral_weight > 0:
            self.spectral_loss = SpectralLoss(**loss_kwargs.get('spectral', {}))
        
        if perceptual_weight > 0:
            try:
                import torchaudio
                self.perceptual_loss = PerceptualLoss(**loss_kwargs.get('perceptual', {}))
            except ImportError:
                print("Warning: torchaudio not available, skipping perceptual loss")
                self.perceptual_weight = 0
    
    def forward(self, 
                pred_spec: torch.Tensor, 
                target_spec: torch.Tensor,
                pred_audio: Optional[torch.Tensor] = None,
                target_audio: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss
        
        Args:
            pred_spec: Predicted complex spectrogram [B, F, T, 2]
            target_spec: Target complex spectrogram [B, F, T, 2]
            pred_audio: Predicted audio [B, T] (optional)
            target_audio: Target audio [B, T] (optional)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Complex MSE loss on spectrograms
        if self.complex_mse_weight > 0:
            complex_mse = self.complex_mse_loss(pred_spec, target_spec)
            loss_dict['complex_mse'] = complex_mse.item()
            total_loss += self.complex_mse_weight * complex_mse
        
        # SI-SDR loss on audio (if available)
        if self.si_sdr_weight > 0 and pred_audio is not None and target_audio is not None:
            si_sdr = self.si_sdr_loss(pred_audio, target_audio)
            loss_dict['si_sdr'] = si_sdr.item()
            total_loss += self.si_sdr_weight * si_sdr
        elif self.si_sdr_weight > 0:
            # Fallback: SI-SDR on magnitude spectrograms
            si_sdr = self.si_sdr_loss(pred_spec, target_spec)
            loss_dict['si_sdr'] = si_sdr.item()
            total_loss += self.si_sdr_weight * si_sdr
        
        # Spectral loss on audio
        if self.spectral_weight > 0 and pred_audio is not None and target_audio is not None:
            spectral = self.spectral_loss(pred_audio, target_audio)
            loss_dict['spectral'] = spectral.item()
            total_loss += self.spectral_weight * spectral
        
        # Perceptual loss on audio
        if self.perceptual_weight > 0 and pred_audio is not None and target_audio is not None:
            perceptual = self.perceptual_loss(pred_audio, target_audio)
            loss_dict['perceptual'] = perceptual.item()
            total_loss += self.perceptual_weight * perceptual
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def test_losses():
    """Test loss functions"""
    batch_size, freq_bins, time_steps = 4, 257, 100
    
    # Create dummy data
    pred_spec = torch.randn(batch_size, freq_bins, time_steps, 2)
    target_spec = torch.randn(batch_size, freq_bins, time_steps, 2)
    pred_audio = torch.randn(batch_size, 16000)
    target_audio = torch.randn(batch_size, 16000)
    
    # Test individual losses
    print("Testing individual losses:")
    
    # SI-SDR Loss
    si_sdr_loss = SISDRLoss()
    si_sdr = si_sdr_loss(pred_audio, target_audio)
    print(f"SI-SDR Loss: {si_sdr.item():.4f}")
    
    # Complex MSE Loss
    complex_mse_loss = ComplexMSELoss()
    complex_mse = complex_mse_loss(pred_spec, target_spec)
    print(f"Complex MSE Loss: {complex_mse.item():.4f}")
    
    # Spectral Loss
    spectral_loss = SpectralLoss()
    spectral = spectral_loss(pred_audio, target_audio)
    print(f"Spectral Loss: {spectral.item():.4f}")
    
    # Combined Loss
    print("\nTesting combined loss:")
    combined_loss = CombinedLoss()
    total_loss, loss_dict = combined_loss(
        pred_spec, target_spec, pred_audio, target_audio
    )
    print(f"Total Loss: {total_loss.item():.4f}")
    print("Loss components:", loss_dict)


if __name__ == "__main__":
    test_losses()

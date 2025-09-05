import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")


class AudioProcessor:
    """Audio processing utilities for DCCRN"""
    
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        window: str = 'hann',
        sample_rate: int = 16000,
        normalize: bool = True
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.sample_rate = sample_rate
        self.normalize = normalize
        
        # Create window
        if window == 'hann':
            self.window_fn = torch.hann_window(win_length)
        elif window == 'hamming':
            self.window_fn = torch.hamming_window(win_length)
        else:
            raise ValueError(f"Unsupported window type: {window}")
    
    def load_audio(self, file_path: Union[str, Path], target_sr: Optional[int] = None) -> torch.Tensor:
        """
        Load audio file and resample if necessary
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (uses self.sample_rate if None)
        
        Returns:
            audio: Audio tensor [T]
        """
        if target_sr is None:
            target_sr = self.sample_rate
            
        try:
            # Try torchaudio first
            audio, sr = torchaudio.load(str(file_path))
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            audio = audio.squeeze(0)  # Remove channel dimension
            
            # Resample if necessary
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                audio = resampler(audio)
                
        except Exception:
            # Fallback to librosa
            audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
            audio = torch.from_numpy(audio).float()
        
        # Normalize
        if self.normalize:
            audio = self.normalize_audio(audio)
            
        return audio
    
    def save_audio(self, audio: torch.Tensor, file_path: Union[str, Path], sample_rate: Optional[int] = None):
        """
        Save audio tensor to file
        
        Args:
            audio: Audio tensor [T]
            file_path: Output file path
            sample_rate: Sample rate (uses self.sample_rate if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # Ensure audio is on CPU
        if audio.is_cuda:
            audio = audio.cpu()
            
        # Normalize to [-1, 1] range
        audio = torch.clamp(audio, -1.0, 1.0)
        
        # Add channel dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Save using torchaudio
        torchaudio.save(str(file_path), audio, sample_rate)
    
    def normalize_audio(self, audio: torch.Tensor, method: str = 'peak') -> torch.Tensor:
        """
        Normalize audio
        
        Args:
            audio: Audio tensor
            method: Normalization method ('peak', 'rms', 'lufs')
        
        Returns:
            normalized_audio: Normalized audio tensor
        """
        if method == 'peak':
            # Peak normalization
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        elif method == 'rms':
            # RMS normalization
            rms = torch.sqrt(torch.mean(audio ** 2))
            if rms > 0:
                audio = audio / rms * 0.1  # Target RMS of 0.1
        elif method == 'lufs':
            # LUFS normalization (simplified)
            # This is a basic implementation, for more accurate LUFS use pyloudnorm
            mean_power = torch.mean(audio ** 2)
            if mean_power > 0:
                audio = audio / torch.sqrt(mean_power) * 0.1
        
        return audio
    
    def add_padding(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Pad or trim audio to target length
        
        Args:
            audio: Audio tensor [T]
            target_length: Target length in samples
        
        Returns:
            padded_audio: Audio tensor [target_length]
        """
        current_length = audio.shape[0]
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            audio = torch.nn.functional.pad(audio, (0, padding))
        elif current_length > target_length:
            # Trim from the end
            audio = audio[:target_length]
            
        return audio
    
    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute Short-Time Fourier Transform
        
        Args:
            audio: Audio tensor [T] or [B, T]
        
        Returns:
            spec: Complex spectrogram [B, F, T, 2] where last dim is [real, imag]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension
            
        # Compute STFT
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_fn,
            return_complex=True,
            pad_mode='reflect'
        )
        
        # Convert to real-imaginary representation
        spec_real = spec.real
        spec_imag = spec.imag
        spec = torch.stack([spec_real, spec_imag], dim=-1)  # [B, F, T, 2]
        
        return spec
    
    def istft(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute Inverse Short-Time Fourier Transform
        
        Args:
            spec: Complex spectrogram [B, F, T, 2] where last dim is [real, imag]
        
        Returns:
            audio: Audio tensor [B, T]
        """
        # Convert to complex tensor
        spec_complex = torch.complex(spec[..., 0], spec[..., 1])
        
        # Compute ISTFT
        audio = torch.istft(
            spec_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window_fn,
            length=None
        )
        
        return audio
    
    def compute_magnitude(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute magnitude spectrogram
        
        Args:
            spec: Complex spectrogram [B, F, T, 2]
        
        Returns:
            magnitude: Magnitude spectrogram [B, F, T]
        """
        real, imag = spec[..., 0], spec[..., 1]
        magnitude = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        return magnitude
    
    def compute_phase(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute phase spectrogram
        
        Args:
            spec: Complex spectrogram [B, F, T, 2]
        
        Returns:
            phase: Phase spectrogram [B, F, T]
        """
        real, imag = spec[..., 0], spec[..., 1]
        phase = torch.atan2(imag, real + 1e-8)
        return phase
    
    def apply_dynamic_range_compression(
        self, 
        spec: torch.Tensor, 
        power: float = 0.3,
        log_offset: float = 1e-6
    ) -> torch.Tensor:
        """
        Apply dynamic range compression to magnitude spectrogram
        
        Args:
            spec: Magnitude spectrogram [B, F, T]
            power: Compression power
            log_offset: Small value to avoid log(0)
        
        Returns:
            compressed_spec: Compressed spectrogram
        """
        return torch.log(spec + log_offset) ** power
    
    def remove_dc_component(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Remove DC component from audio
        
        Args:
            audio: Audio tensor
        
        Returns:
            audio_no_dc: Audio with DC component removed
        """
        return audio - torch.mean(audio, dim=-1, keepdim=True)
    
    def apply_preemphasis(self, audio: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
        """
        Apply preemphasis filter
        
        Args:
            audio: Audio tensor [B, T] or [T]
            coeff: Preemphasis coefficient
        
        Returns:
            filtered_audio: Preemphasized audio
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Apply preemphasis: y[n] = x[n] - coeff * x[n-1]
        filtered = torch.zeros_like(audio)
        filtered[:, 0] = audio[:, 0]
        filtered[:, 1:] = audio[:, 1:] - coeff * audio[:, :-1]
        
        return filtered.squeeze(0) if filtered.shape[0] == 1 else filtered


def test_audio_processor():
    """Test AudioProcessor functionality"""
    processor = AudioProcessor()
    
    # Create dummy audio
    audio = torch.randn(16000)  # 1 second at 16kHz
    print(f"Input audio shape: {audio.shape}")
    
    # Test STFT
    spec = processor.stft(audio)
    print(f"Spectrogram shape: {spec.shape}")
    
    # Test ISTFT
    reconstructed = processor.istft(spec)
    print(f"Reconstructed audio shape: {reconstructed.shape}")
    
    # Test magnitude and phase
    magnitude = processor.compute_magnitude(spec)
    phase = processor.compute_phase(spec)
    print(f"Magnitude shape: {magnitude.shape}")
    print(f"Phase shape: {phase.shape}")
    
    # Test reconstruction error
    mse = torch.mean((audio - reconstructed.squeeze()) ** 2)
    print(f"Reconstruction MSE: {mse.item():.6f}")


if __name__ == "__main__":
    test_audio_processor()

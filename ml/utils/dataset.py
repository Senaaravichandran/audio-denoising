import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from .audio_utils import AudioProcessor
import librosa


class AudioPairDataset(Dataset):
    """Dataset for paired clean/noisy audio files"""
    
    def __init__(
        self,
        clean_dir: Union[str, Path],
        noisy_dir: Union[str, Path],
        processor: AudioProcessor,
        segment_length: Optional[int] = None,
        overlap: float = 0.5,
        augmentation: bool = True,
        cache_spectrograms: bool = False,
        min_snr: float = -10.0,
        max_snr: float = 10.0
    ):
        """
        Initialize dataset
        
        Args:
            clean_dir: Directory containing clean audio files
            noisy_dir: Directory containing noisy audio files
            processor: AudioProcessor instance
            segment_length: Length of audio segments in samples (None for full files)
            overlap: Overlap ratio for segmentation
            augmentation: Whether to apply data augmentation
            cache_spectrograms: Whether to cache spectrograms in memory
            min_snr: Minimum SNR for dynamic noise mixing
            max_snr: Maximum SNR for dynamic noise mixing
        """
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.processor = processor
        self.segment_length = segment_length
        self.overlap = overlap
        self.augmentation = augmentation
        self.cache_spectrograms = cache_spectrograms
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.training = True  # Add training flag for augmentation
        
        # Find paired files
        self.file_pairs = self._find_file_pairs()
        
        # Generate segments if segment_length is specified
        if segment_length is not None:
            self.segments = self._generate_segments()
        else:
            self.segments = [(i, 0, None) for i in range(len(self.file_pairs))]
        
        # Cache for spectrograms
        self._spectrogram_cache = {} if cache_spectrograms else None
        
        print(f"Found {len(self.file_pairs)} file pairs")
        print(f"Generated {len(self.segments)} segments")
    
    def set_training(self, training: bool):
        """Set training mode for augmentation"""
        self.training = training
    
    def _find_file_pairs(self) -> List[Tuple[Path, Path]]:
        """Find pairs of clean and noisy files"""
        clean_files = list(self.clean_dir.glob("*.wav"))
        clean_files.extend(list(self.clean_dir.glob("*.flac")))
        clean_files.extend(list(self.clean_dir.glob("*.mp3")))
        
        noisy_files = list(self.noisy_dir.glob("*.wav"))
        noisy_files.extend(list(self.noisy_dir.glob("*.flac")))
        noisy_files.extend(list(self.noisy_dir.glob("*.mp3")))
        
        # Create mapping based on filename (without extension)
        clean_dict = {f.stem: f for f in clean_files}
        noisy_dict = {f.stem: f for f in noisy_files}
        
        # Find matching pairs
        pairs = []
        for stem in clean_dict:
            if stem in noisy_dict:
                pairs.append((clean_dict[stem], noisy_dict[stem]))
        
        if len(pairs) == 0:
            raise ValueError("No matching pairs found. Check file naming convention.")
        
        return pairs
    
    def _generate_segments(self) -> List[Tuple[int, int, Optional[int]]]:
        """Generate segments from audio files"""
        segments = []
        
        for file_idx, (clean_path, noisy_path) in enumerate(self.file_pairs):
            # Get audio length
            try:
                audio_info = librosa.get_samplerate(str(clean_path))
                duration = librosa.get_duration(path=str(clean_path))
                audio_length = int(duration * self.processor.sample_rate)
            except:
                # Fallback: load audio to get length
                audio = self.processor.load_audio(clean_path)
                audio_length = len(audio)
            
            if audio_length <= self.segment_length:
                # File is shorter than segment length, use entire file
                segments.append((file_idx, 0, audio_length))
            else:
                # Generate overlapping segments
                step_size = int(self.segment_length * (1 - self.overlap))
                start = 0
                
                while start + self.segment_length <= audio_length:
                    segments.append((file_idx, start, start + self.segment_length))
                    start += step_size
                
                # Add final segment if there's remaining audio
                if start < audio_length:
                    segments.append((file_idx, audio_length - self.segment_length, audio_length))
        
        return segments
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample
        
        Returns:
            noisy_spec: Noisy spectrogram [F, T, 2]
            clean_spec: Clean spectrogram [F, T, 2]
        """
        file_idx, start_sample, end_sample = self.segments[idx]
        clean_path, noisy_path = self.file_pairs[file_idx]
        
        # Check cache first
        cache_key = (file_idx, start_sample, end_sample)
        if self._spectrogram_cache is not None and cache_key in self._spectrogram_cache:
            return self._spectrogram_cache[cache_key]
        
        # Load audio segments
        clean_audio = self._load_audio_segment(clean_path, start_sample, end_sample)
        noisy_audio = self._load_audio_segment(noisy_path, start_sample, end_sample)
        
        # Apply augmentation only during training
        if self.augmentation and getattr(self, 'training', True):
            clean_audio, noisy_audio = self._apply_augmentation(clean_audio, noisy_audio)
        
        # Convert to spectrograms
        clean_spec = self.processor.stft(clean_audio).squeeze(0)  # Remove batch dim
        noisy_spec = self.processor.stft(noisy_audio).squeeze(0)  # Remove batch dim
        
        # Cache if enabled
        if self._spectrogram_cache is not None:
            self._spectrogram_cache[cache_key] = (noisy_spec, clean_spec)
        
        return noisy_spec, clean_spec
    
    def _load_audio_segment(
        self, 
        file_path: Path, 
        start_sample: Optional[int], 
        end_sample: Optional[int]
    ) -> torch.Tensor:
        """Load audio segment from file"""
        # Load full audio
        audio = self.processor.load_audio(file_path)
        
        # Extract segment
        if start_sample is not None and end_sample is not None:
            audio = audio[start_sample:end_sample]
        
        # Pad if necessary
        if self.segment_length is not None and len(audio) < self.segment_length:
            audio = self.processor.add_padding(audio, self.segment_length)
        
        return audio
    
    def _apply_augmentation(
        self, 
        clean_audio: torch.Tensor, 
        noisy_audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation"""
        
        # 1. Random gain
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            clean_audio = clean_audio * gain
            noisy_audio = noisy_audio * gain
        
        # 2. Random SNR adjustment
        if random.random() < 0.3:
            target_snr = random.uniform(self.min_snr, self.max_snr)
            noisy_audio = self._adjust_snr(clean_audio, noisy_audio, target_snr)
        
        # 3. Time shifting
        if random.random() < 0.3:
            max_shift = min(len(clean_audio) // 10, 1600)  # Max 0.1s shift
            shift = random.randint(-max_shift, max_shift)
            
            if shift > 0:
                clean_audio = torch.cat([torch.zeros(shift), clean_audio[:-shift]])
                noisy_audio = torch.cat([torch.zeros(shift), noisy_audio[:-shift]])
            elif shift < 0:
                clean_audio = torch.cat([clean_audio[-shift:], torch.zeros(-shift)])
                noisy_audio = torch.cat([noisy_audio[-shift:], torch.zeros(-shift)])
        
        # 4. Random phase shift
        if random.random() < 0.2:
            phase_shift = random.uniform(-np.pi, np.pi)
            # Apply phase shift in frequency domain
            clean_spec = self.processor.stft(clean_audio)
            clean_phase = self.processor.compute_phase(clean_spec) + phase_shift
            clean_magnitude = self.processor.compute_magnitude(clean_spec)
            
            # Reconstruct with shifted phase
            clean_real = clean_magnitude * torch.cos(clean_phase)
            clean_imag = clean_magnitude * torch.sin(clean_phase)
            clean_spec_shifted = torch.stack([clean_real, clean_imag], dim=-1)
            clean_audio = self.processor.istft(clean_spec_shifted)
        
        return clean_audio, noisy_audio
    
    def _adjust_snr(
        self, 
        clean: torch.Tensor, 
        noisy: torch.Tensor, 
        target_snr: float
    ) -> torch.Tensor:
        """Adjust SNR of noisy signal"""
        noise = noisy - clean
        
        # Calculate current powers
        clean_power = torch.mean(clean ** 2)
        noise_power = torch.mean(noise ** 2)
        
        if noise_power == 0:
            return noisy
        
        # Calculate required noise scaling
        target_noise_power = clean_power / (10 ** (target_snr / 10))
        noise_scale = torch.sqrt(target_noise_power / noise_power)
        
        # Apply scaling
        scaled_noise = noise * noise_scale
        adjusted_noisy = clean + scaled_noise
        
        return adjusted_noisy
    
    def train(self):
        """Set dataset to training mode"""
        self.training = True
    
    def eval(self):
        """Set dataset to evaluation mode"""
        self.training = False


class CollateFunction:
    """Custom collate function for variable length sequences"""
    
    def __init__(self, pad_to_max: bool = True):
        self.pad_to_max = pad_to_max
    
    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Collate batch of spectrograms
        
        Args:
            batch: List of (noisy_spec, clean_spec) tuples
        
        Returns:
            noisy_batch: Batched noisy spectrograms [B, F, T, 2]
            clean_batch: Batched clean spectrograms [B, F, T, 2]
        """
        noisy_specs, clean_specs = zip(*batch)
        
        if self.pad_to_max:
            # Find maximum time dimension
            max_time = max(spec.shape[1] for spec in noisy_specs)
            
            # Pad all spectrograms to max length
            noisy_padded = []
            clean_padded = []
            
            for noisy, clean in zip(noisy_specs, clean_specs):
                freq_bins, time_steps, channels = noisy.shape
                
                if time_steps < max_time:
                    pad_width = max_time - time_steps
                    noisy = torch.nn.functional.pad(noisy, (0, 0, 0, pad_width))
                    clean = torch.nn.functional.pad(clean, (0, 0, 0, pad_width))
                
                noisy_padded.append(noisy)
                clean_padded.append(clean)
            
            noisy_batch = torch.stack(noisy_padded)
            clean_batch = torch.stack(clean_padded)
        else:
            # Stack without padding (all sequences must have same length)
            noisy_batch = torch.stack(noisy_specs)
            clean_batch = torch.stack(clean_specs)
        
        return noisy_batch, clean_batch


def create_dataloaders(
    clean_train_dir: Union[str, Path],
    noisy_train_dir: Union[str, Path],
    clean_val_dir: Union[str, Path],
    noisy_val_dir: Union[str, Path],
    processor: AudioProcessor,
    batch_size: int = 8,
    num_workers: int = 4,
    segment_length: Optional[int] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    # Create datasets
    train_dataset = AudioPairDataset(
        clean_train_dir, 
        noisy_train_dir, 
        processor, 
        segment_length=segment_length,
        augmentation=True,
        **dataset_kwargs
    )
    
    val_dataset = AudioPairDataset(
        clean_val_dir, 
        noisy_val_dir, 
        processor, 
        segment_length=segment_length,
        augmentation=False,
        **dataset_kwargs
    )
    
    # Set training mode
    train_dataset.train()
    val_dataset.eval()
    
    # Create collate function
    collate_fn = CollateFunction(pad_to_max=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader


def test_dataset():
    """Test dataset functionality"""
    from audio_utils import AudioProcessor
    
    # Create processor
    processor = AudioProcessor()
    
    # Create dummy data directories (would be your actual data paths)
    clean_dir = "../../data/clean"
    noisy_dir = "../../data/noisy"
    
    try:
        # Create dataset
        dataset = AudioPairDataset(
            clean_dir, 
            noisy_dir, 
            processor,
            segment_length=32000,  # 2 seconds
            augmentation=True
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test single sample
        noisy_spec, clean_spec = dataset[0]
        print(f"Noisy spec shape: {noisy_spec.shape}")
        print(f"Clean spec shape: {clean_spec.shape}")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=CollateFunction())
        
        for batch_idx, (noisy_batch, clean_batch) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Noisy batch shape: {noisy_batch.shape}")
            print(f"  Clean batch shape: {clean_batch.shape}")
            break
            
    except Exception as e:
        print(f"Test failed (expected if data directories don't exist): {e}")


if __name__ == "__main__":
    test_dataset()

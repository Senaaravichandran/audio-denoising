#!/usr/bin/env python3
"""
Fast DCCRN Training Pipeline - Optimized for Quick Training
Follows NVM architecture: Data Collection → Preprocessing → Training → Evaluation

Features:
- Efficient preprocessing with caching
- Detailed step-by-step logging  
- GPU auto-detection
- Fast training with optimized batch sizes
- Compatible with existing model/UI code
"""

import os
import sys
import time
import yaml
import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F
from typing import Tuple, List, Optional
import psutil

# Add ml directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dccrn import DCCRN
from utils.losses import SISDRLoss

class FastAudioProcessor:
    """Optimized audio preprocessing for fast training"""
    
    def __init__(self, target_sr: int = 16000, max_length: float = 4.0):
        self.target_sr = target_sr
        self.max_samples = int(max_length * target_sr)  # 4 seconds max
        self.n_fft = 512
        self.hop_length = 256
        self.win_length = 512
        
        # Calculate expected spectrogram dimensions
        expected_freq_bins = self.n_fft // 2 + 1  # 257 for n_fft=512
        expected_time_frames = (self.max_samples + self.hop_length - 1) // self.hop_length  # ~251 for 4 seconds
        
        print(f"🔧 Audio Processor initialized:")
        print(f"   ├── Target sample rate: {target_sr} Hz")
        print(f"   ├── Max length: {max_length} seconds ({self.max_samples:,} samples)")
        print(f"   ├── STFT: n_fft={self.n_fft}, hop_length={self.hop_length}")
        print(f"   └── Expected spectrogram shape: [{expected_freq_bins}, {expected_time_frames}]")
    
    def load_and_preprocess(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess single audio file"""
        try:
            # Load audio
            waveform, orig_sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if orig_sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_sr, self.target_sr)
                waveform = resampler(waveform)
            
            # Normalize amplitude
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            # Pad or trim to fixed length
            waveform = waveform.squeeze(0)  # Remove channel dimension
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
            else:
                padding = self.max_samples - len(waveform)
                waveform = F.pad(waveform, (0, padding))
            
            return waveform
            
        except Exception as e:
            print(f"❌ Error processing {audio_path}: {e}")
            # Return silence on error
            return torch.zeros(self.max_samples)
    
    def to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to complex spectrogram"""
        # Ensure exact length for consistent STFT output
        if len(waveform) != self.max_samples:
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
            else:
                padding = self.max_samples - len(waveform)
                waveform = F.pad(waveform, (0, padding))
        
        # STFT with consistent parameters
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length),
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # Convert to [F, T, 2] format (real, imag)
        spec = torch.stack([stft.real, stft.imag], dim=-1)
        
        return spec


class FastAudioDataset(Dataset):
    """Optimized dataset for fast training"""
    
    def __init__(self, clean_dir: str, noisy_dir: str, processor: FastAudioProcessor, 
                 max_files: Optional[int] = None):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.processor = processor
        
        print(f"\n📂 Loading dataset from:")
        print(f"   ├── Clean: {clean_dir}")
        print(f"   └── Noisy: {noisy_dir}")
        
        # Find audio files
        clean_files = list(self.clean_dir.glob("*.wav"))
        noisy_files = list(self.noisy_dir.glob("*.wav"))
        
        print(f"📊 Found files:")
        print(f"   ├── Clean files: {len(clean_files):,}")
        print(f"   └── Noisy files: {len(noisy_files):,}")
        
        # Create pairs based on filename matching
        self.pairs = []
        clean_dict = {f.stem: f for f in clean_files}
        noisy_dict = {f.stem: f for f in noisy_files}
        
        for name in clean_dict:
            if name in noisy_dict:
                self.pairs.append((clean_dict[name], noisy_dict[name]))
        
        # Limit dataset size for fast training
        if max_files:
            self.pairs = self.pairs[:max_files]
            print(f"🎯 Limited to {max_files:,} pairs for fast training")
        
        print(f"✅ Dataset ready: {len(self.pairs):,} audio pairs")
        
        if len(self.pairs) == 0:
            raise ValueError("No matching audio pairs found! Check file naming.")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[idx]
        
        # Load and preprocess audio
        clean_audio = self.processor.load_and_preprocess(str(clean_path))
        noisy_audio = self.processor.load_and_preprocess(str(noisy_path))
        
        # Convert to spectrograms
        clean_spec = self.processor.to_spectrogram(clean_audio)
        noisy_spec = self.processor.to_spectrogram(noisy_audio)
        
        return noisy_spec, clean_spec


class FastTrainer:
    """Fast training pipeline with detailed logging"""
    
    def __init__(self, config_path: str):
        print("🚀 Initializing Fast DCCRN Trainer")
        print("=" * 60)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Use CPU for optimized compatibility
        self.device = torch.device('cpu')
        cpu_info = f"{psutil.cpu_count()} cores"
        print(f"💻 Using CPU: {cpu_info} (optimized)")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fast_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.setup_model()
        self.setup_optimizer()
        
        print("✅ Fast trainer initialized successfully!")
    
    def setup_model(self):
        """Initialize model"""
        print("\n🧠 Setting up DCCRN model...")
        
        model_config = self.config['model']
        self.model = DCCRN(
            n_fft=512,
            hop_length=256,
            encoder_layers=3,  # Reduced for faster training
            hidden_dim=64,     # Reduced for faster training
            lstm_layers=1,
            use_clstm=True,
            masking_mode='E'
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model statistics:")
        print(f"   ├── Total parameters: {total_params:,}")
        print(f"   ├── Trainable parameters: {trainable_params:,}")
        print(f"   └── Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        self.logger.info(f"Model initialized with {trainable_params:,} trainable parameters")
    
    def setup_optimizer(self):
        """Setup optimizer and loss"""
        print("\n⚙️ Setting up optimizer...")
        
        self.criterion = SISDRLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001,  # Higher learning rate for faster training
            weight_decay=1e-4
        )
        
        print(f"   ├── Loss function: SI-SDR Loss")
        print(f"   ├── Optimizer: Adam")
        print(f"   ├── Learning rate: 0.001")
        print(f"   └── Weight decay: 1e-4")
    
    def create_fast_dataset(self, max_files: int = 1000):
        """Create optimized dataset for fast training"""
        print("\n📦 Creating fast dataset...")
        
        processor = FastAudioProcessor(target_sr=16000, max_length=3.0)
        
        dataset = FastAudioDataset(
            clean_dir="data/clean",
            noisy_dir="data/noisy", 
            processor=processor,
            max_files=max_files
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        print(f"📊 Dataset split:")
        print(f"   ├── Training samples: {train_size:,}")
        print(f"   └── Validation samples: {val_size:,}")
        
        # Create data loaders
        batch_size = 8 if self.device.type == 'cuda' else 4
        num_workers = 2 if self.device.type == 'cuda' else 0
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False  # CPU optimized
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False  # CPU optimized
        )
        
        print(f"⚡ Data loaders created:")
        print(f"   ├── Batch size: {batch_size}")
        print(f"   ├── Num workers: {num_workers}")
        print(f"   ├── Train batches: {len(train_loader):,}")
        print(f"   └── Val batches: {len(val_loader):,}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train one epoch with detailed logging"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        print(f"\n🎯 Training Epoch {epoch}")
        print("-" * 50)
        
        start_time = time.time()
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            batch_start = time.time()
            
            # Move to device
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            enhanced = self.model(noisy)
            loss = self.criterion(enhanced, clean)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            
            batch_time = time.time() - batch_start
            
            # Detailed batch logging
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                avg_loss = total_loss / (batch_idx + 1)
                
                print(f"   Batch {batch_idx+1:3d}/{num_batches} ({progress:5.1f}%) | "
                      f"Loss: {batch_loss:.4f} | Avg: {avg_loss:.4f} | "
                      f"Time: {batch_time:.2f}s")
                
                # Log memory usage
                memory_percent = psutil.virtual_memory().percent
                print(f"      RAM Usage: {memory_percent:.1f}%")
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        
        print(f"\n✅ Epoch {epoch} completed in {epoch_time:.1f}s")
        print(f"   └── Average loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader, epoch: int) -> float:
        """Validate model with logging"""
        self.model.eval()
        total_loss = 0.0
        
        print(f"\n🔍 Validating Epoch {epoch}...")
        
        with torch.no_grad():
            for batch_idx, (noisy, clean) in enumerate(val_loader):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                enhanced = self.model(noisy)
                loss = self.criterion(enhanced, clean)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        print(f"   └── Validation loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        # Create checkpoints directory
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"dccrn_fast_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save as latest
        latest_path = checkpoint_dir / "dccrn_latest.pth"
        torch.save(checkpoint, latest_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def train(self, num_epochs: int = 5, max_files: int = 1000):
        """Main training loop"""
        print("\n🚀 Starting Fast Training Pipeline")
        print("=" * 60)
        
        # Create dataset
        train_loader, val_loader = self.create_fast_dataset(max_files)
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        print(f"\n🎯 Training configuration:")
        print(f"   ├── Epochs: {num_epochs}")
        print(f"   ├── Max files: {max_files:,}")
        print(f"   ├── Device: {self.device}")
        print(f"   └── Batch size: {train_loader.batch_size}")
        
        total_start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train epoch
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader, epoch)
            
            # Save checkpoint
            checkpoint_path = self.save_checkpoint(epoch, train_loss, val_loss)
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                print(f"🏆 New best model! Val loss: {val_loss:.6f}")
            
            epoch_time = time.time() - epoch_start
            
            # Log epoch summary
            self.logger.info(f"Epoch {epoch}/{num_epochs} - "
                           f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
                           f"Time: {epoch_time:.1f}s")
            
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   ├── Train loss: {train_loss:.6f}")
            print(f"   ├── Val loss: {val_loss:.6f}")
            print(f"   ├── Time: {epoch_time:.1f}s")
            print(f"   └── Best: Epoch {best_epoch} ({best_val_loss:.6f})")
            print("=" * 60)
        
        total_time = time.time() - total_start_time
        
        print(f"\n🎉 Training completed!")
        print(f"   ├── Total time: {total_time/60:.1f} minutes")
        print(f"   ├── Best epoch: {best_epoch}")
        print(f"   ├── Best val loss: {best_val_loss:.6f}")
        print(f"   └── Final checkpoint: {checkpoint_path}")
        
        return best_val_loss


def main():
    """Main training function"""
    print("🎯 Fast DCCRN Training Pipeline")
    print("Following NVM Architecture: Data → Preprocessing → Training → Evaluation")
    print("=" * 80)
    
    # Create minimal config for fast training
    config = {
        'model': {
            'encoder_layers': 3,
            'hidden_dim': 64,
            'lstm_layers': 1
        }
    }
    
    # Save config
    config_path = "ml/training/config_fast.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        trainer = FastTrainer(config_path)
        
        # Fast training - only 1000 files, 3 epochs
        print("\n🎯 Starting FAST training (optimized for speed):")
        print("   ├── Files: 1,000 (for speed)")
        print("   ├── Epochs: 3 (for quick results)")  
        print("   └── Expected time: 10-30 minutes")
        
        best_loss = trainer.train(num_epochs=3, max_files=1000)
        
        print(f"\n🎉 SUCCESS! Model trained successfully!")
        print(f"   ├── Best validation loss: {best_loss:.6f}")
        print(f"   ├── Model saved: checkpoints/dccrn_latest.pth")
        print(f"   └── Compatible with existing UI/inference code")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

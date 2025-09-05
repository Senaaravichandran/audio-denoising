"""
Fast Training Script for Lightweight DCCRN
Optimized for low-resource systems (Intel i3, 8GB RAM, Limited Storage)
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchaudio
import numpy as np
from pathlib import Path
import logging
import time
from typing import Dict, Tuple, Optional

# Add ml directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dccrn_lightweight import LightweightDCCRN
from utils.audio_utils import AudioProcessor
from utils.dataset import PairedAudioDataset
from utils.losses import SISDRLoss

class FastTrainer:
    """Fast trainer optimized for low-resource systems"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.device = torch.device('cpu')  # Force CPU for compatibility
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.audio_processor = AudioProcessor(
            sample_rate=self.config['data']['sample_rate'],
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length'],
            win_length=self.config['data']['win_length']
        )
        
        self.model = LightweightDCCRN(**self.config['model']).to(self.device)
        self.criterion = SISDRLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Model initialized with {self.model.get_model_size():,} parameters")
        self.logger.info(f"Estimated memory usage: {self.model.get_memory_usage():.2f} MB")
    
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_limited_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare a limited dataset for fast training"""
        
        clean_dir = Path(self.config['data']['clean_dir'])
        noisy_dir = Path(self.config['data']['noisy_dir'])
        
        # Get limited file list
        max_files = self.config['processing']['max_files_per_subset']
        clean_files = sorted(list(clean_dir.glob('*.wav')))[:max_files]
        noisy_files = sorted(list(noisy_dir.glob('*.wav')))[:max_files]
        
        self.logger.info(f"Using {len(clean_files)} clean files and {len(noisy_files)} noisy files")
        
        # Create dataset
        dataset = PairedAudioDataset(
            clean_files=clean_files,
            noisy_files=noisy_files,
            audio_processor=self.audio_processor,
            max_length=self.config['data']['max_length'],
            normalize=self.config['processing']['normalize_audio']
        )
        
        # Split dataset
        val_size = int(len(dataset) * self.config['validation']['split_ratio'])
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders with minimal workers
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,  # No multiprocessing to save memory
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['validation']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            enhanced = self.model(noisy)
            
            # Calculate loss
            loss = self.criterion(enhanced, clean)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clip_norm']
            )
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % self.config['logging']['console_log_interval'] == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                    f'Loss: {loss.item():.6f}'
                )
            
            # Force garbage collection to save memory
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                enhanced = self.model(noisy)
                loss = self.criterion(enhanced, clean)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f'New best model saved with loss: {loss:.6f}')
        
        # Cleanup old checkpoints to save space
        if self.config['checkpoint']['keep_best_only'] and not is_best:
            try:
                os.remove(checkpoint_path)
            except:
                pass
    
    def train(self):
        """Main training loop"""
        start_time = time.time()
        
        # Prepare dataset
        self.logger.info("Preparing dataset...")
        train_loader, val_loader = self.prepare_limited_dataset()
        
        # Training loop
        best_loss = float('inf')
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            # Log results
            self.logger.info(
                f'Epoch {epoch}/{num_epochs} - '
                f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                f'Time: {epoch_time:.2f}s'
            )
            
            # Save checkpoint
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            
            if epoch % self.config['checkpoint']['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time/60:.2f} minutes')
        self.logger.info(f'Best validation loss: {best_loss:.6f}')
        
        return best_loss

def main():
    """Main function"""
    # Use lightweight config
    config_path = os.path.join(os.path.dirname(__file__), 'config_lightweight.yaml')
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    print("Starting fast training for lightweight DCCRN...")
    print("Optimized for low-resource systems")
    
    try:
        trainer = FastTrainer(config_path)
        best_loss = trainer.train()
        print(f"\nTraining completed successfully!")
        print(f"Best validation loss: {best_loss:.6f}")
        print(f"Model saved in: {trainer.checkpoint_dir}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

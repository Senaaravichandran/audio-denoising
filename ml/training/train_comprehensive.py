#!/usr/bin/env python3
"""
Comprehensive DCCRN Training Script
Training on ALL 23,075+ audio files for maximum model performance
Optimized for Intel i3-7020U with 8GB RAM
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
import json
from typing import Dict, Tuple, Optional
import gc
import psutil

# Add ml directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dccrn import DCCRN
from utils.audio_utils import AudioProcessor
from utils.dataset import AudioPairDataset
from utils.losses import SISDRLoss

class ComprehensiveTrainer:
    """Comprehensive trainer for full dataset training"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.device = torch.device('cpu')  # CPU optimized for i3
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['output']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set CPU optimization
        torch.set_num_threads(self.config['performance']['torch_threads'])
        
        # Initialize components
        self.audio_processor = AudioProcessor(
            sample_rate=self.config['data']['sample_rate'],
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length'],
            win_length=self.config['data']['win_length']
        )
        
        # Initialize model
        model_config = self.config['model']
        self.model = DCCRN(
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length'],
            win_length=self.config['data']['win_length'],
            encoder_layers=model_config['encoder_layers'],
            hidden_dim=model_config['lstm_hidden_size'],
            lstm_layers=model_config['lstm_layers'],
            use_clstm=model_config['use_clstm'],
            kernel_size=tuple(model_config['kernel_size']),
            stride=(2, 1),
            use_cbn=model_config['use_cbn'],
            masking_mode=model_config['masking_mode']
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = SISDRLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config['training']['patience'],
            factor=self.config['training']['factor'],
            min_lr=self.config['training']['min_lr']
        )
        
        # Create directories
        self.checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.steps_since_improvement = 0
        
        # Load metadata
        self.load_training_metadata()
        
        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        self.logger.info(f"Training dataset size: {self.training_pairs:,} pairs")
    
    def load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_training_metadata(self):
        """Load metadata from data preparation"""
        metadata_path = Path('data/training_metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.training_pairs = metadata['total_pairs']
                self.avg_duration = metadata['avg_duration']
                self.logger.info(f"Loaded metadata: {self.training_pairs:,} training pairs")
        else:
            self.logger.warning("No training metadata found. Estimating dataset size...")
            # Estimate based on file counts
            clean_files = len(list(Path('data/clean').glob('*.wav')))
            self.training_pairs = clean_files * 4  # 4 noise variants per file
    
    def prepare_comprehensive_dataset(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare comprehensive dataset for training"""
        
        self.logger.info("Loading comprehensive dataset...")
        
        clean_dir = Path(self.config['data']['clean_dir'])
        noisy_dir = Path(self.config['data']['noisy_dir'])
        
        # Get all file pairs
        clean_files = sorted(list(clean_dir.glob('*.wav')))
        noisy_files = sorted(list(noisy_dir.glob('*.wav')))
        
        self.logger.info(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")
        
        # Create paired dataset
        dataset = AudioPairDataset(
            clean_dir=clean_dir,
            noisy_dir=noisy_dir,
            processor=self.audio_processor,
            segment_length=int(self.config['data']['max_length'] * self.config['data']['sample_rate']),
            augmentation=self.config['data']['normalize']
        )
        
        # Split dataset
        train_size = int(len(dataset) * self.config['data']['train_split'])
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        self.logger.info(f"Training samples: {len(train_dataset):,}")
        self.logger.info(f"Validation samples: {len(val_dataset):,}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory'],
            prefetch_factor=self.config['data']['prefetch_factor']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        self.logger.info(f"Starting epoch {epoch} with {num_batches:,} batches")
        
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
                self.config['training']['gradient_clip']
            )
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % self.config['output']['log_interval'] == 0:
                memory_usage = psutil.virtual_memory().percent
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx:,}/{num_batches:,} '
                    f'({batch_idx/num_batches*100:.1f}%), '
                    f'Loss: {loss.item():.6f}, '
                    f'Memory: {memory_usage:.1f}%'
                )
            
            # Memory cleanup
            if batch_idx % self.config['system']['cleanup_interval'] == 0:
                gc.collect()
                
            # Save checkpoint periodically
            if batch_idx % self.config['system']['checkpoint_interval'] == 0 and batch_idx > 0:
                self.save_checkpoint(epoch, loss.item(), is_best=False)
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        self.logger.info("Running validation...")
        
        with torch.no_grad():
            for batch_idx, (noisy, clean) in enumerate(val_loader):
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                enhanced = self.model(noisy)
                loss = self.criterion(enhanced, clean)
                total_loss += loss.item()
                
                # Log validation progress
                if batch_idx % 100 == 0:
                    self.logger.info(f"Validation batch {batch_idx}/{len(val_loader)}")
        
        avg_loss = total_loss / len(val_loader)
        self.logger.info(f"Validation completed. Average loss: {avg_loss:.6f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        if epoch % self.config['output']['save_every_n_epochs'] == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f'🏆 New best model saved! Loss: {loss:.6f}')
        
        # Cleanup old checkpoints to save space
        if self.config['system']['keep_checkpoints'] > 0:
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if len(checkpoints) > self.config['system']['keep_checkpoints']:
                for old_checkpoint in checkpoints[:-self.config['system']['keep_checkpoints']]:
                    try:
                        os.remove(old_checkpoint)
                    except:
                        pass
    
    def train(self):
        """Main training loop"""
        start_time = time.time()
        
        self.logger.info("Starting comprehensive DCCRN training...")
        self.logger.info(f"Training configuration:")
        self.logger.info(f"   - Total training pairs: {self.training_pairs:,}")
        self.logger.info(f"   - Batch size: {self.config['training']['batch_size']}")
        self.logger.info(f"   - Epochs: {self.config['training']['num_epochs']}")
        self.logger.info(f"   - Learning rate: {self.config['training']['learning_rate']}")
        
        # Prepare dataset
        train_loader, val_loader = self.prepare_comprehensive_dataset()
        
        # Training loop
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            # Check if best model
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1
            
            # Log results
            self.logger.info(
                f'📈 Epoch {epoch}/{num_epochs} Summary:'
            )
            self.logger.info(f'   • Train Loss: {train_loss:.6f}')
            self.logger.info(f'   • Val Loss: {val_loss:.6f}')
            self.logger.info(f'   • Best Loss: {self.best_loss:.6f}')
            self.logger.info(f'   • Learning Rate: {self.optimizer.param_groups[0]["lr"]:.8f}')
            self.logger.info(f'   • Epoch Time: {epoch_time/60:.2f} minutes')
            self.logger.info(f'   • Total Time: {(time.time() - start_time)/3600:.2f} hours')
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if (self.config['validation']['early_stopping']['enable'] and 
                self.steps_since_improvement >= self.config['validation']['early_stopping']['patience']):
                self.logger.info(f"⏹️  Early stopping triggered after {epoch} epochs")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f'🎉 Training completed!')
        self.logger.info(f'   • Total time: {total_time/3600:.2f} hours')
        self.logger.info(f'   • Best validation loss: {self.best_loss:.6f}')
        self.logger.info(f'   • Model saved in: {self.checkpoint_dir}')
        
        return self.best_loss

def main():
    """Main function"""
    config_path = os.path.join(os.path.dirname(__file__), 'config_comprehensive.yaml')
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return
    
    print("Starting Comprehensive DCCRN Training")
    print("=" * 60)
    print("Using ALL 23,075+ audio files for maximum performance")
    print("Optimized for Intel i3-7020U with 8GB RAM")
    print("=" * 60)
    
    try:
        trainer = ComprehensiveTrainer(config_path)
        best_loss = trainer.train()
        
        print(f"\nTRAINING COMPLETED SUCCESSFULLY!")
        print(f"Best validation loss: {best_loss:.6f}")
        print(f"Model saved in: {trainer.checkpoint_dir}")
        print(f"Ready for inference!")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

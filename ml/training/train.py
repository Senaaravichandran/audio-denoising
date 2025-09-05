import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml
from datetime import datetime
import warnings

# Import our modules
import sys
sys.path.append('..')
from models.dccrn import DCCRN
from utils.audio_utils import AudioProcessor
from utils.dataset import create_dataloaders
from utils.losses import CombinedLoss

warnings.filterwarnings("ignore")


class DCCRNTrainer:
    """Trainer class for DCCRN model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cpu')
        print(f"Using device: {self.device} (CPU optimized)")
        
        # Create directories
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['training']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_model()
        self._init_data()
        self._init_optimizer()
        self._init_loss()
        self._init_logging()
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float('inf')
        
    def _init_model(self):
        """Initialize DCCRN model"""
        model_config = self.config['model']
        self.model = DCCRN(**model_config)
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model initialized:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Device: CPU (optimized)")
    
    def _init_data(self):
        """Initialize data loaders"""
        data_config = self.config['data']
        
        # Create audio processor
        self.processor = AudioProcessor(**data_config['processor'])
        
        # Create data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            clean_train_dir=data_config['clean_train_dir'],
            noisy_train_dir=data_config['noisy_train_dir'],
            clean_val_dir=data_config['clean_val_dir'],
            noisy_val_dir=data_config['noisy_val_dir'],
            processor=self.processor,
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            segment_length=data_config.get('segment_length'),
            **data_config.get('dataset_kwargs', {})
        )
        
        print(f"Data loaders initialized:")
        print(f"  Training batches: {len(self.train_loader)}")
        print(f"  Validation batches: {len(self.val_loader)}")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        opt_config = self.config['optimizer']
        
        # Optimizer
        if opt_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                weight_decay=opt_config.get('weight_decay', 1e-6)
            )
        elif opt_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                weight_decay=opt_config.get('weight_decay', 1e-2)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
        
        # Scheduler
        if 'scheduler' in opt_config:
            sched_config = opt_config['scheduler']
            if sched_config['type'] == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=sched_config.get('T_max', self.config['training']['epochs']),
                    eta_min=sched_config.get('eta_min', 1e-6)
                )
            elif sched_config['type'] == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=sched_config.get('step_size', 30),
                    gamma=sched_config.get('gamma', 0.5)
                )
            elif sched_config['type'] == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=sched_config.get('factor', 0.5),
                    patience=sched_config.get('patience', 10),
                    min_lr=sched_config.get('min_lr', 1e-6)
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
    
    def _init_loss(self):
        """Initialize loss function"""
        loss_config = self.config['loss']
        self.criterion = CombinedLoss(**loss_config)
    
    def _init_logging(self):
        """Initialize logging"""
        self.writer = SummaryWriter(self.log_dir)
        
        # Save config
        config_path = self.log_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, (noisy_spec, clean_spec) in enumerate(pbar):
            # Move to device
            noisy_spec = noisy_spec.to(self.device)
            clean_spec = clean_spec.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Model prediction
            pred_spec = self.model(noisy_spec)
            
            # Compute loss
            loss, loss_dict = self.criterion(pred_spec, clean_spec)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if 'grad_clip' in self.config['training']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to tensorboard
            if self.step % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.step)
                
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, self.step)
            
            self.step += 1
        
        # Average losses
        epoch_loss /= len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_loss, epoch_losses
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        val_loss = 0.0
        val_losses = {}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            
            for noisy_spec, clean_spec in pbar:
                # Move to device
                noisy_spec = noisy_spec.to(self.device)
                clean_spec = clean_spec.to(self.device)
                
                # Forward pass
                pred_spec = self.model(noisy_spec)
                
                # Compute loss
                loss, loss_dict = self.criterion(pred_spec, clean_spec)
                
                val_loss += loss.item()
                for key, value in loss_dict.items():
                    if key not in val_losses:
                        val_losses[key] = 0.0
                    val_losses[key] += value
                
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        # Average losses
        val_loss /= len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)
        
        return val_loss, val_losses
    
    def save_checkpoint(self, is_best=False, filename=None):
        """Save model checkpoint"""
        if filename is None:
            filename = f'checkpoint_epoch_{self.epoch}.pt'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Get model state dict (handle DataParallel)
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
            # Also save just the model for inference
            model_path = self.checkpoint_dir / 'dccrn_model.pt'
            torch.save(model_state_dict, model_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, step {self.step}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Total epochs: {self.config['training']['epochs']}")
        
        for epoch in range(self.epoch, self.config['training']['epochs']):
            self.epoch = epoch
            
            # Train
            train_loss, train_losses = self.train_epoch()
            
            # Validate
            val_loss, val_losses = self.validate()
            
            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            
            for key, value in train_losses.items():
                self.writer.add_scalar(f'epoch/train_{key}', value, epoch)
            
            for key, value in val_losses.items():
                self.writer.add_scalar(f'epoch/val_{key}', value, epoch)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(is_best=is_best)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Best Val Loss: {self.best_val_loss:.6f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Save final checkpoint
        self.save_checkpoint(is_best=False, filename='final_model.pt')
        
        print("Training completed!")
        self.writer.close()


def load_config(config_path):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train DCCRN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create trainer
    trainer = DCCRNTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

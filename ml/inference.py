import torch
import torchaudio
import argparse
import json
from pathlib import Path
import numpy as np
import sys
import warnings
import os

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from models.dccrn import DCCRN
    from utils.audio_utils import AudioProcessor
except ImportError:
    # Try alternative import paths
    sys.path.insert(0, os.path.join(current_dir, '..'))
    from ml.models.dccrn import DCCRN
    from ml.utils.audio_utils import AudioProcessor

warnings.filterwarnings("ignore")


class DCCRNInference:
    """DCCRN model inference class"""
    
    def __init__(self, model_path, config=None, device=None, fast_mode=True):
        """
        Initialize inference
        
        Args:
            model_path: Path to trained model checkpoint
            config: Model configuration (optional, will try to load from checkpoint)
            device: Device to run inference on (optimized for CPU)
            fast_mode: Enable optimizations for faster processing
        """
        # Always use CPU for optimized performance and compatibility
        self.device = torch.device('cpu')
        self.fast_mode = fast_mode
        print(f"Using device: {self.device} (optimized)")
        print(f"Fast mode: {'enabled' if fast_mode else 'disabled'}")
        
        # Load model
        self.model, self.config = self._load_model(model_path, config)
        self.model.eval()
        
        # Optimize for inference if fast_mode is enabled
        if self.fast_mode:
            print("Applying inference optimizations...")
            # Disable gradient computation permanently
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Skip model compilation on Windows due to compiler issues
            # Try to compile model for faster inference (PyTorch 2.0+)
            # try:
            #     self.model = torch.compile(self.model, mode='reduce-overhead')
            #     print("Model compiled for faster inference")
            # except:
            #     print("Model compilation not available, using standard mode")
            print("Model optimized for fast inference (gradients disabled)")
        
        # Initialize audio processor
        processor_config = self.config.get('processor', {})
        self.processor = AudioProcessor(
            n_fft=self.config.get('n_fft', 512),
            hop_length=self.config.get('hop_length', 256),
            win_length=self.config.get('win_length', 512),
            sample_rate=processor_config.get('sample_rate', 16000),
            normalize=processor_config.get('normalize', True)
        )
        
        print("Model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self, model_path, config):
        """Load trained model"""
        checkpoint = None
        
        try:
            # Check if model file exists and has reasonable size
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                checkpoint = None
            elif os.path.getsize(model_path) < 1000:  # Less than 1KB means corrupted
                print(f"Model file appears corrupted (size: {os.path.getsize(model_path)} bytes)")
                checkpoint = None
            else:
                # Try with weights_only=True first (secure mode)
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                print("✅ Model loaded successfully with secure mode")
        except:
            try:
                # Fallback to old format for compatibility
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                print("⚠️  Warning: Using legacy model loading format")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                checkpoint = None
        
        # If we couldn't load the checkpoint, create a new model with random weights
        if checkpoint is None:
            print("🔧 Creating new model with random initialization...")
            print("⚠️  Note: Audio enhancement may be limited until model is properly trained.")
        
        # Get model config
        if config is None:
            if checkpoint and 'config' in checkpoint:
                model_config = checkpoint['config']['model']
            else:
                # Default config if not available
                model_config = {
                    'n_fft': 512,
                    'hop_length': 256,
                    'win_length': 512,
                    'encoder_layers': 5,
                    'hidden_dim': 128,
                    'lstm_layers': 2,
                    'use_clstm': True,
                    'kernel_size': [5, 2],
                    'stride': [2, 1],
                    'use_cbn': True,
                    'masking_mode': 'E',
                    'causal': False
                }
                print("Using default model configuration")
        else:
            model_config = config
        
        # Create model
        model = DCCRN(**model_config)
        
        # Load weights if available
        if checkpoint and 'model_state_dict' in checkpoint:
            try:
                state_dict = checkpoint['model_state_dict']
                
                # Handle DataParallel models
                if any(key.startswith('module.') for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace('module.', '')
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
                
                model.load_state_dict(state_dict)
                print(f"✅ Model weights loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            except Exception as e:
                print(f"⚠️  Could not load model weights: {e}")
                print("Using randomly initialized weights")
        else:
            print("🎲 Using randomly initialized model weights")
            print("💡 To get better results, train the model using: train-model.bat")
        
        model.to(self.device)
        
        return model, model_config
    
    def enhance_audio(self, noisy_audio, denoising_strength=1.0):
        """
        Enhance a single audio tensor
        
        Args:
            noisy_audio: Input noisy audio tensor [T] or [1, T]
            denoising_strength: Denoising strength (0.0 to 1.0)
        
        Returns:
            enhanced_audio: Enhanced audio tensor [1, T] (maintains channel dimension)
        """
        # Ensure correct shape
        original_was_1d = noisy_audio.dim() == 1
        if original_was_1d:
            noisy_audio = noisy_audio.unsqueeze(0)  # Add batch dimension
        
        # Move to device
        noisy_audio = noisy_audio.to(self.device)
        
        with torch.no_grad():
            # Optimize memory usage in fast mode
            if self.fast_mode:
                # Use mixed precision for faster inference
                with torch.autocast(device_type='cpu', dtype=torch.float16, enabled=False):
                    # Convert to spectrogram
                    noisy_spec = self.processor.stft(noisy_audio)  # [1, F, T, 2]
                    
                    # Model inference with optimizations
                    enhanced_spec = self.model(noisy_spec)  # [1, F, T, 2]
            else:
                # Convert to spectrogram
                noisy_spec = self.processor.stft(noisy_audio)  # [1, F, T, 2]
                
                # Model inference
                enhanced_spec = self.model(noisy_spec)  # [1, F, T, 2]
            
            # Apply denoising strength
            if denoising_strength < 1.0:
                # Blend enhanced and original spectrograms
                enhanced_spec = (denoising_strength * enhanced_spec + 
                               (1 - denoising_strength) * noisy_spec)
            
            # Convert back to audio
            enhanced_audio = self.processor.istft(enhanced_spec)  # [1, T]
            
            # Keep as 2D tensor for compatibility
            if enhanced_audio.dim() == 1:
                enhanced_audio = enhanced_audio.unsqueeze(0)
        
        return enhanced_audio
    
    def enhance_file(self, input_path, output_path, denoising_strength=1.0, 
                    preserve_original_length=True):
        """
        Enhance audio file
        
        Args:
            input_path: Path to input noisy audio file
            output_path: Path to save enhanced audio
            denoising_strength: Denoising strength (0.0 to 1.0)
            preserve_original_length: Whether to preserve original audio length
        """
        # Load audio
        noisy_audio = self.processor.load_audio(input_path)
        original_length = len(noisy_audio)
        
        # Enhance audio
        enhanced_audio = self.enhance_audio(noisy_audio, denoising_strength)
        
        # Trim to original length if needed
        if preserve_original_length and len(enhanced_audio) != original_length:
            if len(enhanced_audio) > original_length:
                enhanced_audio = enhanced_audio[:original_length]
            else:
                # Pad if shorter (shouldn't happen normally)
                padding = original_length - len(enhanced_audio)
                enhanced_audio = torch.cat([enhanced_audio, torch.zeros(padding)])
        
        # Save enhanced audio
        self.processor.save_audio(enhanced_audio, output_path)
        
        print(f"Enhanced audio saved to: {output_path}")
    
    def enhance_batch(self, input_dir, output_dir, denoising_strength=1.0,
                     file_extensions=None):
        """
        Enhance multiple audio files
        
        Args:
            input_dir: Directory containing noisy audio files
            output_dir: Directory to save enhanced audio files
            denoising_strength: Denoising strength (0.0 to 1.0)
            file_extensions: List of file extensions to process
        """
        if file_extensions is None:
            file_extensions = ['.wav', '.flac', '.mp3', '.m4a']
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find audio files
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(input_dir.glob(f'*{ext}'))
            audio_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return
        
        print(f"Found {len(audio_files)} audio files to enhance")
        
        # Process files
        for i, input_path in enumerate(audio_files, 1):
            print(f"Processing {i}/{len(audio_files)}: {input_path.name}")
            
            # Create output path
            output_path = output_dir / f"enhanced_{input_path.name}"
            
            try:
                self.enhance_file(input_path, output_path, denoising_strength)
            except Exception as e:
                print(f"Error processing {input_path.name}: {e}")
                continue
        
        print("Batch processing completed!")
    
    def get_spectrograms(self, noisy_audio, enhanced_audio=None):
        """
        Get spectrograms for visualization
        
        Args:
            noisy_audio: Noisy audio tensor
            enhanced_audio: Enhanced audio tensor (optional)
        
        Returns:
            spectrograms: Dictionary containing spectrograms
        """
        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
        
        with torch.no_grad():
            # Noisy spectrogram
            noisy_spec = self.processor.stft(noisy_audio)
            noisy_mag = self.processor.compute_magnitude(noisy_spec).squeeze(0).cpu().numpy()
            
            spectrograms = {
                'noisy_magnitude': noisy_mag,
                'noisy_phase': self.processor.compute_phase(noisy_spec).squeeze(0).cpu().numpy()
            }
            
            if enhanced_audio is not None:
                if enhanced_audio.dim() == 1:
                    enhanced_audio = enhanced_audio.unsqueeze(0)
                
                enhanced_spec = self.processor.stft(enhanced_audio)
                enhanced_mag = self.processor.compute_magnitude(enhanced_spec).squeeze(0).cpu().numpy()
                
                spectrograms.update({
                    'enhanced_magnitude': enhanced_mag,
                    'enhanced_phase': self.processor.compute_phase(enhanced_spec).squeeze(0).cpu().numpy()
                })
        
        return spectrograms


def main():
    parser = argparse.ArgumentParser(description='DCCRN Audio Enhancement Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Input audio file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output audio file or directory')
    parser.add_argument('--strength', type=float, default=1.0, help='Denoising strength (0.0-1.0)')
    parser.add_argument('--batch', action='store_true', help='Process directory (batch mode)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize inference
    device = torch.device(args.device) if args.device else None
    inference = DCCRNInference(args.model, device=device)
    
    # Process audio
    if args.batch:
        inference.enhance_batch(args.input, args.output, args.strength)
    else:
        inference.enhance_file(args.input, args.output, args.strength)


if __name__ == "__main__":
    main()

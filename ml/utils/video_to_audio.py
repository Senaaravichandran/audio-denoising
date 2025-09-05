#!/usr/bin/env python3
"""
Video to Audio Converter using moviepy
Alternative to FFmpeg for audio extraction
"""

import sys
import os
from pathlib import Path

def convert_video_to_audio(video_path: str, audio_path: str, format: str = 'wav') -> bool:
    """
    Convert video to audio using moviepy
    
    Args:
        video_path: Path to input video file
        audio_path: Path to output audio file
        format: Output audio format ('wav', 'mp3', etc.)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import moviepy (install if needed)
        try:
            from moviepy import VideoFileClip
        except ImportError:
            print("Installing moviepy...")
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'moviepy'])
            from moviepy import VideoFileClip
        
        print(f"Converting {video_path} to {audio_path}")
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Extract audio
        audio = video.audio
        
        # Write audio file
        audio.write_audiofile(
            audio_path,
            codec='pcm_s16le' if format == 'wav' else 'mp3',
            logger=None
        )
        
        # Cleanup
        audio.close()
        video.close()
        
        print(f"Audio extraction successful: {audio_path}")
        return True
        
    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return False

def main():
    """Main function for command line usage"""
    if len(sys.argv) != 3:
        print("Usage: python video_to_audio.py <video_path> <audio_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    
    # Determine format from extension
    format = Path(audio_path).suffix.lower().lstrip('.')
    
    success = convert_video_to_audio(video_path, audio_path, format)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

"""
Video Processing Service for SonicPurge
Handles video URL extraction and audio extraction from video files
"""

import os
import sys
import json
import argparse
import tempfile
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import re

class VideoProcessingService:
    """Service for extracting audio from video URLs and files"""
    
    def __init__(self):
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']
        self.output_formats = ['wav', 'mp3', 'flac']
    
    def extract_audio_from_url(self, video_url: str, output_path: str, output_format: str = 'wav') -> dict:
        """
        Extract audio from video URL using yt-dlp (if available) or direct download
        
        Args:
            video_url: URL of the video
            output_path: Path to save extracted audio
            output_format: Output format (wav, mp3, flac)
            
        Returns:
            Dict with success status and metadata
        """
        try:
            print(f"[PROCESSING] Extracting audio from video URL...")
            print(f"   URL: {video_url}")
            print(f"   Output: {output_path}")
            print(f"   Format: {output_format}")
            
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Try yt-dlp first (best option for video URLs)
            if self._try_ytdlp_extraction(video_url, output_path, output_format):
                return self._get_success_result(output_path, "yt-dlp")
            
            # Try youtube-dl as fallback
            if self._try_youtube_dl_extraction(video_url, output_path, output_format):
                return self._get_success_result(output_path, "youtube-dl")
            
            # Try ffmpeg direct download (for direct video file URLs)
            if self._try_ffmpeg_extraction(video_url, output_path, output_format):
                return self._get_success_result(output_path, "ffmpeg")
            
            # If all methods fail
            return {
                'success': False,
                'error': 'Failed to extract audio. Please check if the URL is valid and publicly accessible.'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Video processing failed: {str(e)}'
            }
    
    def extract_audio_from_file(self, video_path: str, output_path: str, output_format: str = 'wav') -> dict:
        """
        Extract audio from local video file using ffmpeg
        
        Args:
            video_path: Path to video file
            output_path: Path to save extracted audio
            output_format: Output format (wav, mp3, flac)
            
        Returns:
            Dict with success status and metadata
        """
        try:
            print(f"[PROCESSING] Extracting audio from video file...")
            print(f"   Input: {video_path}")
            print(f"   Output: {output_path}")
            print(f"   Format: {output_format}")
            
            if not os.path.exists(video_path):
                return {
                    'success': False,
                    'error': 'Video file not found'
                }
            
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Use ffmpeg to extract audio
            if self._extract_with_ffmpeg(video_path, output_path, output_format):
                return self._get_success_result(output_path, "ffmpeg")
            else:
                return {
                    'success': False,
                    'error': 'Failed to extract audio from video file'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Video file processing failed: {str(e)}'
            }
    
    def _try_ytdlp_extraction(self, url: str, output_path: str, format: str) -> bool:
        """Try extracting with yt-dlp"""
        try:
            print("   Trying yt-dlp extraction...")
            
            # Try both direct yt-dlp command and python module
            commands_to_try = [
                ['yt-dlp', '--version'],
                ['python', '-m', 'yt_dlp', '--version']
            ]
            
            yt_dlp_cmd = None
            for cmd in commands_to_try:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        yt_dlp_cmd = cmd[:-1]  # Remove --version
                        print(f"   Found yt-dlp: {' '.join(cmd[:-1])}")
                        break
                except:
                    continue
            
            if not yt_dlp_cmd:
                print("   yt-dlp not available")
                return False
            
            # Use yt-dlp to extract audio
            cmd = yt_dlp_cmd + [
                '--extract-audio',
                '--audio-format', format,
                '--audio-quality', '0',  # Best quality
                '--output', output_path.replace(f'.{format}', '.%(ext)s'),
                url
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print("   ✅ yt-dlp extraction successful")
                # yt-dlp might have changed the filename, find the actual output
                self._find_and_rename_output(output_path, format)
                return True
            else:
                print(f"   ❌ yt-dlp failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("   ❌ yt-dlp timeout")
            return False
        except FileNotFoundError:
            print("   yt-dlp not found")
            return False
        except Exception as e:
            print(f"   yt-dlp error: {e}")
            return False
    
    def _try_youtube_dl_extraction(self, url: str, output_path: str, format: str) -> bool:
        """Try extracting with youtube-dl"""
        try:
            print("   Trying youtube-dl extraction...")
            
            # Try both direct youtube-dl command and python module
            commands_to_try = [
                ['youtube-dl', '--version'],
                ['python', '-m', 'youtube_dl', '--version']
            ]
            
            youtube_dl_cmd = None
            for cmd in commands_to_try:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        youtube_dl_cmd = cmd[:-1]  # Remove --version
                        print(f"   Found youtube-dl: {' '.join(cmd[:-1])}")
                        break
                except:
                    continue
            
            if not youtube_dl_cmd:
                print("   youtube-dl not available")
                return False
            
            # Use youtube-dl to extract audio
            cmd = youtube_dl_cmd + [
                '--extract-audio',
                '--audio-format', format,
                '--audio-quality', '0',  # Best quality
                '--output', output_path.replace(f'.{format}', '.%(ext)s'),
                url
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print("   ✅ youtube-dl extraction successful")
                self._find_and_rename_output(output_path, format)
                return True
            else:
                print(f"   ❌ youtube-dl failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("   ❌ youtube-dl timeout")
            return False
        except FileNotFoundError:
            print("   youtube-dl not found")
            return False
        except Exception as e:
            print(f"   youtube-dl error: {e}")
            return False
    
    def _try_ffmpeg_extraction(self, url: str, output_path: str, format: str) -> bool:
        """Try direct ffmpeg extraction (for direct video URLs)"""
        try:
            print("   Trying ffmpeg direct extraction...")
            
            # Check if it looks like a direct video URL
            if not self._is_direct_video_url(url):
                print("   Not a direct video URL")
                return False
            
            return self._extract_with_ffmpeg(url, output_path, format)
            
        except Exception as e:
            print(f"   ffmpeg direct extraction error: {e}")
            return False
    
    def _extract_with_ffmpeg(self, input_path: str, output_path: str, format: str) -> bool:
        """Extract audio using ffmpeg"""
        try:
            print("   Using ffmpeg for audio extraction...")
            
            # Check if ffmpeg is available
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("   ffmpeg not available")
                return False
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vn',  # No video
                '-acodec', self._get_audio_codec(format),
                '-ar', '16000',  # 16kHz sample rate for DCCRN
                '-ac', '1',  # Mono for DCCRN compatibility
                '-y',  # Overwrite output
                output_path
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0 and os.path.exists(output_path):
                print("   ✅ ffmpeg extraction successful")
                return True
            else:
                print(f"   ❌ ffmpeg failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("   ❌ ffmpeg timeout")
            return False
        except FileNotFoundError:
            print("   ffmpeg not found")
            return False
        except Exception as e:
            print(f"   ffmpeg error: {e}")
            return False
    
    def _get_audio_codec(self, format: str) -> str:
        """Get appropriate audio codec for format"""
        codec_map = {
            'wav': 'pcm_s16le',
            'mp3': 'libmp3lame',
            'flac': 'flac'
        }
        return codec_map.get(format, 'pcm_s16le')
    
    def _is_direct_video_url(self, url: str) -> bool:
        """Check if URL points directly to a video file"""
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            return any(path.endswith(ext) for ext in self.supported_video_formats)
        except:
            return False
    
    def _find_and_rename_output(self, expected_path: str, format: str):
        """Find the actual output file and rename it to expected path"""
        try:
            output_dir = Path(expected_path).parent
            filename_base = Path(expected_path).stem
            
            # Look for files with similar names
            for file in output_dir.glob(f"{filename_base}*"):
                if file.suffix.lower() in [f'.{format}', '.m4a', '.webm']:
                    if file.name != Path(expected_path).name:
                        file.rename(expected_path)
                        print(f"   Renamed {file.name} to {Path(expected_path).name}")
                    break
        except Exception as e:
            print(f"   Warning: Could not rename output file: {e}")
    
    def _get_success_result(self, output_path: str, method: str) -> dict:
        """Get success result with metadata"""
        try:
            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            return {
                'success': True,
                'output_path': output_path,
                'file_size': file_size,
                'extraction_method': method,
                'message': f'Audio extracted successfully using {method}'
            }
        except Exception as e:
            return {
                'success': True,
                'output_path': output_path,
                'message': f'Audio extracted using {method}',
                'warning': f'Could not get file metadata: {e}'
            }

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Video Processing Service for SonicPurge')
    parser.add_argument('--url', '-u', help='Video URL to process')
    parser.add_argument('--file', '-f', help='Video file to process')
    parser.add_argument('--output', '-o', required=True, help='Output audio file path')
    parser.add_argument('--format', default='wav', choices=['wav', 'mp3', 'flac'], help='Output format')
    parser.add_argument('--json', action='store_true', help='Output result as JSON')
    
    args = parser.parse_args()
    
    if not args.url and not args.file:
        print("❌ Either --url or --file must be provided")
        sys.exit(1)
    
    service = VideoProcessingService()
    
    if args.url:
        result = service.extract_audio_from_url(args.url, args.output, args.format)
    else:
        result = service.extract_audio_from_file(args.file, args.output, args.format)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result['success']:
            print(f"\n[SUCCESS] {result['message']}")
            print(f"   Output: {result.get('output_path', args.output)}")
        else:
            print(f"\n[ERROR] {result['error']}")
            sys.exit(1)

if __name__ == "__main__":
    main()

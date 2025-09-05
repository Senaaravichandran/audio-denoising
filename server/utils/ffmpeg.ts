import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';

export interface AudioMetadata {
  duration: number;
  format: string;
  bitrate: number;
  sampleRate: number;
  channels: number;
  codec: string;
}

export async function getAudioMetadata(filePath: string): Promise<AudioMetadata> {
  return new Promise((resolve, reject) => {
    const ffprobe = spawn('ffprobe', [
      '-v', 'quiet',
      '-print_format', 'json',
      '-show_format',
      '-show_streams',
      filePath
    ]);

    let output = '';
    ffprobe.stdout.on('data', (data) => {
      output += data.toString();
    });

    ffprobe.on('close', (code) => {
      if (code === 0) {
        try {
          const metadata = JSON.parse(output);
          const audioStream = metadata.streams.find((stream: any) => stream.codec_type === 'audio');
          
          if (!audioStream) {
            reject(new Error('No audio stream found'));
            return;
          }

          resolve({
            duration: parseFloat(metadata.format.duration || '0'),
            format: metadata.format.format_name,
            bitrate: parseInt(metadata.format.bit_rate || '0'),
            sampleRate: parseInt(audioStream.sample_rate || '0'),
            channels: audioStream.channels || 0,
            codec: audioStream.codec_name || 'unknown'
          });
        } catch (error) {
          reject(new Error('Failed to parse metadata'));
        }
      } else {
        reject(new Error(`ffprobe failed with code ${code}`));
      }
    });

    ffprobe.on('error', reject);
  });
}

export async function convertAudioFormat(inputPath: string, outputFormat: string): Promise<string> {
  const outputPath = inputPath.replace(/\.[^/.]+$/, `.${outputFormat}`);
  
  return new Promise((resolve, reject) => {
    const args = ['-i', inputPath];
    
    // Add format-specific arguments
    switch (outputFormat.toLowerCase()) {
      case 'mp3':
        args.push('-codec:a', 'libmp3lame', '-b:a', '320k');
        break;
      case 'flac':
        args.push('-codec:a', 'flac');
        break;
      case 'aac':
        args.push('-codec:a', 'aac', '-b:a', '256k');
        break;
      case 'ogg':
        args.push('-codec:a', 'libvorbis', '-q:a', '5');
        break;
      case 'wav':
      default:
        args.push('-codec:a', 'pcm_s16le');
        break;
    }
    
    args.push('-y', outputPath);
    
    const ffmpeg = spawn('ffmpeg', args);
    
    ffmpeg.on('close', (code) => {
      if (code === 0) {
        resolve(outputPath);
      } else {
        reject(new Error(`FFmpeg conversion failed with code ${code}`));
      }
    });
    
    ffmpeg.on('error', reject);
  });
}

export async function extractAudioFromVideo(videoPath: string): Promise<string> {
  const outputPath = videoPath.replace(/\.[^/.]+$/, '_extracted.wav');
  
  console.log(`Starting FFmpeg audio extraction:`);
  console.log(`  Input: ${videoPath}`);
  console.log(`  Output: ${outputPath}`);
  
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn('ffmpeg', [
      '-i', videoPath,
      '-vn', // No video
      '-acodec', 'pcm_s16le', // Use uncompressed audio
      '-ar', '44100', // Sample rate
      '-ac', '2', // Stereo
      '-y', // Overwrite output file
      outputPath
    ]);
    
    let stderr = '';
    
    ffmpeg.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    ffmpeg.on('close', (code) => {
      console.log(`FFmpeg process finished with code: ${code}`);
      if (stderr) {
        console.log(`FFmpeg stderr output:`, stderr);
      }
      
      if (code === 0) {
        console.log(`Audio extraction successful: ${outputPath}`);
        resolve(outputPath);
      } else {
        console.error(`Video audio extraction failed with code ${code}`);
        console.error(`FFmpeg stderr:`, stderr);
        reject(new Error(`Video audio extraction failed with code ${code}. FFmpeg error: ${stderr}`));
      }
    });
    
    ffmpeg.on('error', (error) => {
      console.error(`FFmpeg spawn error:`, error);
      reject(error);
    });
  });
}

export async function normalizeAudio(inputPath: string): Promise<string> {
  const outputPath = inputPath.replace(/\.[^/.]+$/, '_normalized.wav');
  
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn('ffmpeg', [
      '-i', inputPath,
      '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
      '-y',
      outputPath
    ]);
    
    ffmpeg.on('close', (code) => {
      if (code === 0) {
        resolve(outputPath);
      } else {
        reject(new Error(`Audio normalization failed with code ${code}`));
      }
    });
    
    ffmpeg.on('error', reject);
  });
}

export async function splitAudioChannels(inputPath: string): Promise<{ left: string; right: string }> {
  const leftPath = inputPath.replace(/\.[^/.]+$/, '_left.wav');
  const rightPath = inputPath.replace(/\.[^/.]+$/, '_right.wav');
  
  return new Promise((resolve, reject) => {
    // Extract left channel
    const leftProcess = spawn('ffmpeg', [
      '-i', inputPath,
      '-af', 'pan=mono|c0=c0',
      '-y', leftPath
    ]);
    
    leftProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Left channel extraction failed with code ${code}`));
        return;
      }
      
      // Extract right channel
      const rightProcess = spawn('ffmpeg', [
        '-i', inputPath,
        '-af', 'pan=mono|c0=c1',
        '-y', rightPath
      ]);
      
      rightProcess.on('close', (rightCode) => {
        if (rightCode === 0) {
          resolve({ left: leftPath, right: rightPath });
        } else {
          reject(new Error(`Right channel extraction failed with code ${rightCode}`));
        }
      });
      
      rightProcess.on('error', reject);
    });
    
    leftProcess.on('error', reject);
  });
}

export async function combineAudioWithVideo(videoPath: string, audioPath: string, outputFormat: string = 'mp4'): Promise<string> {
  // Always output as MP4 for video files
  const outputPath = path.join('outputs', `enhanced_video_${Date.now()}.mp4`);
  
  console.log(`🎬 Starting video-audio combination:`);
  console.log(`  📹 Video: ${videoPath}`);
  console.log(`  🔊 Audio: ${audioPath}`);
  console.log(`  📤 Output: ${outputPath}`);
  
  return new Promise((resolve, reject) => {
    // Use a more compatible FFmpeg command
    const ffmpeg = spawn('ffmpeg', [
      '-i', videoPath,          // Input video file
      '-i', audioPath,          // Input enhanced audio file
      '-c:v', 'copy',           // Copy video codec (no re-encoding)
      '-c:a', 'aac',            // Use AAC audio codec
      '-b:a', '128k',           // Set audio bitrate to 128k (more compatible)
      '-ar', '44100',           // Set audio sample rate to 44.1kHz
      '-ac', '2',               // Force stereo output (duplicate mono to both channels)
      '-map', '0:v:0',          // Map video stream from first input
      '-map', '1:a:0',          // Map audio stream from second input
      '-shortest',              // Match duration to shortest stream
      '-avoid_negative_ts', 'make_zero', // Handle timestamp issues
      '-fflags', '+genpts',     // Generate presentation timestamps
      '-y',                     // Overwrite output file without asking
      outputPath
    ]);
    
    let stderr = '';
    let stdout = '';
    
    ffmpeg.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    ffmpeg.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    ffmpeg.on('close', (code) => {
      console.log(`🎬 FFmpeg video combination finished with code: ${code}`);
      
      if (code === 0) {
        console.log(`✅ Video combination successful: ${outputPath}`);
        resolve(outputPath);
      } else {
        console.error(`❌ Video combination failed with code ${code}`);
        console.error(`📋 FFmpeg stderr:`, stderr);
        console.error(`📋 FFmpeg stdout:`, stdout);
        reject(new Error(`Video combination failed. FFmpeg error code: ${code}`));
      }
    });
    
    ffmpeg.on('error', (error) => {
      console.error(`❌ FFmpeg spawn error:`, error);
      reject(new Error(`FFmpeg process failed: ${error.message}`));
    });
  });
}

export async function getSupportedFormats(): Promise<string[]> {
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn('ffmpeg', ['-formats']);
    
    let output = '';
    ffmpeg.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    ffmpeg.on('close', (code) => {
      if (code === 0) {
        const formats = output
          .split('\n')
          .filter(line => line.includes('E') && (line.includes('audio') || line.includes('A')))
          .map(line => line.split(/\s+/)[1])
          .filter(format => format && format !== 'E');
        
        resolve(formats);
      } else {
        reject(new Error(`Failed to get supported formats`));
      }
    });
    
    ffmpeg.on('error', reject);
  });
}

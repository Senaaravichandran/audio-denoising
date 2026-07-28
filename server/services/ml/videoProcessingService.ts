import { spawn } from 'child_process';
import * as path from 'path';
import { promises as fs } from 'fs';
import { dccrnService } from './dccrnService';

export interface VideoProcessingOptions {
  inputPath: string;
  outputPath: string;
  denoisingStrength?: number;
  preserveVideoQuality?: boolean;
}

export interface VideoProcessingResult {
  success: boolean;
  outputPath?: string;
  audioPath?: string;
  error?: string;
  metadata?: {
    duration?: number;
    resolution?: string;
    audioFormat?: string;
    videoFormat?: string;
  };
}

export class VideoProcessingService {
  private ffmpegPath: string;

  constructor() {
    // FFmpeg path - can be configured via environment variable
    this.ffmpegPath = process.env.FFMPEG_PATH || 'ffmpeg';
  }

  /**
   * Check if FFmpeg is available
   */
  async isFFmpegAvailable(): Promise<boolean> {
    return new Promise((resolve) => {
      const process = spawn(this.ffmpegPath, ['-version']);
      
      process.on('close', (code: number | null) => {
        resolve(code === 0);
      });
      
      process.on('error', () => {
        resolve(false);
      });
    });
  }

  /**
   * Extract audio from video file
   */
  async extractAudio(
    videoPath: string, 
    audioPath: string,
    format: string = 'wav'
  ): Promise<{ success: boolean; error?: string }> {
    return new Promise((resolve) => {
      const args = [
        '-i', videoPath,
        '-vn', // No video
        '-acodec', format === 'wav' ? 'pcm_s16le' : 'aac',
        '-ar', '16000', // Sample rate for DCCRN
        '-ac', '1', // Mono
        '-y', // Overwrite output
        audioPath
      ];

      console.log(`Extracting audio: ${this.ffmpegPath} ${args.join(' ')}`);

      const process = spawn(this.ffmpegPath, args);
      let stderr = '';

      process.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      process.on('close', (code: number | null) => {
        if (code === 0) {
          resolve({ success: true });
        } else {
          console.error('FFmpeg audio extraction failed:', stderr);
          resolve({
            success: false,
            error: stderr || `FFmpeg exited with code ${code}`
          });
        }
      });

      process.on('error', (error: Error) => {
        resolve({
          success: false,
          error: error.message
        });
      });
    });
  }

  /**
   * Replace audio in video file
   */
  async replaceAudio(
    videoPath: string,
    audioPath: string,
    outputPath: string,
    preserveVideoQuality: boolean = true
  ): Promise<{ success: boolean; error?: string }> {
    return new Promise((resolve) => {
      const args = [
        '-i', videoPath,
        '-i', audioPath,
        '-c:v', preserveVideoQuality ? 'copy' : 'libx264', // Copy video stream if preserving quality
        '-c:a', 'aac',
        '-map', '0:v:0', // Video from first input
        '-map', '1:a:0', // Audio from second input
        '-shortest', // Match shortest stream duration
        '-y', // Overwrite output
        outputPath
      ];

      if (!preserveVideoQuality) {
        args.push('-crf', '23'); // Reasonable quality for re-encoding
      }

      console.log(`Replacing audio: ${this.ffmpegPath} ${args.join(' ')}`);

      const process = spawn(this.ffmpegPath, args);
      let stderr = '';

      process.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      process.on('close', (code: number | null) => {
        if (code === 0) {
          resolve({ success: true });
        } else {
          console.error('FFmpeg audio replacement failed:', stderr);
          resolve({
            success: false,
            error: stderr || `FFmpeg exited with code ${code}`
          });
        }
      });

      process.on('error', (error: Error) => {
        resolve({
          success: false,
          error: error.message
        });
      });
    });
  }

  /**
   * Get video metadata
   */
  async getVideoMetadata(videoPath: string): Promise<any> {
    return new Promise((resolve) => {
      const args = [
        '-i', videoPath,
        '-f', 'null',
        '-'
      ];

      const process = spawn(this.ffmpegPath, args);
      let stderr = '';

      process.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      process.on('close', () => {
        // Parse FFmpeg output for metadata
        const metadata: any = {};
        
        // Extract duration
        const durationMatch = stderr.match(/Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})/);
        if (durationMatch) {
          const hours = parseInt(durationMatch[1]);
          const minutes = parseInt(durationMatch[2]);
          const seconds = parseFloat(durationMatch[3]);
          metadata.duration = hours * 3600 + minutes * 60 + seconds;
        }

        // Extract video resolution
        const resolutionMatch = stderr.match(/Video:.*?(\d{3,4}x\d{3,4})/);
        if (resolutionMatch) {
          metadata.resolution = resolutionMatch[1];
        }

        // Extract video format
        const videoFormatMatch = stderr.match(/Video: (\w+)/);
        if (videoFormatMatch) {
          metadata.videoFormat = videoFormatMatch[1];
        }

        // Extract audio format
        const audioFormatMatch = stderr.match(/Audio: (\w+)/);
        if (audioFormatMatch) {
          metadata.audioFormat = audioFormatMatch[1];
        }

        resolve(metadata);
      });

      process.on('error', () => {
        resolve({});
      });
    });
  }

  /**
   * Process video with audio enhancement
   */
  async processVideo(options: VideoProcessingOptions): Promise<VideoProcessingResult> {
    const {
      inputPath,
      outputPath,
      denoisingStrength = 1.0,
      preserveVideoQuality = true
    } = options;

    try {
      // Get video metadata
      const metadata = await this.getVideoMetadata(inputPath);

      // Generate temporary file paths
      const tempDir = path.join(process.cwd(), 'temp');
      await fs.mkdir(tempDir, { recursive: true });

      const originalAudioPath = path.join(tempDir, `audio_${Date.now()}_original.wav`);
      const enhancedAudioPath = path.join(tempDir, `audio_${Date.now()}_enhanced.wav`);

      // Step 1: Extract audio from video
      console.log('Extracting audio from video...');
      const extractResult = await this.extractAudio(inputPath, originalAudioPath);
      if (!extractResult.success) {
        return {
          success: false,
          error: `Audio extraction failed: ${extractResult.error}`
        };
      }

      // Step 2: Enhance audio using DCCRN
      console.log('Enhancing audio with DCCRN...');
      const enhanceResult = await dccrnService.enhanceAudio({
        inputPath: originalAudioPath,
        outputPath: enhancedAudioPath,
        denoisingStrength
      });

      if (!enhanceResult.success) {
        // Cleanup
        await this.cleanupTempFiles([originalAudioPath]);
        return {
          success: false,
          error: `Audio enhancement failed: ${enhanceResult.error}`
        };
      }

      // Step 3: Replace audio in video
      console.log('Replacing audio in video...');
      const replaceResult = await this.replaceAudio(
        inputPath,
        enhancedAudioPath,
        outputPath,
        preserveVideoQuality
      );

      if (!replaceResult.success) {
        // Cleanup
        await this.cleanupTempFiles([originalAudioPath, enhancedAudioPath]);
        return {
          success: false,
          error: `Audio replacement failed: ${replaceResult.error}`
        };
      }

      // Cleanup temporary files
      await this.cleanupTempFiles([originalAudioPath, enhancedAudioPath]);

      return {
        success: true,
        outputPath,
        metadata
      };

    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Convert video format
   */
  async convertVideo(
    inputPath: string,
    outputPath: string,
    options: {
      videoCodec?: string;
      audioCodec?: string;
      quality?: number;
      resolution?: string;
    } = {}
  ): Promise<{ success: boolean; error?: string }> {
    const {
      videoCodec = 'libx264',
      audioCodec = 'aac',
      quality = 23,
      resolution
    } = options;

    return new Promise((resolve) => {
      const args = [
        '-i', inputPath,
        '-c:v', videoCodec,
        '-c:a', audioCodec,
        '-crf', quality.toString(),
      ];

      if (resolution) {
        args.push('-s', resolution);
      }

      args.push('-y', outputPath);

      const process = spawn(this.ffmpegPath, args);
      let stderr = '';

      process.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      process.on('close', (code: number | null) => {
        if (code === 0) {
          resolve({ success: true });
        } else {
          resolve({
            success: false,
            error: stderr || `FFmpeg exited with code ${code}`
          });
        }
      });

      process.on('error', (error: Error) => {
        resolve({
          success: false,
          error: error.message
        });
      });
    });
  }

  /**
   * Clean up temporary files
   */
  private async cleanupTempFiles(filePaths: string[]): Promise<void> {
    for (const filePath of filePaths) {
      try {
        await fs.unlink(filePath);
      } catch (error) {
        console.warn(`Failed to cleanup temp file ${filePath}:`, error);
      }
    }
  }
}

// Singleton instance
export const videoProcessingService = new VideoProcessingService();

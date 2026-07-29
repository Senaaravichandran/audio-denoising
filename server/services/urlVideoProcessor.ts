import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { groqService } from './groqService';

export interface URLVideoProcessingResult {
  success: boolean;
  outputPath?: string;
  downloadedVideoPath?: string;
  extractedAudioPath?: string; // Add this to return the extracted audio path
  metadata?: {
    title: string;
    duration: number;
    platform: string;
    originalUrl: string;
    downloadType?: 'audio' | 'video';
  };
  error?: string;
  aiExplanation?: string;
}

export interface URLVideoProcessingOptions {
  denoisingStrength: number;
  processingMode: 'fast' | 'balanced';
  quality: 'best' | 'worst' | 'bestvideo+bestaudio';
  downloadType?: 'audio' | 'video';
}

export class URLVideoProcessor {
  private dccrnProcessor: any;
  private pythonPath: string;

  constructor(dccrnProcessor: any) {
    this.dccrnProcessor = dccrnProcessor;
    // Use the virtual environment Python executable
    this.pythonPath = process.env.NODE_ENV === 'production' ? 'python' : path.join(process.cwd(), '.venv', 'Scripts', 'python.exe');
  }

  /**
   * Detect platform from URL for better processing
   */
  private detectPlatform(url: string): string {
    const urlLower = url.toLowerCase();
    
    if (urlLower.includes('youtube.com') || urlLower.includes('youtu.be')) {
      return 'YouTube';
    } else if (urlLower.includes('tiktok.com')) {
      return 'TikTok';
    } else if (urlLower.includes('twitter.com') || urlLower.includes('x.com')) {
      return 'Twitter/X';
    } else if (urlLower.includes('instagram.com')) {
      return 'Instagram';
    } else if (urlLower.includes('facebook.com') || urlLower.includes('fb.watch')) {
      return 'Facebook';
    } else if (urlLower.includes('vimeo.com')) {
      return 'Vimeo';
    } else if (urlLower.includes('dailymotion.com')) {
      return 'Dailymotion';
    } else if (urlLower.includes('twitch.tv')) {
      return 'Twitch';
    } else if (urlLower.includes('soundcloud.com')) {
      return 'SoundCloud';
    } else if (urlLower.includes('reddit.com')) {
      return 'Reddit';
    } else if (urlLower.includes('linkedin.com')) {
      return 'LinkedIn';
    } else if (urlLower.includes('discord.com') || urlLower.includes('cdn.discordapp.com')) {
      return 'Discord';
    } else if (urlLower.includes('streamable.com')) {
      return 'Streamable';
    } else {
      return 'Unknown Platform';
    }
  }

  /**
   * Download video from URL using yt-dlp with enhanced error handling
   */
  async downloadVideoFromUrl(
    url: string,
    outputDir: string = 'uploads',
    progressCallback?: (progress: { progress: number; message: string; stage: string }) => void
  ): Promise<{ success: boolean; videoPath?: string; metadata?: any; error?: string }> {
    return new Promise((resolve) => {
      try {
        const timestamp = Date.now();
        const randomId = crypto.randomBytes(8).toString('hex');
        const outputTemplate = path.join(outputDir, `downloaded_${timestamp}_${randomId}.%(ext)s`);

        const platform = this.detectPlatform(url);
        console.log(`🌐 Starting video download from URL: ${url}`);
        console.log(`🎯 Detected platform: ${platform}`);
        console.log(`📁 Output template: ${outputTemplate}`);
        
        progressCallback?.({
          progress: 5,
          message: `Connecting to ${platform}...`,
          stage: 'download'
        });

        // Enhanced yt-dlp command with platform-specific optimizations
        const baseArgs = [
          '-m', 'yt_dlp',
          '--output', outputTemplate,
          '--print', 'after_move:filepath',
          '--print', 'title',
          '--print', 'duration',
          '--print', 'extractor',
          '--no-playlist',
          '--no-warnings',
          '--ignore-errors',
          '--retries', '5',
          '--fragment-retries', '5',
          '--geo-bypass',
          '--socket-timeout', '30',
          '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ];

        // Platform-specific format selection
        let formatArgs = [];
        switch (platform) {
          case 'YouTube':
            formatArgs = ['--format', 'best[height<=720][ext=mp4]/best[ext=mp4]/best'];
            break;
          case 'TikTok':
            formatArgs = ['--format', 'best[ext=mp4]/mp4/best'];
            break;
          case 'Twitter/X':
            formatArgs = ['--format', 'best[ext=mp4]/best'];
            break;
          case 'Instagram':
            formatArgs = ['--format', 'best[ext=mp4]/best'];
            break;
          case 'Facebook':
            formatArgs = ['--format', 'best[ext=mp4]/best'];
            break;
          default:
            formatArgs = ['--format', 'best[height<=720]/best[height<=480]/worst'];
        }

        const ytDlpArgs = [...baseArgs, ...formatArgs, url];

        console.log(`🔧 Running command: ${this.pythonPath} ${ytDlpArgs.join(' ')}`);

        const ytDlp = spawn(this.pythonPath, ytDlpArgs, {
          stdio: ['pipe', 'pipe', 'pipe'],
          env: { ...process.env, PYTHONUNBUFFERED: '1' }
        });

        let downloadedPath = '';
        let metadata = {
          title: 'Downloaded Video',
          duration: 0,
          platform: platform,
          originalUrl: url
        };

        let stdoutData = '';
        let stderrData = '';

        ytDlp.stdout.on('data', (data) => {
          const output = data.toString().trim();
          stdoutData += output + '\n';
          console.log(`📦 yt-dlp stdout: ${output}`);
          
          // Parse output for file path
          if (output.includes('.mp4') || output.includes('.webm') || output.includes('.mkv')) {
            downloadedPath = output.trim();
            console.log(`📹 Downloaded video path detected: ${downloadedPath}`);
          }
          
          // Parse metadata
          if (!metadata.title || metadata.title === 'Downloaded Video') {
            if (output.length > 0 && !output.includes('filepath') && !output.includes('.mp4')) {
              metadata.title = output.substring(0, 100); // Limit title length
            }
          }

          // Update progress
          progressCallback?.({
            progress: Math.min(25, 5 + Math.random() * 20),
            message: `Downloading: ${metadata.title}`,
            stage: 'download'
          });
        });

        ytDlp.stderr.on('data', (data) => {
          const error = data.toString();
          stderrData += error + '\n';
          console.log(`📥 yt-dlp stderr: ${error}`);
          
          // Parse download progress if available
          const progressMatch = error.match(/(\d+\.?\d*)%/);
          if (progressMatch) {
            const downloadProgress = parseFloat(progressMatch[1]);
            progressCallback?.({
              progress: Math.round(5 + (downloadProgress * 0.25)),
              message: `Downloading... ${downloadProgress.toFixed(1)}%`,
              stage: 'download'
            });
          }
        });

        // Set timeout for download
        const downloadTimeout = setTimeout(() => {
          console.log(`⏰ Download timeout reached for URL: ${url}`);
          ytDlp.kill('SIGTERM');
          resolve({
            success: false,
            error: 'Download timeout - URL may not be accessible or server is slow'
          });
        }, 120000); // 2 minutes timeout

        ytDlp.on('close', (code) => {
          clearTimeout(downloadTimeout);
          
          console.log(`🏁 yt-dlp process finished with code: ${code}`);
          console.log(`📤 Full stdout: ${stdoutData}`);
          console.log(`📤 Full stderr: ${stderrData}`);

          if (code === 0 && downloadedPath) {
            console.log(`✅ Video downloaded successfully: ${downloadedPath}`);
            progressCallback?.({
              progress: 30,
              message: 'Video download completed!',
              stage: 'download'
            });
            
            resolve({
              success: true,
              videoPath: downloadedPath,
              metadata
            });
          } else {
            console.error(`❌ yt-dlp failed with code: ${code}`);
            
            // Provide more specific error messages
            let errorMessage = `Download failed (exit code: ${code})`;
            
            if (stderrData.includes('HTTP Error 403') || stderrData.includes('Forbidden')) {
              errorMessage = 'Access denied - Video may be private or geo-restricted';
            } else if (stderrData.includes('HTTP Error 404') || stderrData.includes('Not Found')) {
              errorMessage = 'Video not found - Please check the URL';
            } else if (stderrData.includes('Unsupported URL')) {
              errorMessage = 'Unsupported URL - Platform not supported by yt-dlp';
            } else if (stderrData.includes('No video formats found')) {
              errorMessage = 'No downloadable video found at this URL';
            } else if (stderrData.includes('Private video')) {
              errorMessage = 'Private video - Cannot download private content';
            }
            
            resolve({
              success: false,
              error: errorMessage
            });
          }
        });

        ytDlp.on('error', (error) => {
          clearTimeout(downloadTimeout);
          console.error(`❌ yt-dlp spawn error:`, error);
          
          let errorMessage = 'Failed to start video download';
          if (error.message.includes('ENOENT')) {
            errorMessage = 'Python or yt-dlp not found - Please install Python and yt-dlp';
          }
          
          resolve({
            success: false,
            error: errorMessage
          });
        });

      } catch (error) {
        console.error(`❌ Download setup error:`, error);
        resolve({
          success: false,
          error: `Download setup failed: ${error instanceof Error ? error.message : 'Unknown error'}`
        });
      }
    });
  }

  /**
   * Fallback: Download direct video URLs using curl
   */
  async downloadDirectVideoUrl(
    url: string,
    outputDir: string = 'uploads',
    progressCallback?: (progress: { progress: number; message: string; stage: string }) => void
  ): Promise<{ success: boolean; videoPath?: string; metadata?: any; error?: string }> {
    try {
      console.log(`📥 Attempting direct video download: ${url}`);
      
      const timestamp = Date.now();
      const randomId = crypto.randomBytes(8).toString('hex');
      
      // Determine file extension from URL or default to mp4
      let extension = 'mp4';
      try {
        const urlPath = new URL(url).pathname;
        const ext = path.extname(urlPath);
        if (ext && ['.mp4', '.webm', '.avi', '.mov', '.mkv'].includes(ext)) {
          extension = ext.slice(1);
        }
      } catch {}
      
      const outputPath = path.join(outputDir, `direct_download_${timestamp}_${randomId}.${extension}`);
      
      progressCallback?.({
        progress: 5,
        message: 'Starting direct video download...',
        stage: 'download'
      });

      // Use curl for downloading (available on most systems)
      return new Promise((resolve) => {
        const curl = spawn('curl', [
          '-L', // Follow redirects
          '-o', outputPath,
          '--progress-bar',
          '--max-time', '300', // 5 minute timeout
          url
        ]);

        curl.on('close', (code) => {
          if (code === 0) {
            console.log(`✅ Direct download successful: ${outputPath}`);
            resolve({
              success: true,
              videoPath: outputPath,
              metadata: {
                title: 'Downloaded Video',
                duration: 0,
                platform: 'Direct URL',
                originalUrl: url
              }
            });
          } else {
            console.error(`❌ Direct download failed with code: ${code}`);
            resolve({
              success: false,
              error: 'Failed to download video from direct URL'
            });
          }
        });

        curl.on('error', (error) => {
          console.error(`❌ Direct download error:`, error);
          resolve({
            success: false,
            error: 'Direct download failed'
          });
        });
      });

    } catch (error) {
      console.error(`❌ Direct download setup error:`, error);
      return {
        success: false,
        error: 'Failed to setup direct download'
      };
    }
  }

  /**
   * Extract audio from video using Python moviepy
   */
  async extractAudioFromVideo(videoPath: string, audioPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log(`🎵 Extracting audio: ${videoPath} → ${audioPath}`);
      
      // Use absolute path to the Python script
      const pythonScript = path.join(process.cwd(), 'ml/utils/video_to_audio.py');
      
      const python = spawn(this.pythonPath, [pythonScript, videoPath, audioPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUNBUFFERED: '1' }
      });

      let stdout = '';
      let stderr = '';

      python.stdout?.on('data', (data) => {
        const output = data.toString();
        stdout += output;
        console.log(`🐍 Python stdout: ${output.trim()}`);
      });

      python.stderr?.on('data', (data) => {
        const output = data.toString();
        stderr += output;
        console.log(`🐍 Python stderr: ${output.trim()}`);
      });

      python.on('close', (code) => {
        if (code === 0) {
          console.log(`✅ Audio extraction successful`);
          resolve();
        } else {
          console.error(`❌ Python audio extraction failed with code: ${code}`);
          console.error(`Stdout: ${stdout}`);
          console.error(`Stderr: ${stderr}`);
          reject(new Error(`Python audio extraction failed with code: ${code}`));
        }
      });

      python.on('error', (error) => {
        console.error(`❌ Python spawn error:`, error);
        reject(new Error(`Python spawn error: ${error.message}`));
      });
    });
  }

  /**
   * Combine enhanced audio with original video
   */
  async combineAudioWithVideo(videoPath: string, audioPath: string, outputPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn('ffmpeg', [
        '-i', videoPath,
        '-i', audioPath,
        '-c:v', 'copy', // Copy video stream
        '-c:a', 'aac', // AAC audio codec
        '-map', '0:v:0', // Map video from first input
        '-map', '1:a:0', // Map audio from second input
        '-shortest', // Finish when shortest stream ends
        '-y', // Overwrite output file
        outputPath
      ]);

      ffmpeg.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`FFmpeg video combination failed with code: ${code}`));
        }
      });

      ffmpeg.on('error', (error) => {
        reject(new Error(`FFmpeg spawn error: ${error.message}`));
      });
    });
  }

  /**
   * Process video from URL - complete pipeline
   */
  async processVideoFromUrl(
    url: string,
    options: URLVideoProcessingOptions & { downloadType?: 'audio' | 'video' },
    progressCallback?: (progress: { progress: number; message: string; stage: string }) => void
  ): Promise<URLVideoProcessingResult> {
    try {
      console.log(`🚀 Starting URL video processing pipeline for: ${url}`);

      // Decide download method based on URL extension
      const directVideoExtensions = ['.mp4', '.webm', '.avi', '.mov', '.mkv'];
      let useDirectDownload = false;
      try {
        const urlPath = new URL(url).pathname;
        const ext = urlPath ? urlPath.toLowerCase().slice(urlPath.lastIndexOf('.')) : '';
        if (directVideoExtensions.includes(ext)) {
          useDirectDownload = true;
        }
      } catch {}

      let downloadResult;
      if (useDirectDownload) {
        console.log('📥 Detected direct video file URL, using direct download.');
        downloadResult = await this.downloadDirectVideoUrl(url, 'uploads', progressCallback);
      } else {
        console.log('📥 Using yt-dlp for social/video platform URL.');
        downloadResult = await this.downloadVideoFromUrl(url, 'uploads', progressCallback);
      }

      if (!downloadResult.success) {
        throw new Error(downloadResult.error || 'Failed to download video. The source URL may not be supported or the video is unavailable.');
      }


      const downloadedVideoPath = downloadResult.videoPath!;
      console.log(`✅ Video downloaded: ${downloadedVideoPath}`);

      // Validate downloaded file (basic check: file exists and > 100KB)
      let isValidVideo = false;
      try {
        const stats = await fs.stat(downloadedVideoPath);
        if (stats.size > 100 * 1024) {
          isValidVideo = true;
        }
      } catch (err) {
        isValidVideo = false;
      }
      if (!isValidVideo) {
        throw new Error('Downloaded file is not a valid video. The source URL may not be supported or the video is unavailable.');
      }

      // Generate output paths
      const timestamp = Date.now();
      const extractedAudioPath = path.join('uploads', `url_extracted_${timestamp}.wav`);
      const enhancedAudioPath = path.join('outputs', `url_enhanced_audio_${timestamp}.wav`);
      const finalVideoPath = path.join('outputs', `url_enhanced_video_${timestamp}.mp4`);

      // STAGE 2: Extract audio from downloaded video (30-35%)
      progressCallback?.({
        progress: 32,
        message: 'Extracting audio from downloaded video...',
        stage: 'extraction'
      });

      await this.extractAudioFromVideo(downloadedVideoPath, extractedAudioPath);
      console.log(`✅ Audio extracted: ${extractedAudioPath}`);

      // STAGE 3: Enhance audio with DCCRN (35-85%)
      progressCallback?.({
        progress: 35,
        message: 'Starting AI-powered audio enhancement...',
        stage: 'enhancement'
      });

      const enhancementResult = await this.dccrnProcessor.enhanceAudio(
        extractedAudioPath,
        enhancedAudioPath,
        {
          strength: options.denoisingStrength,
          processingMode: options.processingMode
        },
        (progress: any) => {
          // Map DCCRN progress to 35-85% range
          const mappedProgress = 35 + (progress.progress * 0.5);
          progressCallback?.({
            progress: Math.round(mappedProgress),
            message: progress.message,
            stage: 'enhancement'
          });
        }
      );

      if (!enhancementResult.success) {
        throw new Error(`Audio enhancement failed: ${enhancementResult.error}`);
      }

      console.log(`✅ Audio enhanced: ${enhancedAudioPath}`);

      // Determine final output based on downloadType
      const downloadType = options.downloadType || 'audio';
      let finalOutputPath = enhancedAudioPath;

      if (downloadType === 'video') {
        // STAGE 4: Combine enhanced audio with original video (85-95%)
        progressCallback?.({
          progress: 87,
          message: 'Combining enhanced audio with video...',
          stage: 'combination'
        });

        await this.combineAudioWithVideo(downloadedVideoPath, enhancedAudioPath, finalVideoPath);
        console.log(`✅ Final video created: ${finalVideoPath}`);
        finalOutputPath = finalVideoPath;
      } else {
        // For audio-only, we already have the enhanced audio
        progressCallback?.({
          progress: 87,
          message: 'Preparing enhanced audio for download...',
          stage: 'finalization'
        });
      }

      // STAGE 5: Cleanup and finalize (95-100%)
      progressCallback?.({
        progress: 95,
        message: 'Cleaning up temporary files...',
        stage: 'finalization'
      });

      // Clean up temporary files
      try {
        // Don't delete extractedAudioPath - keep it for original audio serving
        await fs.unlink(downloadedVideoPath); // Remove downloaded original video
        
        // For audio-only, don't clean up the enhanced audio since it's our final output
        if (downloadType === 'video') {
          // Keep video output, remove enhanced audio since it's embedded
          await fs.unlink(enhancedAudioPath);
        }
        
        console.log(`🧹 Cleaned up temporary files (kept extracted audio)`);
      } catch (cleanupError) {
        console.warn(`⚠️ Cleanup warning:`, cleanupError);
      }

      progressCallback?.({
        progress: 100,
        message: `${downloadType === 'audio' ? 'Enhanced audio' : 'Enhanced video'} ready for download!`,
        stage: 'completed'
      });

      // Generate AI explanation using Groq service for social media content
      console.log('🤖 About to generate AI explanation...');
      console.log('Enhancement result has aiExplanation:', !!enhancementResult.aiExplanation);
      
      const aiExplanation = enhancementResult.aiExplanation || 
        await this.callGroqExplainer(downloadResult.metadata, options);

      console.log('🔍 Final AI explanation length:', aiExplanation?.length || 0);
      console.log('🔍 AI explanation preview:', aiExplanation?.substring(0, 100) + '...');

      return {
        success: true,
        outputPath: finalOutputPath,
        downloadedVideoPath,
        extractedAudioPath, // Return the extracted audio path
        metadata: {
          title: downloadResult.metadata?.title || 'Unknown',
          duration: downloadResult.metadata?.duration || 0,
          platform: downloadResult.metadata?.platform || 'Unknown',
          originalUrl: url,
          downloadType
        },
        aiExplanation
      };

    } catch (error) {
      console.error(`❌ URL video processing error:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown processing error'
      };
    }
  }

  /**
   * Generate specialized explanation for social media content
   */
  private generateSocialMediaExplanation(metadata: any, options: any): string {
    const platform = metadata?.platform || 'social media';
    const title = metadata?.title || 'Unknown';
    const duration = metadata?.duration || 0;
    const downloadType = options?.downloadType || 'audio';
    
    return `🎯 Social Media Content Enhancement Complete!

✅ PROCESSING SUMMARY:
Successfully processed ${platform} content and enhanced the audio quality using our advanced DCCRN AI model.

📱 SOURCE INFORMATION:
• Platform: ${platform}
• Title: "${title}"
• Duration: ${duration} seconds
• Content Type: ${downloadType === 'audio' ? 'Audio-only' : 'Video with audio'}
• Processing Mode: ${options?.processingMode || 'balanced'}

🔧 ENHANCEMENT PROCESS:
• Downloaded content using yt-dlp for optimal quality
• Extracted high-quality audio from the ${platform} content
• Applied DCCRN (Deep Complex Convolution Recurrent Network) AI enhancement
• Noise reduction level: ${Math.round((options?.denoisingStrength || 0.8) * 10)}/10
• Preserved voice characteristics while removing background noise

🎵 AUDIO IMPROVEMENTS:
• Removed compression artifacts from social media encoding
• Enhanced speech clarity and intelligibility
• Reduced background noise, music interference, and digital distortion
• Improved overall audio quality for better listening experience
• Maintained natural sound dynamics and voice characteristics

⚡ TECHNICAL DETAILS:
• AI Model: DCCRN - Specialized for real-world audio enhancement
• Processing: Spectral domain enhancement optimized for social media content
• Output: Professional-quality ${downloadType === 'audio' ? 'audio file' : 'video with enhanced audio'}
• Compatibility: Enhanced for clarity across all playback devices

Your ${platform} content is now ready with significantly improved audio quality!`;
  }

  /**
   * Call Groq AI service for enhanced explanations
   */
  private async callGroqExplainer(metadata: any, options: any): Promise<string> {
    try {
      console.log('🤖 Generating AI explanation with Groq for social media content...');
      
      // Use the groqService directly instead of Python script
      const explanation = await groqService.generateSocialMediaExplanation({
        platform: metadata?.platform || 'Unknown Platform',
        title: metadata?.title || 'Social Media Content',
        duration: metadata?.duration || 0,
        downloadType: options?.downloadType || 'audio',
        processingMode: options?.processingMode || 'balanced',
        denoisingStrength: options?.denoisingStrength || 0.8,
        originalUrl: metadata?.originalUrl || 'Unknown URL'
      });

      console.log('✅ Groq AI explanation generated successfully for social media content');
      return explanation;

    } catch (error) {
      console.warn('⚠️ Error calling Groq AI for social media content, using fallback:', error);
      return this.generateSocialMediaExplanation(metadata, options);
    }
  }
}

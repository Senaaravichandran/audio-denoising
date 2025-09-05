import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { groqService, type AudioEnhancementOptions } from './groqService';
import { convertAudioFormat, extractAudioFromVideo, getAudioMetadata } from '../utils/ffmpeg';

export interface ProcessingProgress {
  stage: 'analysis' | 'enhancement' | 'conversion' | 'completed';
  progress: number;
  message: string;
  timeRemaining?: number;
}

export interface AudioProcessingResult {
  success: boolean;
  outputPath?: string;
  metadata?: any;
  analysisResult?: any;
  error?: string;
}

export class AudioProcessor {
  private activeJobs = new Map<string, AbortController>();

  async processAudioFile(
    jobId: string,
    inputPath: string,
    options: AudioEnhancementOptions,
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<AudioProcessingResult> {
    const abortController = new AbortController();
    this.activeJobs.set(jobId, abortController);

    try {
      // Stage 1: Analysis
      onProgress?.({
        stage: 'analysis',
        progress: 10,
        message: 'Analyzing audio for noise patterns...'
      });

      const analysisResult = await groqService.analyzeAudioNoise(inputPath);
      
      if (abortController.signal.aborted) {
        throw new Error('Processing cancelled');
      }

      onProgress?.({
        stage: 'analysis',
        progress: 30,
        message: `Detected ${analysisResult.noiseType} noise (${Math.round(analysisResult.noiseLevel * 100)}% noise level)`
      });

      // Stage 2: Enhancement
      onProgress?.({
        stage: 'enhancement',
        progress: 40,
        message: 'Applying AI-powered noise reduction...'
      });

      const enhancedPath = await groqService.enhanceAudio(inputPath, options);
      
      if (abortController.signal.aborted) {
        throw new Error('Processing cancelled');
      }

      onProgress?.({
        stage: 'enhancement',
        progress: 70,
        message: 'Audio enhancement completed'
      });

      // Stage 3: Format conversion if needed
      onProgress?.({
        stage: 'conversion',
        progress: 80,
        message: 'Converting to output format...'
      });

      const outputFormat = options.processingMode === 'music-enhance' ? 'flac' : 'wav';
      const finalOutputPath = await convertAudioFormat(enhancedPath, outputFormat);

      onProgress?.({
        stage: 'conversion',
        progress: 90,
        message: 'Finalizing output...'
      });

      // Get metadata for the processed file
      const metadata = await getAudioMetadata(finalOutputPath);

      onProgress?.({
        stage: 'completed',
        progress: 100,
        message: 'Processing completed successfully!'
      });

      return {
        success: true,
        outputPath: finalOutputPath,
        metadata,
        analysisResult
      };

    } catch (error) {
      console.error(`Error processing audio job ${jobId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    } finally {
      this.activeJobs.delete(jobId);
    }
  }

  async processVideoFile(
    jobId: string,
    inputPath: string,
    options: AudioEnhancementOptions,
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<AudioProcessingResult> {
    try {
      console.log(`Starting video processing for job ${jobId}, input: ${inputPath}`);
      
      // Check if input file exists
      try {
        await fs.access(inputPath);
        console.log(`Video file exists: ${inputPath}`);
      } catch (error) {
        console.error(`Video file not found: ${inputPath}`);
        throw new Error(`Video file not found: ${inputPath}`);
      }

      // Extract audio from video first
      onProgress?.({
        stage: 'analysis',
        progress: 5,
        message: 'Extracting audio from video...'
      });

      console.log(`Extracting audio from video: ${inputPath}`);
      const extractedAudioPath = await extractAudioFromVideo(inputPath);
      console.log(`Audio extracted to: ${extractedAudioPath}`);
      
      // Check if extracted audio file exists
      try {
        await fs.access(extractedAudioPath);
        console.log(`Extracted audio file exists: ${extractedAudioPath}`);
      } catch (error) {
        console.error(`Extracted audio file not found: ${extractedAudioPath}`);
        throw new Error(`Audio extraction failed - output file not created`);
      }
      
      onProgress?.({
        stage: 'analysis',
        progress: 20,
        message: 'Audio extracted successfully'
      });

      // Process the extracted audio
      console.log(`Processing extracted audio: ${extractedAudioPath}`);
      const result = await this.processAudioFile(jobId, extractedAudioPath, options, onProgress);
      
      // Clean up extracted audio file
      try {
        await fs.unlink(extractedAudioPath);
        console.log(`Cleaned up extracted audio file: ${extractedAudioPath}`);
      } catch (error) {
        console.warn(`Failed to clean up extracted audio file: ${extractedAudioPath}`, error);
      }
      
      return result;

    } catch (error) {
      console.error(`Error processing video job ${jobId}:`, error);
      console.error(`Error details:`, {
        name: error instanceof Error ? error.name : 'Unknown',
        message: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : undefined
      });
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  async processBatchFiles(
    files: Array<{ jobId: string; inputPath: string; options: AudioEnhancementOptions }>,
    onProgress?: (jobId: string, progress: ProcessingProgress) => void
  ): Promise<Map<string, AudioProcessingResult>> {
    const results = new Map<string, AudioProcessingResult>();
    
    // Process files in parallel with concurrency limit
    const concurrencyLimit = 3;
    const chunks = this.chunkArray(files, concurrencyLimit);
    
    for (const chunk of chunks) {
      const promises = chunk.map(file => 
        this.processAudioFile(
          file.jobId,
          file.inputPath,
          file.options,
          (progress) => onProgress?.(file.jobId, progress)
        ).then(result => ({ jobId: file.jobId, result }))
      );
      
      const chunkResults = await Promise.all(promises);
      chunkResults.forEach(({ jobId, result }) => {
        results.set(jobId, result);
      });
    }
    
    return results;
  }

  cancelJob(jobId: string): boolean {
    const controller = this.activeJobs.get(jobId);
    if (controller) {
      controller.abort();
      this.activeJobs.delete(jobId);
      return true;
    }
    return false;
  }

  getActiveJobs(): string[] {
    return Array.from(this.activeJobs.keys());
  }

  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }

  async processLongAudioFile(
    jobId: string,
    inputPath: string,
    options: AudioEnhancementOptions,
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<AudioProcessingResult> {
    try {
      // Get audio duration and split into chunks if necessary
      const metadata = await getAudioMetadata(inputPath);
      const duration = metadata.duration || 0;
      
      // If longer than 10 minutes, process in chunks
      if (duration > 600) {
        return await this.processInChunks(jobId, inputPath, options, duration, onProgress);
      } else {
        return await this.processAudioFile(jobId, inputPath, options, onProgress);
      }
    } catch (error) {
      console.error(`Error processing long audio file ${jobId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  private async processInChunks(
    jobId: string,
    inputPath: string,
    options: AudioEnhancementOptions,
    duration: number,
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<AudioProcessingResult> {
    const chunkDuration = 300; // 5 minutes per chunk
    const numChunks = Math.ceil(duration / chunkDuration);
    const processedChunks: string[] = [];

    for (let i = 0; i < numChunks; i++) {
      const startTime = i * chunkDuration;
      const chunkProgress = (i / numChunks) * 100;
      
      onProgress?.({
        stage: 'enhancement',
        progress: chunkProgress,
        message: `Processing chunk ${i + 1} of ${numChunks}...`
      });

      // Extract chunk
      const chunkPath = await this.extractAudioChunk(inputPath, startTime, chunkDuration, i);
      
      // Process chunk
      const chunkResult = await this.processAudioFile(`${jobId}_chunk_${i}`, chunkPath, options);
      
      if (!chunkResult.success || !chunkResult.outputPath) {
        throw new Error(`Failed to process chunk ${i + 1}`);
      }
      
      processedChunks.push(chunkResult.outputPath);
    }

    // Concatenate processed chunks
    onProgress?.({
      stage: 'conversion',
      progress: 90,
      message: 'Combining processed chunks...'
    });

    const finalOutputPath = await this.concatenateAudioChunks(processedChunks, jobId);
    
    return {
      success: true,
      outputPath: finalOutputPath,
      metadata: await getAudioMetadata(finalOutputPath)
    };
  }

  private async extractAudioChunk(inputPath: string, startTime: number, duration: number, index: number): Promise<string> {
    const outputPath = path.join(path.dirname(inputPath), `chunk_${index}_${path.basename(inputPath)}`);
    
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn('ffmpeg', [
        '-i', inputPath,
        '-ss', startTime.toString(),
        '-t', duration.toString(),
        '-c', 'copy',
        '-y',
        outputPath
      ]);

      ffmpeg.on('close', (code) => {
        if (code === 0) {
          resolve(outputPath);
        } else {
          reject(new Error(`FFmpeg chunk extraction failed with code ${code}`));
        }
      });

      ffmpeg.on('error', reject);
    });
  }

  private async concatenateAudioChunks(chunkPaths: string[], jobId: string): Promise<string> {
    const outputPath = path.join(path.dirname(chunkPaths[0]), `final_${jobId}.wav`);
    const concatListPath = path.join(path.dirname(chunkPaths[0]), `concat_${jobId}.txt`);
    
    // Create concat file list
    const concatList = chunkPaths.map(p => `file '${p}'`).join('\n');
    await fs.writeFile(concatListPath, concatList);
    
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn('ffmpeg', [
        '-f', 'concat',
        '-safe', '0',
        '-i', concatListPath,
        '-c', 'copy',
        '-y',
        outputPath
      ]);

      ffmpeg.on('close', async (code) => {
        if (code === 0) {
          // Clean up temporary files
          await fs.unlink(concatListPath);
          for (const chunkPath of chunkPaths) {
            await fs.unlink(chunkPath).catch(() => {}); // Ignore errors
          }
          resolve(outputPath);
        } else {
          reject(new Error(`FFmpeg concatenation failed with code ${code}`));
        }
      });

      ffmpeg.on('error', reject);
    });
  }
}

export const audioProcessor = new AudioProcessor();

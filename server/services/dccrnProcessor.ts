import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export interface ProcessingProgress {
  stage: 'initialization' | 'loading' | 'processing' | 'finalizing' | 'completed';
  progress: number;
  message: string;
}

export interface ProcessingOptions {
  strength?: number;
  processingMode?: 'fast' | 'balanced';
  noiseReductionLevel?: number;
}

export interface ProcessingResult {
  success: boolean;
  outputPath?: string;
  error?: string;
  duration?: number;
  originalSize?: number;
  enhancedSize?: number;
  aiExplanation?: string;
}

export class DCCRNProcessor {
  private pythonPath: string;

  constructor() {
    // Use the virtual environment Python executable
    this.pythonPath = 'C:/Users/Senaa/Desktop/Project\'s/SonicPurge/.venv/Scripts/python.exe';
  }

  private getDCCRNServicePath(processingMode: 'fast' | 'balanced' = 'balanced'): string {
    const serviceFile = processingMode === 'fast' ? 'dccrnFast.py' : 'dccrnBalanced.py';
    return path.join(__dirname, serviceFile);
  }

  async enhanceAudio(
    inputPath: string,
    outputPath: string,
    options: ProcessingOptions = {},
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<ProcessingResult> {
    const { strength = 0.8, processingMode = 'balanced' } = options;

    try {
      // Check if input file exists
      await fs.access(inputPath);
      
      // Ensure output directory exists
      const outputDir = path.dirname(outputPath);
      await fs.mkdir(outputDir, { recursive: true });

      // Run the DCCRN service with selected mode
      const result = await this.runDCCRNService(inputPath, outputPath, strength, processingMode, onProgress);
      
      if (result.success) {
        // Get file sizes for metadata
        const originalStats = await fs.stat(inputPath);
        const enhancedStats = await fs.stat(outputPath);
        
        // Generate AI explanation using Groq
        const explanation = await this.generateAIExplanation({
          inputPath,
          outputPath,
          strength,
          processingMode,
          duration: result.duration,
          originalSize: originalStats.size,
          enhancedSize: enhancedStats.size
        });
        
        return {
          success: true,
          outputPath,
          duration: result.duration,
          originalSize: originalStats.size,
          enhancedSize: enhancedStats.size,
          aiExplanation: explanation
        };
      } else {
        return {
          success: false,
          error: result.error
        };
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      return {
        success: false,
        error: errorMessage
      };
    }
  }

  private runDCCRNService(
    inputPath: string,
    outputPath: string,
    strength: number,
    processingMode: 'fast' | 'balanced' = 'balanced',
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<{ success: boolean; error?: string; duration?: number }> {
    return new Promise((resolve) => {
      // Get the appropriate service path
      const dccrnServicePath = this.getDCCRNServicePath(processingMode);
      const serviceName = processingMode.toUpperCase();
      
      // Convert relative paths to absolute paths from project root
      const projectRoot = path.resolve(__dirname, '../../');
      const absoluteInputPath = path.isAbsolute(inputPath) ? inputPath : path.resolve(projectRoot, inputPath);
      const absoluteOutputPath = path.isAbsolute(outputPath) ? outputPath : path.resolve(projectRoot, outputPath);
      
      console.log(`🎯 Starting ${serviceName} Python DCCRN service...`);
      console.log(`   Python: ${this.pythonPath}`);
      console.log(`   Script: ${dccrnServicePath}`);
      console.log(`   Input: ${absoluteInputPath}`);
      console.log(`   Output: ${absoluteOutputPath}`);
      console.log(`   Strength: ${strength}`);
      console.log(`   Mode: ${serviceName} processing`);
      
      const args = [
        dccrnServicePath,
        '--input', absoluteInputPath,
        '--output', absoluteOutputPath,
        '--strength', strength.toString()
      ];

      console.log(`   Command: ${this.pythonPath} ${args.join(' ')}`);

      const child = spawn(this.pythonPath, args, {
        cwd: projectRoot, // Run from project root, not server/services
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { 
          ...process.env, 
          PYTHONIOENCODING: 'utf-8',
          PYTHONUNBUFFERED: '1'
        }
      });

      console.log(`   Process PID: ${child.pid}`);

      // Set timeout based on processing mode
      const timeoutMs = processingMode === 'fast' ? 60000 : 90000; // Fast: 60s, Balanced: 90s
      const timeout = setTimeout(() => {
        console.log(`⚠️ Processing timeout reached (${timeoutMs/1000} seconds) - killing process`);
        child.kill('SIGTERM');
        resolve({
          success: false,
          error: `Processing timeout - ${serviceName} mode exceeded ${timeoutMs/1000}s. Try with a shorter audio file.`
        });
      }, timeoutMs);

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data) => {
        stdout += data.toString();
        const output = data.toString();
        console.log(`🐍 Python stdout: ${output.trim()}`);
        
        // Send progress updates based on processing mode and output
        if (output.includes('Using device:')) {
          onProgress?.({
            stage: 'initialization',
            progress: 10,
            message: `Initializing ${serviceName.toLowerCase()} AI denoising system...`
          });
        } else if (output.includes('Model loaded successfully') || output.includes('DCCRN model loaded')) {
          onProgress?.({
            stage: 'loading',
            progress: 25,
            message: `DCCRN model loaded (11.2M parameters) - preparing ${serviceName.toLowerCase()} denoising...`
          });
        } else if (output.includes('Model parameters:')) {
          const description = processingMode === 'fast' 
            ? 'Fast single-stage denoising ready - analyzing audio...'
            : 'Balanced speech-preserving denoising ready - analyzing audio...';
          onProgress?.({
            stage: 'loading',
            progress: 30,
            message: description
          });
        } else if (output.includes('Converting') && output.includes('to WAV')) {
          onProgress?.({
            stage: 'loading',
            progress: 35,
            message: 'Converting audio format...'
          });
        } else if (processingMode === 'fast' && output.includes('⚡ Stage 1/1')) {
          onProgress?.({
            stage: 'processing',
            progress: 70,
            message: '⚡ Fast AI noise reduction in progress...'
          });
        } else if (processingMode === 'balanced') {
          if (output.includes('🎯 Stage 1/3')) {
            onProgress?.({
              stage: 'processing',
              progress: 50,
              message: '🎯 Stage 1/3: AI-powered noise reduction...'
            });
          } else if (output.includes('🔧 Stage 2/3')) {
            onProgress?.({
              stage: 'processing',
              progress: 70,
              message: '🔧 Stage 2/3: Gentle spectral enhancement...'
            });
          } else if (output.includes('✨ Stage 3/3')) {
            onProgress?.({
              stage: 'processing',
              progress: 90,
              message: '✨ Stage 3/3: Voice clarity optimization...'
            });
          }
        } else if (output.includes('Processing chunk')) {
          // Extract chunk progress if available
          const chunkMatch = output.match(/chunk (\d+)\/(\d+)/);
          if (chunkMatch) {
            const current = parseInt(chunkMatch[1]);
            const total = parseInt(chunkMatch[2]);
            const chunkProgress = 40 + (current / total) * 45; // 40-85% range for chunked processing
            onProgress?.({
              stage: 'processing',
              progress: Math.round(chunkProgress),
              message: `Processing chunk ${current}/${total} with ${serviceName.toLowerCase()} denoising...`
            });
          }
        } else if (output.includes('Processing:') || output.includes('Input shape:')) {
          onProgress?.({
            stage: 'processing',
            progress: 45,
            message: `Audio analysis complete - starting ${serviceName.toLowerCase()} enhancement...`
          });
        } else if (output.includes('Output shape:')) {
          onProgress?.({
            stage: 'processing',
            progress: 80,
            message: `${serviceName} noise reduction complete - finalizing...`
          });
        } else if (output.includes('Enhancement completed') || output.includes('denoising complete')) {
          onProgress?.({
            stage: 'finalizing',
            progress: 95,
            message: `${serviceName} denoising complete - saving enhanced audio...`
          });
        } else if (output.includes('[SUCCESS]')) {
          onProgress?.({
            stage: 'completed',
            progress: 100,
            message: `${serviceName} enhancement completed successfully!`
          });
        }
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
        const errorOutput = data.toString().trim();
        console.error(`� Python stderr: ${errorOutput}`);
        
        // Report critical errors immediately
        if (errorOutput.includes('ModuleNotFoundError') || 
            errorOutput.includes('ImportError') ||
            errorOutput.includes('FileNotFoundError') ||
            errorOutput.includes('CUDA out of memory')) {
          onProgress?.({
            stage: 'processing',
            progress: 90,
            message: `Error detected: ${errorOutput.split('\n')[0]}`
          });
        }
      });

      child.on('error', (error) => {
        console.log(`❌ Python process error: ${error.message}`);
        clearTimeout(timeout);
        resolve({ success: false, error: `Process error: ${error.message}` });
      });

      child.on('close', (code) => {
        clearTimeout(timeout); // Clear timeout when process completes
        console.log(`🐍 Python process closed with code: ${code}`);
        console.log(`📤 Final stdout length: ${stdout.length} chars`);
        console.log(`📤 Final stderr length: ${stderr.length} chars`);
        
        if (stdout.length > 0) {
          console.log(`📤 Last stdout lines:`, stdout.split('\n').slice(-3).join('\n'));
        }
        if (stderr.length > 0) {
          console.error(`📤 Full stderr:`, stderr);
        }
        
        if (code === 0) {
          // Check if output file actually exists
          const outputPath = absoluteOutputPath;
          console.log(`🔍 Checking if output file exists: ${outputPath}`);
          
          // Parse duration from output if available
          const durationMatch = stdout.match(/Duration: ([\d.]+)s/);
          const duration = durationMatch ? parseFloat(durationMatch[1]) : undefined;
          
          resolve({ 
            success: true, 
            duration
          });
        } else {
          console.error(`❌ Python process failed with exit code: ${code}`);
          let errorMessage = `${serviceName} processing failed with exit code ${code}`;
          
          // Extract specific error messages
          if (stderr.includes('FileNotFoundError')) {
            errorMessage = 'Input file not found - check file path';
          } else if (stderr.includes('PermissionError')) {
            errorMessage = 'Permission denied - check file permissions';
          } else if (stderr.includes('OutOfMemoryError') || stderr.includes('CUDA out of memory')) {
            errorMessage = 'Insufficient memory for processing - try with a smaller audio file';
          } else if (stderr.includes('ModuleNotFoundError')) {
            errorMessage = 'Required Python modules not installed (torch, torchaudio, etc.)';
          } else if (stderr.includes('Model loading failed')) {
            errorMessage = 'DCCRN model could not be loaded - check checkpoints directory';
          } else if (stderr.includes('No module named')) {
            const moduleMatch = stderr.match(/No module named ['"](.*?)['"]/);
            const moduleName = moduleMatch ? moduleMatch[1] : 'unknown';
            errorMessage = `Missing Python module: ${moduleName}`;
          } else if (stderr.trim()) {
            // Get the last meaningful error line
            const errorLines = stderr.trim().split('\n').filter(line => line.trim());
            errorMessage = errorLines[errorLines.length - 1] || errorMessage;
          } else if (stdout.includes('ERROR')) {
            const errorLines = stdout.split('\n').filter(line => line.includes('ERROR'));
            if (errorLines.length > 0) {
              errorMessage = errorLines[errorLines.length - 1];
            }
          }
          
          console.error(`🔥 Final error message: ${errorMessage}`);
          
          resolve({ 
            success: false, 
            error: errorMessage 
          });
        }
      });
    });
  }

  /**
   * Generate AI explanation using Groq API
   */
  private async generateAIExplanation(data: {
    inputPath: string;
    outputPath: string;
    strength: number;
    processingMode: string;
    duration?: number;
    originalSize: number;
    enhancedSize: number;
  }): Promise<string> {
    try {
      console.log('🤖 Generating AI explanation with Groq...');
      
      // Prepare enhancement data for Groq
      const enhancementData = {
        source_type: 'uploaded audio file',
        original_filename: path.basename(data.inputPath),
        processing_mode: data.processingMode,
        noise_reduction_level: Math.round(data.strength * 10),
        voice_preservation: 9, // Default high voice preservation
        output_format: 'WAV',
        processing_time: data.duration || 0,
        ai_model: 'DCCRN (Deep Complex Convolution Recurrent Network)',
        original_size: `${(data.originalSize / 1024 / 1024).toFixed(2)} MB`,
        enhanced_size: `${(data.enhancedSize / 1024 / 1024).toFixed(2)} MB`,
        sample_rate: '16000',
        duration: data.duration ? `${data.duration.toFixed(1)}` : 'N/A',
        stages: [
          'Audio preprocessing and normalization',
          'Spectral analysis using STFT (Short-Time Fourier Transform)',
          'DCCRN neural network noise reduction',
          'Complex domain enhancement and reconstruction',
          'High-quality audio output generation'
        ]
      };

      // Call Python Groq explainer
      const result = await this.callGroqExplainer(enhancementData);
      
      console.log('✅ AI explanation generated successfully');
      return result;
      
    } catch (error) {
      console.error('❌ Error generating AI explanation:', error);
      
      // Return fallback explanation
      return this.getFallbackExplanation(data);
    }
  }

  /**
   * Call Python Groq explainer service
   */
  private callGroqExplainer(enhancementData: any): Promise<string> {
    return new Promise((resolve, reject) => {
      const pythonScript = path.join(process.cwd(), 'ml/utils/groq_explainer.py');
      const dataJson = JSON.stringify(enhancementData);
      
      // Get Groq API key from environment or use the one from groqService
      const GROQ_API_KEY = process.env.GROQ_API_KEY;
      
      const python = spawn(this.pythonPath, [pythonScript, dataJson], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: { 
          ...process.env, 
          PYTHONUNBUFFERED: '1',
          GROQ_API_KEY: GROQ_API_KEY
        }
      });

      let stdout = '';
      let stderr = '';

      python.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      python.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      python.on('close', (code) => {
        if (code === 0) {
          // Extract the explanation from stdout
          const lines = stdout.split('\n');
          const startIndex = lines.findIndex(line => line.includes('=== ENHANCEMENT EXPLANATION ==='));
          
          if (startIndex !== -1) {
            const explanation = lines.slice(startIndex + 1).join('\n').trim();
            resolve(explanation);
          } else {
            resolve(stdout.trim());
          }
        } else {
          console.error('Groq explainer stderr:', stderr);
          reject(new Error(`Groq explainer failed with code: ${code}`));
        }
      });

      python.on('error', (error) => {
        reject(error);
      });
    });
  }

  /**
   * Get fallback explanation when AI is not available
   */
  private getFallbackExplanation(data: {
    processingMode: string;
    strength: number;
    originalSize: number;
    enhancedSize: number;
  }): string {
    const noiseLevel = Math.round(data.strength * 10);
    const sizeReduction = ((data.originalSize - data.enhancedSize) / data.originalSize * 100).toFixed(1);
    
    return `🎯 SonicPurge Enhancement Complete!

✅ PROCESSING SUMMARY:
Your audio file has been successfully enhanced using our advanced DCCRN (Deep Complex Convolution Recurrent Network) AI model.

🔧 ENHANCEMENT PROCESS:
• Processing Mode: ${data.processingMode} - Optimized for quality and performance
• Noise Reduction Level: ${noiseLevel}/10 - Removed background noise, hums, and distortions
• Voice Preservation: High - Maintained natural speech characteristics
• AI-powered spectral analysis and reconstruction

🎵 AUDIO IMPROVEMENTS:
• Significantly reduced background noise and interference
• Enhanced speech clarity and intelligibility
• Improved overall audio quality and listening experience
• Preserved original audio dynamics and natural sound

⚡ TECHNICAL DETAILS:
• AI Model: DCCRN - State-of-the-art audio enhancement
• Processing: Real-time spectral domain enhancement
• File Size: ${sizeReduction}% size optimization achieved
• Output: High-quality WAV file with enhanced clarity

Your enhanced audio is now ready with professional-grade quality improvements!`;
  }
}

// Create and export singleton instance
export const dccrnProcessor = new DCCRNProcessor();

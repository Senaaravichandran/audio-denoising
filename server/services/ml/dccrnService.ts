/// <reference types="node" />
import { spawn } from 'child_process';
import * as path from 'path';
import { promises as fs } from 'fs';
import { Buffer } from 'buffer';

export interface DCCRNInferenceOptions {
  inputPath: string;
  outputPath: string;
  denoisingStrength?: number;
  modelPath?: string;
}

export interface InferenceResult {
  success: boolean;
  outputPath?: string;
  error?: string;
  spectrograms?: {
    noisy?: string;
    enhanced?: string;
  };
}

export class DCCRNService {
  private modelPath: string;
  private pythonExecutable: string;
  private inferenceScriptPath: string;

  constructor() {
    this.modelPath = path.join(process.cwd(), 'checkpoints', 'dccrn_model.pt');
    this.pythonExecutable = process.env.PYTHON_EXECUTABLE || 'python';
    this.inferenceScriptPath = path.join(process.cwd(), 'ml', 'inference.py');
  }

  /**
   * Check if the DCCRN model exists
   */
  async isModelAvailable(): Promise<boolean> {
    try {
      await fs.access(this.modelPath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Run DCCRN inference on audio file
   */
  async enhanceAudio(options: DCCRNInferenceOptions): Promise<InferenceResult> {
    const {
      inputPath,
      outputPath,
      denoisingStrength = 1.0,
      modelPath = this.modelPath
    } = options;

    return new Promise((resolve) => {
      const args = [
        this.inferenceScriptPath,
        '--model', modelPath,
        '--input', inputPath,
        '--output', outputPath,
        '--strength', denoisingStrength.toString()
      ];

      console.log(`Running DCCRN inference: ${this.pythonExecutable} ${args.join(' ')}`);

      const process = spawn(this.pythonExecutable, args, {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      process.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
      });

      process.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
      });

      process.on('close', async (code: number | null) => {
        if (code === 0) {
          // Check if output file exists
          try {
            await fs.access(outputPath);
            resolve({
              success: true,
              outputPath
            });
          } catch {
            resolve({
              success: false,
              error: 'Output file was not created'
            });
          }
        } else {
          console.error(`DCCRN inference failed with code ${code}`);
          console.error('stderr:', stderr);
          resolve({
            success: false,
            error: stderr || `Process exited with code ${code}`
          });
        }
      });

      process.on('error', (error) => {
        console.error('DCCRN inference process error:', error);
        resolve({
          success: false,
          error: error.message
        });
      });
    });
  }

  /**
   * Enhance audio with different strength levels
   */
  async enhanceAudioWithStrength(
    inputPath: string,
    outputDir: string,
    strength: 'mild' | 'medium' | 'strong' = 'medium'
  ): Promise<InferenceResult> {
    const strengthMap = {
      mild: 0.3,
      medium: 0.7,
      strong: 1.0
    };

    const denoisingStrength = strengthMap[strength];
    const outputFilename = `enhanced_${strength}_${path.basename(inputPath)}`;
    const outputPath = path.join(outputDir, outputFilename);

    return this.enhanceAudio({
      inputPath,
      outputPath,
      denoisingStrength
    });
  }

  /**
   * Batch process multiple audio files
   */
  async enhanceBatch(
    inputDir: string,
    outputDir: string,
    denoisingStrength: number = 1.0
  ): Promise<InferenceResult> {
    return new Promise((resolve) => {
      const args = [
        this.inferenceScriptPath,
        '--model', this.modelPath,
        '--input', inputDir,
        '--output', outputDir,
        '--strength', denoisingStrength.toString(),
        '--batch'
      ];

      const process = spawn(this.pythonExecutable, args, {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stderr = '';

      process.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0) {
          resolve({
            success: true,
            outputPath: outputDir
          });
        } else {
          resolve({
            success: false,
            error: stderr || `Process exited with code ${code}`
          });
        }
      });

      process.on('error', (error) => {
        resolve({
          success: false,
          error: error.message
        });
      });
    });
  }

  /**
   * Generate spectrograms for visualization
   */
  async generateSpectrograms(
    noisyPath: string,
    enhancedPath: string,
    outputDir: string
  ): Promise<{ noisy: string; enhanced: string } | null> {
    // This would call a Python script to generate spectrograms
    // For now, return null as placeholder
    return null;
  }

  /**
   * Train the DCCRN model
   */
  async trainModel(
    cleanDataDir: string,
    noisyDataDir: string,
    configPath?: string,
    resumeFromCheckpoint?: string
  ): Promise<{ success: boolean; error?: string }> {
    return new Promise((resolve) => {
      const trainScriptPath = path.join(global.process.cwd(), 'ml', 'training', 'train.py');
      const config: string = configPath || path.join(global.process.cwd(), 'ml', 'training', 'config.yaml');

      const args: string[] = [trainScriptPath, '--config', config];
      
      if (resumeFromCheckpoint) {
        args.push('--resume', resumeFromCheckpoint);
      }

      console.log(`Starting DCCRN training: ${this.pythonExecutable} ${args.join(' ')}`);

      const childProcess = spawn(this.pythonExecutable, args, {
        stdio: ['pipe', 'pipe', 'pipe'],
        detached: true // Allow process to continue running
      });

      let stderr = '';

      childProcess.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
        console.log('Training:', data.toString());
      });

      childProcess.on('close', (code: number | null) => {
        if (code === 0) {
          resolve({ success: true });
        } else {
          resolve({
            success: false,
            error: stderr || `Training process exited with code ${code}`
          });
        }
      });

      childProcess.on('error', (error: Error) => {
        resolve({
          success: false,
          error: error.message
        });
      });

      // For long-running training, we might want to resolve immediately
      // and handle progress updates via WebSocket
      setTimeout(() => {
        resolve({ success: true });
      }, 1000);
    });
  }
}

// Singleton instance
export const dccrnService = new DCCRNService();

/**
 * Audio utility functions for format conversion, analysis, and manipulation
 */

export interface AudioMetadata {
  duration: number;
  format: string;
  bitrate: number;
  sampleRate: number;
  channels: number;
  codec: string;
  fileSize: number;
}

export interface AudioAnalysis {
  rms: number;
  peak: number;
  spectralCentroid: number;
  zeroCrossingRate: number;
  spectralRolloff: number;
  mfcc: number[];
}

export interface NoiseProfile {
  frequencies: number[];
  amplitudes: number[];
  type: 'stationary' | 'non-stationary' | 'impulse';
  characteristics: {
    lowFrequency: boolean;
    midFrequency: boolean;
    highFrequency: boolean;
    periodic: boolean;
    broadband: boolean;
  };
}

/**
 * Supported audio formats with their MIME types
 */
export const SUPPORTED_FORMATS = {
  audio: {
    'wav': 'audio/wav',
    'mp3': 'audio/mpeg',
    'flac': 'audio/flac',
    'aac': 'audio/aac',
    'ogg': 'audio/ogg',
    'm4a': 'audio/mp4',
    'wma': 'audio/x-ms-wma',
    'aiff': 'audio/aiff',
    'au': 'audio/au',
  },
  video: {
    'mp4': 'video/mp4',
    'avi': 'video/x-msvideo',
    'mov': 'video/quicktime',
    'mkv': 'video/x-matroska',
    'webm': 'video/webm',
    'flv': 'video/x-flv',
    'wmv': 'video/x-ms-wmv',
  }
} as const;

/**
 * Check if a file is a supported audio format
 */
export function isAudioFile(filename: string): boolean {
  const extension = getFileExtension(filename);
  return extension in SUPPORTED_FORMATS.audio;
}

/**
 * Check if a file is a supported video format
 */
export function isVideoFile(filename: string): boolean {
  const extension = getFileExtension(filename);
  return extension in SUPPORTED_FORMATS.video;
}

/**
 * Get file extension from filename
 */
export function getFileExtension(filename: string): string {
  return filename.split('.').pop()?.toLowerCase() || '';
}

/**
 * Get MIME type for a given file
 */
export function getMimeType(filename: string): string {
  const extension = getFileExtension(filename);
  return SUPPORTED_FORMATS.audio[extension as keyof typeof SUPPORTED_FORMATS.audio] ||
         SUPPORTED_FORMATS.video[extension as keyof typeof SUPPORTED_FORMATS.video] ||
         'application/octet-stream';
}

/**
 * Format file size in human-readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format duration in human-readable format
 */
export function formatDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '0:00';
  
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  }
  
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

/**
 * Convert audio buffer to WAV blob
 */
export function audioBufferToWav(audioBuffer: AudioBuffer): Blob {
  const numberOfChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const length = audioBuffer.length;
  const buffer = new ArrayBuffer(44 + length * numberOfChannels * 2);
  const view = new DataView(buffer);
  
  // WAV header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length * numberOfChannels * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numberOfChannels * 2, true);
  view.setUint16(32, numberOfChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, length * numberOfChannels * 2, true);
  
  // Convert float samples to 16-bit PCM
  const offset = 44;
  let index = 0;
  
  for (let i = 0; i < length; i++) {
    for (let channel = 0; channel < numberOfChannels; channel++) {
      const sample = audioBuffer.getChannelData(channel)[i];
      const intSample = Math.max(-1, Math.min(1, sample)) * 0x7FFF;
      view.setInt16(offset + index * 2, intSample, true);
      index++;
    }
  }
  
  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Analyze audio buffer for basic metrics
 */
export function analyzeAudioBuffer(audioBuffer: AudioBuffer): AudioAnalysis {
  const channelData = audioBuffer.getChannelData(0); // Use first channel
  const length = channelData.length;
  
  let rmsSum = 0;
  let peak = 0;
  let zeroCrossings = 0;
  
  // Calculate RMS and peak
  for (let i = 0; i < length; i++) {
    const sample = Math.abs(channelData[i]);
    rmsSum += sample * sample;
    peak = Math.max(peak, sample);
    
    // Count zero crossings
    if (i > 0 && 
        ((channelData[i] >= 0) !== (channelData[i - 1] >= 0))) {
      zeroCrossings++;
    }
  }
  
  const rms = Math.sqrt(rmsSum / length);
  const zeroCrossingRate = zeroCrossings / length;
  
  // Simple spectral analysis (would need FFT for proper implementation)
  const spectralCentroid = estimateSpectralCentroid(channelData);
  const spectralRolloff = estimateSpectralRolloff(channelData);
  const mfcc = estimateMFCC(channelData);
  
  return {
    rms,
    peak,
    spectralCentroid,
    zeroCrossingRate,
    spectralRolloff,
    mfcc,
  };
}

/**
 * Estimate spectral centroid (simplified)
 */
function estimateSpectralCentroid(channelData: Float32Array): number {
  // Simplified estimation based on time-domain features
  // In reality, this would require FFT analysis
  let weightedSum = 0;
  let magnitudeSum = 0;
  
  for (let i = 0; i < channelData.length; i++) {
    const magnitude = Math.abs(channelData[i]);
    weightedSum += i * magnitude;
    magnitudeSum += magnitude;
  }
  
  return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
}

/**
 * Estimate spectral rolloff (simplified)
 */
function estimateSpectralRolloff(channelData: Float32Array): number {
  // Simplified estimation
  const threshold = 0.85;
  let cumulativeEnergy = 0;
  let totalEnergy = 0;
  
  for (let i = 0; i < channelData.length; i++) {
    totalEnergy += channelData[i] * channelData[i];
  }
  
  const targetEnergy = totalEnergy * threshold;
  
  for (let i = 0; i < channelData.length; i++) {
    cumulativeEnergy += channelData[i] * channelData[i];
    if (cumulativeEnergy >= targetEnergy) {
      return i / channelData.length;
    }
  }
  
  return 1.0;
}

/**
 * Estimate MFCC coefficients (simplified)
 */
function estimateMFCC(channelData: Float32Array): number[] {
  // Simplified MFCC estimation
  // In reality, this would require mel-scale filtering and DCT
  const numCoeffs = 13;
  const coefficients: number[] = [];
  
  const segmentSize = Math.floor(channelData.length / numCoeffs);
  
  for (let i = 0; i < numCoeffs; i++) {
    let energy = 0;
    const start = i * segmentSize;
    const end = Math.min(start + segmentSize, channelData.length);
    
    for (let j = start; j < end; j++) {
      energy += channelData[j] * channelData[j];
    }
    
    coefficients.push(Math.log(energy + 1e-10));
  }
  
  return coefficients;
}

/**
 * Generate a noise profile from audio data
 */
export function generateNoiseProfile(audioBuffer: AudioBuffer): NoiseProfile {
  const analysis = analyzeAudioBuffer(audioBuffer);
  const channelData = audioBuffer.getChannelData(0);
  
  // Simple frequency analysis (would need FFT for proper implementation)
  const frequencies: number[] = [];
  const amplitudes: number[] = [];
  
  // Divide into frequency bands
  const numBands = 20;
  const bandSize = Math.floor(channelData.length / numBands);
  
  for (let i = 0; i < numBands; i++) {
    const start = i * bandSize;
    const end = Math.min(start + bandSize, channelData.length);
    const frequency = (i + 1) * (audioBuffer.sampleRate / 2) / numBands;
    
    let energy = 0;
    for (let j = start; j < end; j++) {
      energy += channelData[j] * channelData[j];
    }
    
    frequencies.push(frequency);
    amplitudes.push(Math.sqrt(energy / (end - start)));
  }
  
  // Determine noise characteristics
  const lowFreqEnergy = amplitudes.slice(0, 3).reduce((a, b) => a + b, 0);
  const midFreqEnergy = amplitudes.slice(3, 12).reduce((a, b) => a + b, 0);
  const highFreqEnergy = amplitudes.slice(12).reduce((a, b) => a + b, 0);
  const totalEnergy = lowFreqEnergy + midFreqEnergy + highFreqEnergy;
  
  const characteristics = {
    lowFrequency: (lowFreqEnergy / totalEnergy) > 0.4,
    midFrequency: (midFreqEnergy / totalEnergy) > 0.4,
    highFrequency: (highFreqEnergy / totalEnergy) > 0.4,
    periodic: analysis.zeroCrossingRate < 0.1,
    broadband: amplitudes.filter(a => a > 0.1).length > numBands * 0.7,
  };
  
  // Determine noise type
  let type: NoiseProfile['type'] = 'stationary';
  if (analysis.zeroCrossingRate > 0.3) {
    type = 'non-stationary';
  } else if (analysis.peak > analysis.rms * 3) {
    type = 'impulse';
  }
  
  return {
    frequencies,
    amplitudes,
    type,
    characteristics,
  };
}

/**
 * Validate audio file constraints
 */
export function validateAudioFile(file: File): { valid: boolean; error?: string } {
  const maxSize = 500 * 1024 * 1024; // 500MB
  const maxDuration = 3 * 60 * 60; // 3 hours in seconds
  
  if (file.size > maxSize) {
    return {
      valid: false,
      error: `File size exceeds maximum limit of ${formatFileSize(maxSize)}`
    };
  }
  
  if (!isAudioFile(file.name) && !isVideoFile(file.name)) {
    return {
      valid: false,
      error: 'Unsupported file format'
    };
  }
  
  return { valid: true };
}

/**
 * Create download link for audio blob
 */
export function downloadAudioBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Calculate audio quality score based on analysis
 */
export function calculateQualityScore(analysis: AudioAnalysis): number {
  // Simple quality scoring based on audio characteristics
  let score = 100;
  
  // Penalize for low RMS (quiet audio)
  if (analysis.rms < 0.1) score -= 20;
  
  // Penalize for high zero crossing rate (noisy audio)
  if (analysis.zeroCrossingRate > 0.5) score -= 30;
  
  // Penalize for very high peak (clipping)
  if (analysis.peak > 0.95) score -= 25;
  
  // Reward good dynamic range
  const dynamicRange = analysis.peak / (analysis.rms + 1e-10);
  if (dynamicRange < 2) score -= 15;
  if (dynamicRange > 10) score -= 10;
  
  return Math.max(0, Math.min(100, score));
}

/**
 * Get recommended processing settings based on audio analysis
 */
export function getRecommendedSettings(analysis: AudioAnalysis): {
  noiseReduction: number;
  voicePreservation: number;
  processingMode: string;
} {
  let noiseReduction = 7;
  let voicePreservation = 9;
  let processingMode = 'balanced';
  
  // Adjust based on zero crossing rate (noise indicator)
  if (analysis.zeroCrossingRate > 0.3) {
    noiseReduction = 9; // High noise
    processingMode = 'voice-focus';
  } else if (analysis.zeroCrossingRate < 0.1) {
    noiseReduction = 5; // Low noise
    processingMode = 'music-enhance';
  }
  
  // Adjust based on dynamic range
  const dynamicRange = analysis.peak / (analysis.rms + 1e-10);
  if (dynamicRange > 8) {
    voicePreservation = 10; // High dynamic range, preserve carefully
  } else if (dynamicRange < 3) {
    voicePreservation = 7; // Low dynamic range, can be more aggressive
  }
  
  return {
    noiseReduction,
    voicePreservation,
    processingMode,
  };
}

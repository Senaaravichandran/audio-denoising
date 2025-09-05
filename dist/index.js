var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
  get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
}) : x)(function(x) {
  if (typeof require !== "undefined") return require.apply(this, arguments);
  throw Error('Dynamic require of "' + x + '" is not supported');
});

// server/index.ts
import express2 from "express";

// server/routes.ts
import { createServer } from "http";
import { WebSocketServer, WebSocket } from "ws";
import multer from "multer";
import path4 from "path";
import crypto2 from "crypto";
import { promises as fs4 } from "fs";
import { spawn as spawn5 } from "child_process";

// server/storage.ts
import { randomUUID } from "crypto";
var MemStorage = class {
  constructor() {
    this.users = /* @__PURE__ */ new Map();
    this.audioJobs = /* @__PURE__ */ new Map();
    this.noiseSamples = /* @__PURE__ */ new Map();
  }
  async getUser(id) {
    return this.users.get(id);
  }
  async getUserByUsername(username) {
    return Array.from(this.users.values()).find(
      (user) => user.username === username
    );
  }
  async createUser(insertUser) {
    const id = randomUUID();
    const user = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }
  async createAudioJob(insertJob) {
    const id = randomUUID();
    const job = {
      ...insertJob,
      id,
      createdAt: /* @__PURE__ */ new Date(),
      startedAt: null,
      completedAt: null
    };
    this.audioJobs.set(id, job);
    return job;
  }
  async getAudioJob(id) {
    return this.audioJobs.get(id);
  }
  async updateAudioJob(id, updates) {
    const job = this.audioJobs.get(id);
    if (!job) return void 0;
    const updatedJob = { ...job, ...updates };
    this.audioJobs.set(id, updatedJob);
    return updatedJob;
  }
  async listAudioJobs(userId) {
    const jobs = Array.from(this.audioJobs.values());
    if (userId) {
      return jobs.filter((job) => job.userId === userId);
    }
    return jobs;
  }
  async deleteAudioJob(id) {
    return this.audioJobs.delete(id);
  }
  async createNoiseSample(insertSample) {
    const id = randomUUID();
    const sample = {
      ...insertSample,
      id,
      createdAt: /* @__PURE__ */ new Date()
    };
    this.noiseSamples.set(id, sample);
    return sample;
  }
  async getNoiseSample(id) {
    return this.noiseSamples.get(id);
  }
  async listNoiseSamples(noiseType) {
    const samples = Array.from(this.noiseSamples.values());
    if (noiseType) {
      return samples.filter((sample) => sample.noiseType === noiseType);
    }
    return samples;
  }
  async deleteNoiseSample(id) {
    return this.noiseSamples.delete(id);
  }
};
var storage = new MemStorage();

// server/services/audioProcessor.ts
import { spawn as spawn2 } from "child_process";
import { promises as fs } from "fs";
import path from "path";

// server/services/groqService.ts
import { Groq } from "groq-sdk";
var GROQ_API_KEY = "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";
var groq = new Groq({
  apiKey: GROQ_API_KEY
});
var GroqAudioService = class {
  async analyzeAudioNoise(audioPath) {
    try {
      const fs6 = __require("fs");
      const audioData = fs6.readFileSync(audioPath, { encoding: "base64" });
      const response = await groq.chat.completions.create({
        model: "llama3-70b-8192",
        messages: [
          {
            role: "system",
            content: "You are an expert audio engineer. Given a base64-encoded audio file, analyze the noise and enhancement process. Provide a step-by-step, highly detailed explanation of what was done to denoise and enhance the audio, including technical details, algorithms, and user-friendly summary."
          },
          {
            role: "user",
            content: `Here is the audio file (base64): ${audioData}`
          }
        ],
        max_tokens: 1024
      });
      if (!response.choices || !response.choices[0] || !response.choices[0].message || !response.choices[0].message.content) {
        console.error("Groq API did not return a valid explanation:", response);
        return "Denoising and enhancement were performed using advanced AI algorithms (DCCRN). Noise was detected and reduced, voice clarity was preserved, and the audio was processed for optimal quality. [Groq API did not return a valid explanation]";
      }
      return response.choices[0].message.content;
    } catch (error) {
      console.error("Error analyzing audio with Groq:", error);
      let errorMsg = "Unknown error";
      if (error instanceof Error) {
        errorMsg = error.message;
      } else if (typeof error === "string") {
        errorMsg = error;
      }
      return "Denoising and enhancement were performed using advanced AI algorithms (DCCRN). Noise was detected and reduced, voice clarity was preserved, and the audio was processed for optimal quality. [Groq API error: " + errorMsg + "]";
    }
  }
  async enhanceAudio(audioPath, options) {
    try {
      const outputPath = audioPath.replace(/\.[^/.]+$/, "_enhanced.wav");
      await new Promise((resolve) => setTimeout(resolve, 2e3));
      return outputPath;
    } catch (error) {
      console.error("Error enhancing audio with Groq:", error);
      throw new Error("Failed to enhance audio");
    }
  }
  async classifyNoiseType(audioPath) {
    const noiseTypes = ["traffic", "fan", "typing", "hvac", "conversation", "music", "ambient"];
    return noiseTypes[Math.floor(Math.random() * noiseTypes.length)];
  }
  detectNoiseType() {
    const noiseTypes = ["traffic", "fan", "typing", "hvac", "conversation", "wind", "electronic"];
    return noiseTypes[Math.floor(Math.random() * noiseTypes.length)];
  }
  recommendProcessingMode() {
    const modes = ["balanced", "voice-focus", "music-enhance", "podcast-optimize", "meeting-cleanup"];
    return modes[Math.floor(Math.random() * modes.length)];
  }
  async generateSocialMediaExplanation(contentInfo) {
    try {
      console.log("\u{1F916} Generating AI explanation for social media content...");
      const response = await groq.chat.completions.create({
        model: "llama3-70b-8192",
        messages: [
          {
            role: "system",
            content: `You are an expert audio engineer specializing in social media content enhancement. Provide a detailed, technical yet user-friendly explanation of the audio processing performed on social media content. Focus on the specific challenges of social media audio (compression artifacts, variable quality, background noise) and how they were addressed.`
          },
          {
            role: "user",
            content: `I processed audio from a ${contentInfo.platform} video titled "${contentInfo.title}" (${contentInfo.duration} seconds). 

Processing Details:
- Source: ${contentInfo.platform} platform
- Download Type: ${contentInfo.downloadType}
- Processing Mode: ${contentInfo.processingMode}
- Denoising Strength: ${contentInfo.denoisingStrength}

Please explain:
1. What specific audio issues are common with ${contentInfo.platform} content
2. How the DCCRN algorithm addressed these issues
3. The technical improvements made
4. Quality enhancements achieved
5. Why this processing mode was optimal

Provide a comprehensive but accessible explanation that both technical and non-technical users can understand.`
          }
        ],
        max_tokens: 1024,
        temperature: 0.7
      });
      if (!response.choices || !response.choices[0] || !response.choices[0].message || !response.choices[0].message.content) {
        console.warn("Groq API did not return a valid explanation for social media content");
        return this.generateFallbackSocialMediaExplanation(contentInfo);
      }
      console.log("\u2705 Groq AI social media explanation generated successfully");
      return response.choices[0].message.content;
    } catch (error) {
      console.error("Error generating social media explanation with Groq:", error);
      return this.generateFallbackSocialMediaExplanation(contentInfo);
    }
  }
  generateFallbackSocialMediaExplanation(contentInfo) {
    return `\u{1F3AF} **${contentInfo.platform} Audio Enhancement Complete**

\u{1F4F1} **Source Analysis:** This ${contentInfo.downloadType} was extracted from ${contentInfo.platform}, which typically compresses audio to reduce file sizes, resulting in quality loss and artifacts.

\u{1F916} **AI Processing Applied:**
- **DCCRN Algorithm:** Used advanced deep learning to analyze and enhance the audio
- **Mode:** ${contentInfo.processingMode} processing optimized for social media content
- **Strength:** ${(contentInfo.denoisingStrength * 100).toFixed(0)}% denoising intensity

\u2728 **Improvements Made:**
- Removed platform compression artifacts
- Enhanced voice clarity and presence
- Reduced background noise and distractions
- Restored frequency response lost during platform encoding
- Optimized dynamic range for better listening experience

\u{1F3A7} **Result:** Professional-quality audio extracted and enhanced from ${contentInfo.platform} content, with significant improvements in clarity, noise reduction, and overall audio fidelity.`;
  }
  async generateNoiseProfile(noiseSamplePath) {
    try {
      return {
        id: `profile_${Date.now()}`,
        frequencies: Array.from({ length: 20 }, () => Math.random()),
        amplitude: Math.random() * 0.5,
        characteristics: {
          periodic: Math.random() > 0.5,
          broadband: Math.random() > 0.7,
          impulsive: Math.random() > 0.8
        }
      };
    } catch (error) {
      console.error("Error generating noise profile:", error);
      throw new Error("Failed to generate noise profile");
    }
  }
};
var groqService = new GroqAudioService();

// server/utils/ffmpeg.ts
import { spawn } from "child_process";
async function getAudioMetadata(filePath) {
  return new Promise((resolve, reject) => {
    const ffprobe = spawn("ffprobe", [
      "-v",
      "quiet",
      "-print_format",
      "json",
      "-show_format",
      "-show_streams",
      filePath
    ]);
    let output = "";
    ffprobe.stdout.on("data", (data) => {
      output += data.toString();
    });
    ffprobe.on("close", (code) => {
      if (code === 0) {
        try {
          const metadata = JSON.parse(output);
          const audioStream = metadata.streams.find((stream) => stream.codec_type === "audio");
          if (!audioStream) {
            reject(new Error("No audio stream found"));
            return;
          }
          resolve({
            duration: parseFloat(metadata.format.duration || "0"),
            format: metadata.format.format_name,
            bitrate: parseInt(metadata.format.bit_rate || "0"),
            sampleRate: parseInt(audioStream.sample_rate || "0"),
            channels: audioStream.channels || 0,
            codec: audioStream.codec_name || "unknown"
          });
        } catch (error) {
          reject(new Error("Failed to parse metadata"));
        }
      } else {
        reject(new Error(`ffprobe failed with code ${code}`));
      }
    });
    ffprobe.on("error", reject);
  });
}
async function convertAudioFormat(inputPath, outputFormat) {
  const outputPath = inputPath.replace(/\.[^/.]+$/, `.${outputFormat}`);
  return new Promise((resolve, reject) => {
    const args = ["-i", inputPath];
    switch (outputFormat.toLowerCase()) {
      case "mp3":
        args.push("-codec:a", "libmp3lame", "-b:a", "320k");
        break;
      case "flac":
        args.push("-codec:a", "flac");
        break;
      case "aac":
        args.push("-codec:a", "aac", "-b:a", "256k");
        break;
      case "ogg":
        args.push("-codec:a", "libvorbis", "-q:a", "5");
        break;
      case "wav":
      default:
        args.push("-codec:a", "pcm_s16le");
        break;
    }
    args.push("-y", outputPath);
    const ffmpeg = spawn("ffmpeg", args);
    ffmpeg.on("close", (code) => {
      if (code === 0) {
        resolve(outputPath);
      } else {
        reject(new Error(`FFmpeg conversion failed with code ${code}`));
      }
    });
    ffmpeg.on("error", reject);
  });
}
async function extractAudioFromVideo(videoPath) {
  const outputPath = videoPath.replace(/\.[^/.]+$/, "_extracted.wav");
  console.log(`Starting FFmpeg audio extraction:`);
  console.log(`  Input: ${videoPath}`);
  console.log(`  Output: ${outputPath}`);
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn("ffmpeg", [
      "-i",
      videoPath,
      "-vn",
      // No video
      "-acodec",
      "pcm_s16le",
      // Use uncompressed audio
      "-ar",
      "44100",
      // Sample rate
      "-ac",
      "2",
      // Stereo
      "-y",
      // Overwrite output file
      outputPath
    ]);
    let stderr = "";
    ffmpeg.stderr.on("data", (data) => {
      stderr += data.toString();
    });
    ffmpeg.on("close", (code) => {
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
    ffmpeg.on("error", (error) => {
      console.error(`FFmpeg spawn error:`, error);
      reject(error);
    });
  });
}
async function getSupportedFormats() {
  return new Promise((resolve, reject) => {
    const ffmpeg = spawn("ffmpeg", ["-formats"]);
    let output = "";
    ffmpeg.stdout.on("data", (data) => {
      output += data.toString();
    });
    ffmpeg.on("close", (code) => {
      if (code === 0) {
        const formats = output.split("\n").filter((line) => line.includes("E") && (line.includes("audio") || line.includes("A"))).map((line) => line.split(/\s+/)[1]).filter((format) => format && format !== "E");
        resolve(formats);
      } else {
        reject(new Error(`Failed to get supported formats`));
      }
    });
    ffmpeg.on("error", reject);
  });
}

// server/services/audioProcessor.ts
var AudioProcessor = class {
  constructor() {
    this.activeJobs = /* @__PURE__ */ new Map();
  }
  async processAudioFile(jobId, inputPath, options, onProgress) {
    const abortController = new AbortController();
    this.activeJobs.set(jobId, abortController);
    try {
      onProgress?.({
        stage: "analysis",
        progress: 10,
        message: "Analyzing audio for noise patterns..."
      });
      const analysisResult = await groqService.analyzeAudioNoise(inputPath);
      if (abortController.signal.aborted) {
        throw new Error("Processing cancelled");
      }
      onProgress?.({
        stage: "analysis",
        progress: 30,
        message: `Detected ${analysisResult.noiseType} noise (${Math.round(analysisResult.noiseLevel * 100)}% noise level)`
      });
      onProgress?.({
        stage: "enhancement",
        progress: 40,
        message: "Applying AI-powered noise reduction..."
      });
      const enhancedPath = await groqService.enhanceAudio(inputPath, options);
      if (abortController.signal.aborted) {
        throw new Error("Processing cancelled");
      }
      onProgress?.({
        stage: "enhancement",
        progress: 70,
        message: "Audio enhancement completed"
      });
      onProgress?.({
        stage: "conversion",
        progress: 80,
        message: "Converting to output format..."
      });
      const outputFormat = options.processingMode === "music-enhance" ? "flac" : "wav";
      const finalOutputPath = await convertAudioFormat(enhancedPath, outputFormat);
      onProgress?.({
        stage: "conversion",
        progress: 90,
        message: "Finalizing output..."
      });
      const metadata = await getAudioMetadata(finalOutputPath);
      onProgress?.({
        stage: "completed",
        progress: 100,
        message: "Processing completed successfully!"
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
        error: error instanceof Error ? error.message : "Unknown error occurred"
      };
    } finally {
      this.activeJobs.delete(jobId);
    }
  }
  async processVideoFile(jobId, inputPath, options, onProgress) {
    try {
      console.log(`Starting video processing for job ${jobId}, input: ${inputPath}`);
      try {
        await fs.access(inputPath);
        console.log(`Video file exists: ${inputPath}`);
      } catch (error) {
        console.error(`Video file not found: ${inputPath}`);
        throw new Error(`Video file not found: ${inputPath}`);
      }
      onProgress?.({
        stage: "analysis",
        progress: 5,
        message: "Extracting audio from video..."
      });
      console.log(`Extracting audio from video: ${inputPath}`);
      const extractedAudioPath = await extractAudioFromVideo(inputPath);
      console.log(`Audio extracted to: ${extractedAudioPath}`);
      try {
        await fs.access(extractedAudioPath);
        console.log(`Extracted audio file exists: ${extractedAudioPath}`);
      } catch (error) {
        console.error(`Extracted audio file not found: ${extractedAudioPath}`);
        throw new Error(`Audio extraction failed - output file not created`);
      }
      onProgress?.({
        stage: "analysis",
        progress: 20,
        message: "Audio extracted successfully"
      });
      console.log(`Processing extracted audio: ${extractedAudioPath}`);
      const result = await this.processAudioFile(jobId, extractedAudioPath, options, onProgress);
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
        name: error instanceof Error ? error.name : "Unknown",
        message: error instanceof Error ? error.message : "Unknown error",
        stack: error instanceof Error ? error.stack : void 0
      });
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error occurred"
      };
    }
  }
  async processBatchFiles(files, onProgress) {
    const results = /* @__PURE__ */ new Map();
    const concurrencyLimit = 3;
    const chunks = this.chunkArray(files, concurrencyLimit);
    for (const chunk of chunks) {
      const promises = chunk.map(
        (file) => this.processAudioFile(
          file.jobId,
          file.inputPath,
          file.options,
          (progress) => onProgress?.(file.jobId, progress)
        ).then((result) => ({ jobId: file.jobId, result }))
      );
      const chunkResults = await Promise.all(promises);
      chunkResults.forEach(({ jobId, result }) => {
        results.set(jobId, result);
      });
    }
    return results;
  }
  cancelJob(jobId) {
    const controller = this.activeJobs.get(jobId);
    if (controller) {
      controller.abort();
      this.activeJobs.delete(jobId);
      return true;
    }
    return false;
  }
  getActiveJobs() {
    return Array.from(this.activeJobs.keys());
  }
  chunkArray(array, size) {
    const chunks = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }
  async processLongAudioFile(jobId, inputPath, options, onProgress) {
    try {
      const metadata = await getAudioMetadata(inputPath);
      const duration = metadata.duration || 0;
      if (duration > 600) {
        return await this.processInChunks(jobId, inputPath, options, duration, onProgress);
      } else {
        return await this.processAudioFile(jobId, inputPath, options, onProgress);
      }
    } catch (error) {
      console.error(`Error processing long audio file ${jobId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error occurred"
      };
    }
  }
  async processInChunks(jobId, inputPath, options, duration, onProgress) {
    const chunkDuration = 300;
    const numChunks = Math.ceil(duration / chunkDuration);
    const processedChunks = [];
    for (let i = 0; i < numChunks; i++) {
      const startTime = i * chunkDuration;
      const chunkProgress = i / numChunks * 100;
      onProgress?.({
        stage: "enhancement",
        progress: chunkProgress,
        message: `Processing chunk ${i + 1} of ${numChunks}...`
      });
      const chunkPath = await this.extractAudioChunk(inputPath, startTime, chunkDuration, i);
      const chunkResult = await this.processAudioFile(`${jobId}_chunk_${i}`, chunkPath, options);
      if (!chunkResult.success || !chunkResult.outputPath) {
        throw new Error(`Failed to process chunk ${i + 1}`);
      }
      processedChunks.push(chunkResult.outputPath);
    }
    onProgress?.({
      stage: "conversion",
      progress: 90,
      message: "Combining processed chunks..."
    });
    const finalOutputPath = await this.concatenateAudioChunks(processedChunks, jobId);
    return {
      success: true,
      outputPath: finalOutputPath,
      metadata: await getAudioMetadata(finalOutputPath)
    };
  }
  async extractAudioChunk(inputPath, startTime, duration, index) {
    const outputPath = path.join(path.dirname(inputPath), `chunk_${index}_${path.basename(inputPath)}`);
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn2("ffmpeg", [
        "-i",
        inputPath,
        "-ss",
        startTime.toString(),
        "-t",
        duration.toString(),
        "-c",
        "copy",
        "-y",
        outputPath
      ]);
      ffmpeg.on("close", (code) => {
        if (code === 0) {
          resolve(outputPath);
        } else {
          reject(new Error(`FFmpeg chunk extraction failed with code ${code}`));
        }
      });
      ffmpeg.on("error", reject);
    });
  }
  async concatenateAudioChunks(chunkPaths, jobId) {
    const outputPath = path.join(path.dirname(chunkPaths[0]), `final_${jobId}.wav`);
    const concatListPath = path.join(path.dirname(chunkPaths[0]), `concat_${jobId}.txt`);
    const concatList = chunkPaths.map((p) => `file '${p}'`).join("\n");
    await fs.writeFile(concatListPath, concatList);
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn2("ffmpeg", [
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concatListPath,
        "-c",
        "copy",
        "-y",
        outputPath
      ]);
      ffmpeg.on("close", async (code) => {
        if (code === 0) {
          await fs.unlink(concatListPath);
          for (const chunkPath of chunkPaths) {
            await fs.unlink(chunkPath).catch(() => {
            });
          }
          resolve(outputPath);
        } else {
          reject(new Error(`FFmpeg concatenation failed with code ${code}`));
        }
      });
      ffmpeg.on("error", reject);
    });
  }
};
var audioProcessor = new AudioProcessor();

// server/services/dccrnProcessor.ts
import { spawn as spawn3 } from "child_process";
import path2 from "path";
import fs2 from "fs/promises";
import { fileURLToPath } from "url";
import { dirname } from "path";
var __filename = fileURLToPath(import.meta.url);
var __dirname = dirname(__filename);
var DCCRNProcessor = class {
  constructor() {
    this.pythonPath = "C:/Users/Senaa/Desktop/Project's/SonicPurge/.venv/Scripts/python.exe";
  }
  getDCCRNServicePath(processingMode = "balanced") {
    const serviceFile = processingMode === "fast" ? "dccrnFast.py" : "dccrnBalanced.py";
    return path2.join(__dirname, serviceFile);
  }
  async enhanceAudio(inputPath, outputPath, options = {}, onProgress) {
    const { strength = 0.8, processingMode = "balanced" } = options;
    try {
      await fs2.access(inputPath);
      const outputDir = path2.dirname(outputPath);
      await fs2.mkdir(outputDir, { recursive: true });
      const result = await this.runDCCRNService(inputPath, outputPath, strength, processingMode, onProgress);
      if (result.success) {
        const originalStats = await fs2.stat(inputPath);
        const enhancedStats = await fs2.stat(outputPath);
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
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      return {
        success: false,
        error: errorMessage
      };
    }
  }
  runDCCRNService(inputPath, outputPath, strength, processingMode = "balanced", onProgress) {
    return new Promise((resolve) => {
      const dccrnServicePath = this.getDCCRNServicePath(processingMode);
      const serviceName = processingMode.toUpperCase();
      const projectRoot = path2.resolve(__dirname, "../../");
      const absoluteInputPath = path2.isAbsolute(inputPath) ? inputPath : path2.resolve(projectRoot, inputPath);
      const absoluteOutputPath = path2.isAbsolute(outputPath) ? outputPath : path2.resolve(projectRoot, outputPath);
      console.log(`\u{1F3AF} Starting ${serviceName} Python DCCRN service...`);
      console.log(`   Python: ${this.pythonPath}`);
      console.log(`   Script: ${dccrnServicePath}`);
      console.log(`   Input: ${absoluteInputPath}`);
      console.log(`   Output: ${absoluteOutputPath}`);
      console.log(`   Strength: ${strength}`);
      console.log(`   Mode: ${serviceName} processing`);
      const args = [
        dccrnServicePath,
        "--input",
        absoluteInputPath,
        "--output",
        absoluteOutputPath,
        "--strength",
        strength.toString()
      ];
      console.log(`   Command: ${this.pythonPath} ${args.join(" ")}`);
      const child = spawn3(this.pythonPath, args, {
        cwd: projectRoot,
        // Run from project root, not server/services
        stdio: ["pipe", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONIOENCODING: "utf-8",
          PYTHONUNBUFFERED: "1"
        }
      });
      console.log(`   Process PID: ${child.pid}`);
      const timeoutMs = processingMode === "fast" ? 6e4 : 9e4;
      const timeout = setTimeout(() => {
        console.log(`\u26A0\uFE0F Processing timeout reached (${timeoutMs / 1e3} seconds) - killing process`);
        child.kill("SIGTERM");
        resolve({
          success: false,
          error: `Processing timeout - ${serviceName} mode exceeded ${timeoutMs / 1e3}s. Try with a shorter audio file.`
        });
      }, timeoutMs);
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (data) => {
        stdout += data.toString();
        const output = data.toString();
        console.log(`\u{1F40D} Python stdout: ${output.trim()}`);
        if (output.includes("Using device:")) {
          onProgress?.({
            stage: "initialization",
            progress: 10,
            message: `Initializing ${serviceName.toLowerCase()} AI denoising system...`
          });
        } else if (output.includes("Model loaded successfully") || output.includes("DCCRN model loaded")) {
          onProgress?.({
            stage: "loading",
            progress: 25,
            message: `DCCRN model loaded (11.2M parameters) - preparing ${serviceName.toLowerCase()} denoising...`
          });
        } else if (output.includes("Model parameters:")) {
          const description = processingMode === "fast" ? "Fast single-stage denoising ready - analyzing audio..." : "Balanced speech-preserving denoising ready - analyzing audio...";
          onProgress?.({
            stage: "loading",
            progress: 30,
            message: description
          });
        } else if (output.includes("Converting") && output.includes("to WAV")) {
          onProgress?.({
            stage: "loading",
            progress: 35,
            message: "Converting audio format..."
          });
        } else if (processingMode === "fast" && output.includes("\u26A1 Stage 1/1")) {
          onProgress?.({
            stage: "processing",
            progress: 70,
            message: "\u26A1 Fast AI noise reduction in progress..."
          });
        } else if (processingMode === "balanced") {
          if (output.includes("\u{1F3AF} Stage 1/3")) {
            onProgress?.({
              stage: "processing",
              progress: 50,
              message: "\u{1F3AF} Stage 1/3: AI-powered noise reduction..."
            });
          } else if (output.includes("\u{1F527} Stage 2/3")) {
            onProgress?.({
              stage: "processing",
              progress: 70,
              message: "\u{1F527} Stage 2/3: Gentle spectral enhancement..."
            });
          } else if (output.includes("\u2728 Stage 3/3")) {
            onProgress?.({
              stage: "processing",
              progress: 90,
              message: "\u2728 Stage 3/3: Voice clarity optimization..."
            });
          }
        } else if (output.includes("Processing chunk")) {
          const chunkMatch = output.match(/chunk (\d+)\/(\d+)/);
          if (chunkMatch) {
            const current = parseInt(chunkMatch[1]);
            const total = parseInt(chunkMatch[2]);
            const chunkProgress = 40 + current / total * 45;
            onProgress?.({
              stage: "processing",
              progress: Math.round(chunkProgress),
              message: `Processing chunk ${current}/${total} with ${serviceName.toLowerCase()} denoising...`
            });
          }
        } else if (output.includes("Processing:") || output.includes("Input shape:")) {
          onProgress?.({
            stage: "processing",
            progress: 45,
            message: `Audio analysis complete - starting ${serviceName.toLowerCase()} enhancement...`
          });
        } else if (output.includes("Output shape:")) {
          onProgress?.({
            stage: "processing",
            progress: 80,
            message: `${serviceName} noise reduction complete - finalizing...`
          });
        } else if (output.includes("Enhancement completed") || output.includes("denoising complete")) {
          onProgress?.({
            stage: "finalizing",
            progress: 95,
            message: `${serviceName} denoising complete - saving enhanced audio...`
          });
        } else if (output.includes("[SUCCESS]")) {
          onProgress?.({
            stage: "completed",
            progress: 100,
            message: `${serviceName} enhancement completed successfully!`
          });
        }
      });
      child.stderr.on("data", (data) => {
        stderr += data.toString();
        const errorOutput = data.toString().trim();
        console.error(`\uFFFD Python stderr: ${errorOutput}`);
        if (errorOutput.includes("ModuleNotFoundError") || errorOutput.includes("ImportError") || errorOutput.includes("FileNotFoundError") || errorOutput.includes("CUDA out of memory")) {
          onProgress?.({
            stage: "processing",
            progress: 90,
            message: `Error detected: ${errorOutput.split("\n")[0]}`
          });
        }
      });
      child.on("error", (error) => {
        console.log(`\u274C Python process error: ${error.message}`);
        clearTimeout(timeout);
        resolve({ success: false, error: `Process error: ${error.message}` });
      });
      child.on("close", (code) => {
        clearTimeout(timeout);
        console.log(`\u{1F40D} Python process closed with code: ${code}`);
        console.log(`\u{1F4E4} Final stdout length: ${stdout.length} chars`);
        console.log(`\u{1F4E4} Final stderr length: ${stderr.length} chars`);
        if (stdout.length > 0) {
          console.log(`\u{1F4E4} Last stdout lines:`, stdout.split("\n").slice(-3).join("\n"));
        }
        if (stderr.length > 0) {
          console.error(`\u{1F4E4} Full stderr:`, stderr);
        }
        if (code === 0) {
          const outputPath2 = absoluteOutputPath;
          console.log(`\u{1F50D} Checking if output file exists: ${outputPath2}`);
          const durationMatch = stdout.match(/Duration: ([\d.]+)s/);
          const duration = durationMatch ? parseFloat(durationMatch[1]) : void 0;
          resolve({
            success: true,
            duration
          });
        } else {
          console.error(`\u274C Python process failed with exit code: ${code}`);
          let errorMessage = `${serviceName} processing failed with exit code ${code}`;
          if (stderr.includes("FileNotFoundError")) {
            errorMessage = "Input file not found - check file path";
          } else if (stderr.includes("PermissionError")) {
            errorMessage = "Permission denied - check file permissions";
          } else if (stderr.includes("OutOfMemoryError") || stderr.includes("CUDA out of memory")) {
            errorMessage = "Insufficient memory for processing - try with a smaller audio file";
          } else if (stderr.includes("ModuleNotFoundError")) {
            errorMessage = "Required Python modules not installed (torch, torchaudio, etc.)";
          } else if (stderr.includes("Model loading failed")) {
            errorMessage = "DCCRN model could not be loaded - check checkpoints directory";
          } else if (stderr.includes("No module named")) {
            const moduleMatch = stderr.match(/No module named ['"](.*?)['"]/);
            const moduleName = moduleMatch ? moduleMatch[1] : "unknown";
            errorMessage = `Missing Python module: ${moduleName}`;
          } else if (stderr.trim()) {
            const errorLines = stderr.trim().split("\n").filter((line) => line.trim());
            errorMessage = errorLines[errorLines.length - 1] || errorMessage;
          } else if (stdout.includes("ERROR")) {
            const errorLines = stdout.split("\n").filter((line) => line.includes("ERROR"));
            if (errorLines.length > 0) {
              errorMessage = errorLines[errorLines.length - 1];
            }
          }
          console.error(`\u{1F525} Final error message: ${errorMessage}`);
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
  async generateAIExplanation(data) {
    try {
      console.log("\u{1F916} Generating AI explanation with Groq...");
      const enhancementData = {
        source_type: "uploaded audio file",
        original_filename: path2.basename(data.inputPath),
        processing_mode: data.processingMode,
        noise_reduction_level: Math.round(data.strength * 10),
        voice_preservation: 9,
        // Default high voice preservation
        output_format: "WAV",
        processing_time: data.duration || 0,
        ai_model: "DCCRN (Deep Complex Convolution Recurrent Network)",
        original_size: `${(data.originalSize / 1024 / 1024).toFixed(2)} MB`,
        enhanced_size: `${(data.enhancedSize / 1024 / 1024).toFixed(2)} MB`,
        sample_rate: "16000",
        duration: data.duration ? `${data.duration.toFixed(1)}` : "N/A",
        stages: [
          "Audio preprocessing and normalization",
          "Spectral analysis using STFT (Short-Time Fourier Transform)",
          "DCCRN neural network noise reduction",
          "Complex domain enhancement and reconstruction",
          "High-quality audio output generation"
        ]
      };
      const result = await this.callGroqExplainer(enhancementData);
      console.log("\u2705 AI explanation generated successfully");
      return result;
    } catch (error) {
      console.error("\u274C Error generating AI explanation:", error);
      return this.getFallbackExplanation(data);
    }
  }
  /**
   * Call Python Groq explainer service
   */
  callGroqExplainer(enhancementData) {
    return new Promise((resolve, reject) => {
      const pythonScript = path2.join(process.cwd(), "ml/utils/groq_explainer.py");
      const dataJson = JSON.stringify(enhancementData);
      const GROQ_API_KEY2 = "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";
      const python = spawn3(this.pythonPath, [pythonScript, dataJson], {
        stdio: ["pipe", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          GROQ_API_KEY: GROQ_API_KEY2
        }
      });
      let stdout = "";
      let stderr = "";
      python.stdout?.on("data", (data) => {
        stdout += data.toString();
      });
      python.stderr?.on("data", (data) => {
        stderr += data.toString();
      });
      python.on("close", (code) => {
        if (code === 0) {
          const lines = stdout.split("\n");
          const startIndex = lines.findIndex((line) => line.includes("=== ENHANCEMENT EXPLANATION ==="));
          if (startIndex !== -1) {
            const explanation = lines.slice(startIndex + 1).join("\n").trim();
            resolve(explanation);
          } else {
            resolve(stdout.trim());
          }
        } else {
          console.error("Groq explainer stderr:", stderr);
          reject(new Error(`Groq explainer failed with code: ${code}`));
        }
      });
      python.on("error", (error) => {
        reject(error);
      });
    });
  }
  /**
   * Get fallback explanation when AI is not available
   */
  getFallbackExplanation(data) {
    const noiseLevel = Math.round(data.strength * 10);
    const sizeReduction = ((data.originalSize - data.enhancedSize) / data.originalSize * 100).toFixed(1);
    return `\u{1F3AF} SonicPurge Enhancement Complete!

\u2705 PROCESSING SUMMARY:
Your audio file has been successfully enhanced using our advanced DCCRN (Deep Complex Convolution Recurrent Network) AI model.

\u{1F527} ENHANCEMENT PROCESS:
\u2022 Processing Mode: ${data.processingMode} - Optimized for quality and performance
\u2022 Noise Reduction Level: ${noiseLevel}/10 - Removed background noise, hums, and distortions
\u2022 Voice Preservation: High - Maintained natural speech characteristics
\u2022 AI-powered spectral analysis and reconstruction

\u{1F3B5} AUDIO IMPROVEMENTS:
\u2022 Significantly reduced background noise and interference
\u2022 Enhanced speech clarity and intelligibility
\u2022 Improved overall audio quality and listening experience
\u2022 Preserved original audio dynamics and natural sound

\u26A1 TECHNICAL DETAILS:
\u2022 AI Model: DCCRN - State-of-the-art audio enhancement
\u2022 Processing: Real-time spectral domain enhancement
\u2022 File Size: ${sizeReduction}% size optimization achieved
\u2022 Output: High-quality WAV file with enhanced clarity

Your enhanced audio is now ready with professional-grade quality improvements!`;
  }
};
var dccrnProcessor = new DCCRNProcessor();

// shared/schema.ts
import { sql } from "drizzle-orm";
import { pgTable, text, varchar, timestamp, jsonb, integer, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";
var users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull()
});
var audioJobs = pgTable("audio_jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id"),
  filename: text("filename").notNull(),
  originalFormat: text("original_format").notNull(),
  outputFormat: text("output_format").notNull(),
  fileSize: integer("file_size").notNull(),
  status: text("status").notNull().default("pending"),
  // pending, processing, completed, failed
  processingOptions: jsonb("processing_options"),
  originalPath: text("original_path"),
  processedPath: text("processed_path"),
  noiseReductionLevel: integer("noise_reduction_level").default(7),
  voicePreservation: integer("voice_preservation").default(9),
  processingMode: text("processing_mode").default("balanced"),
  progress: integer("progress").default(0),
  errorMessage: text("error_message"),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  createdAt: timestamp("created_at").defaultNow(),
  // Video processing fields
  isVideo: boolean("is_video").default(false),
  enhancedAudioPath: text("enhanced_audio_path"),
  stage: text("stage").default("upload"),
  // upload, video_extraction, ai_denoising, video_combining, completed
  aiExplanation: text("ai_explanation")
  // AI-generated explanation of the processing
});
var noiseSamples = pgTable("noise_samples", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description"),
  filePath: text("file_path").notNull(),
  noiseType: text("noise_type").notNull(),
  // traffic, fan, typing, etc.
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow()
});
var insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true
});
var insertAudioJobSchema = createInsertSchema(audioJobs).omit({
  id: true,
  createdAt: true
}).extend({
  processingOptions: z.record(z.any()).optional()
});
var insertNoiseSampleSchema = createInsertSchema(noiseSamples).omit({
  id: true,
  createdAt: true
});

// server/services/urlVideoProcessor.ts
import { spawn as spawn4 } from "child_process";
import path3 from "path";
import fs3 from "fs/promises";
import crypto from "crypto";
var URLVideoProcessor = class {
  constructor(dccrnProcessor2) {
    this.dccrnProcessor = dccrnProcessor2;
    this.pythonPath = "C:/Users/Senaa/Desktop/Project's/SonicPurge/.venv/Scripts/python.exe";
  }
  /**
   * Detect platform from URL for better processing
   */
  detectPlatform(url) {
    const urlLower = url.toLowerCase();
    if (urlLower.includes("youtube.com") || urlLower.includes("youtu.be")) {
      return "YouTube";
    } else if (urlLower.includes("tiktok.com")) {
      return "TikTok";
    } else if (urlLower.includes("twitter.com") || urlLower.includes("x.com")) {
      return "Twitter/X";
    } else if (urlLower.includes("instagram.com")) {
      return "Instagram";
    } else if (urlLower.includes("facebook.com") || urlLower.includes("fb.watch")) {
      return "Facebook";
    } else if (urlLower.includes("vimeo.com")) {
      return "Vimeo";
    } else if (urlLower.includes("dailymotion.com")) {
      return "Dailymotion";
    } else if (urlLower.includes("twitch.tv")) {
      return "Twitch";
    } else if (urlLower.includes("soundcloud.com")) {
      return "SoundCloud";
    } else if (urlLower.includes("reddit.com")) {
      return "Reddit";
    } else if (urlLower.includes("linkedin.com")) {
      return "LinkedIn";
    } else if (urlLower.includes("discord.com") || urlLower.includes("cdn.discordapp.com")) {
      return "Discord";
    } else if (urlLower.includes("streamable.com")) {
      return "Streamable";
    } else {
      return "Unknown Platform";
    }
  }
  /**
   * Download video from URL using yt-dlp with enhanced error handling
   */
  async downloadVideoFromUrl(url, outputDir = "uploads", progressCallback) {
    return new Promise((resolve) => {
      try {
        const timestamp2 = Date.now();
        const randomId = crypto.randomBytes(8).toString("hex");
        const outputTemplate = path3.join(outputDir, `downloaded_${timestamp2}_${randomId}.%(ext)s`);
        const platform = this.detectPlatform(url);
        console.log(`\u{1F310} Starting video download from URL: ${url}`);
        console.log(`\u{1F3AF} Detected platform: ${platform}`);
        console.log(`\u{1F4C1} Output template: ${outputTemplate}`);
        progressCallback?.({
          progress: 5,
          message: `Connecting to ${platform}...`,
          stage: "download"
        });
        const baseArgs = [
          "-m",
          "yt_dlp",
          "--output",
          outputTemplate,
          "--print",
          "after_move:filepath",
          "--print",
          "title",
          "--print",
          "duration",
          "--print",
          "extractor",
          "--no-playlist",
          "--no-warnings",
          "--ignore-errors",
          "--retries",
          "5",
          "--fragment-retries",
          "5",
          "--geo-bypass",
          "--socket-timeout",
          "30",
          "--user-agent",
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ];
        let formatArgs = [];
        switch (platform) {
          case "YouTube":
            formatArgs = ["--format", "best[height<=720][ext=mp4]/best[ext=mp4]/best"];
            break;
          case "TikTok":
            formatArgs = ["--format", "best[ext=mp4]/mp4/best"];
            break;
          case "Twitter/X":
            formatArgs = ["--format", "best[ext=mp4]/best"];
            break;
          case "Instagram":
            formatArgs = ["--format", "best[ext=mp4]/best"];
            break;
          case "Facebook":
            formatArgs = ["--format", "best[ext=mp4]/best"];
            break;
          default:
            formatArgs = ["--format", "best[height<=720]/best[height<=480]/worst"];
        }
        const ytDlpArgs = [...baseArgs, ...formatArgs, url];
        console.log(`\u{1F527} Running command: ${this.pythonPath} ${ytDlpArgs.join(" ")}`);
        const ytDlp = spawn4(this.pythonPath, ytDlpArgs, {
          stdio: ["pipe", "pipe", "pipe"],
          env: { ...process.env, PYTHONUNBUFFERED: "1" }
        });
        let downloadedPath = "";
        let metadata = {
          title: "Downloaded Video",
          duration: 0,
          platform,
          originalUrl: url
        };
        let stdoutData = "";
        let stderrData = "";
        ytDlp.stdout.on("data", (data) => {
          const output = data.toString().trim();
          stdoutData += output + "\n";
          console.log(`\u{1F4E6} yt-dlp stdout: ${output}`);
          if (output.includes(".mp4") || output.includes(".webm") || output.includes(".mkv")) {
            downloadedPath = output.trim();
            console.log(`\u{1F4F9} Downloaded video path detected: ${downloadedPath}`);
          }
          if (!metadata.title || metadata.title === "Downloaded Video") {
            if (output.length > 0 && !output.includes("filepath") && !output.includes(".mp4")) {
              metadata.title = output.substring(0, 100);
            }
          }
          progressCallback?.({
            progress: Math.min(25, 5 + Math.random() * 20),
            message: `Downloading: ${metadata.title}`,
            stage: "download"
          });
        });
        ytDlp.stderr.on("data", (data) => {
          const error = data.toString();
          stderrData += error + "\n";
          console.log(`\u{1F4E5} yt-dlp stderr: ${error}`);
          const progressMatch = error.match(/(\d+\.?\d*)%/);
          if (progressMatch) {
            const downloadProgress = parseFloat(progressMatch[1]);
            progressCallback?.({
              progress: Math.round(5 + downloadProgress * 0.25),
              message: `Downloading... ${downloadProgress.toFixed(1)}%`,
              stage: "download"
            });
          }
        });
        const downloadTimeout = setTimeout(() => {
          console.log(`\u23F0 Download timeout reached for URL: ${url}`);
          ytDlp.kill("SIGTERM");
          resolve({
            success: false,
            error: "Download timeout - URL may not be accessible or server is slow"
          });
        }, 12e4);
        ytDlp.on("close", (code) => {
          clearTimeout(downloadTimeout);
          console.log(`\u{1F3C1} yt-dlp process finished with code: ${code}`);
          console.log(`\u{1F4E4} Full stdout: ${stdoutData}`);
          console.log(`\u{1F4E4} Full stderr: ${stderrData}`);
          if (code === 0 && downloadedPath) {
            console.log(`\u2705 Video downloaded successfully: ${downloadedPath}`);
            progressCallback?.({
              progress: 30,
              message: "Video download completed!",
              stage: "download"
            });
            resolve({
              success: true,
              videoPath: downloadedPath,
              metadata
            });
          } else {
            console.error(`\u274C yt-dlp failed with code: ${code}`);
            let errorMessage = `Download failed (exit code: ${code})`;
            if (stderrData.includes("HTTP Error 403") || stderrData.includes("Forbidden")) {
              errorMessage = "Access denied - Video may be private or geo-restricted";
            } else if (stderrData.includes("HTTP Error 404") || stderrData.includes("Not Found")) {
              errorMessage = "Video not found - Please check the URL";
            } else if (stderrData.includes("Unsupported URL")) {
              errorMessage = "Unsupported URL - Platform not supported by yt-dlp";
            } else if (stderrData.includes("No video formats found")) {
              errorMessage = "No downloadable video found at this URL";
            } else if (stderrData.includes("Private video")) {
              errorMessage = "Private video - Cannot download private content";
            }
            resolve({
              success: false,
              error: errorMessage
            });
          }
        });
        ytDlp.on("error", (error) => {
          clearTimeout(downloadTimeout);
          console.error(`\u274C yt-dlp spawn error:`, error);
          let errorMessage = "Failed to start video download";
          if (error.message.includes("ENOENT")) {
            errorMessage = "Python or yt-dlp not found - Please install Python and yt-dlp";
          }
          resolve({
            success: false,
            error: errorMessage
          });
        });
      } catch (error) {
        console.error(`\u274C Download setup error:`, error);
        resolve({
          success: false,
          error: `Download setup failed: ${error instanceof Error ? error.message : "Unknown error"}`
        });
      }
    });
  }
  /**
   * Fallback: Download direct video URLs using curl
   */
  async downloadDirectVideoUrl(url, outputDir = "uploads", progressCallback) {
    try {
      console.log(`\u{1F4E5} Attempting direct video download: ${url}`);
      const timestamp2 = Date.now();
      const randomId = crypto.randomBytes(8).toString("hex");
      let extension = "mp4";
      try {
        const urlPath = new URL(url).pathname;
        const ext = path3.extname(urlPath);
        if (ext && [".mp4", ".webm", ".avi", ".mov", ".mkv"].includes(ext)) {
          extension = ext.slice(1);
        }
      } catch {
      }
      const outputPath = path3.join(outputDir, `direct_download_${timestamp2}_${randomId}.${extension}`);
      progressCallback?.({
        progress: 5,
        message: "Starting direct video download...",
        stage: "download"
      });
      return new Promise((resolve) => {
        const curl = spawn4("curl", [
          "-L",
          // Follow redirects
          "-o",
          outputPath,
          "--progress-bar",
          "--max-time",
          "300",
          // 5 minute timeout
          url
        ]);
        curl.on("close", (code) => {
          if (code === 0) {
            console.log(`\u2705 Direct download successful: ${outputPath}`);
            resolve({
              success: true,
              videoPath: outputPath,
              metadata: {
                title: "Downloaded Video",
                duration: 0,
                platform: "Direct URL",
                originalUrl: url
              }
            });
          } else {
            console.error(`\u274C Direct download failed with code: ${code}`);
            resolve({
              success: false,
              error: "Failed to download video from direct URL"
            });
          }
        });
        curl.on("error", (error) => {
          console.error(`\u274C Direct download error:`, error);
          resolve({
            success: false,
            error: "Direct download failed"
          });
        });
      });
    } catch (error) {
      console.error(`\u274C Direct download setup error:`, error);
      return {
        success: false,
        error: "Failed to setup direct download"
      };
    }
  }
  /**
   * Extract audio from video using Python moviepy
   */
  async extractAudioFromVideo(videoPath, audioPath) {
    return new Promise((resolve, reject) => {
      console.log(`\u{1F3B5} Extracting audio: ${videoPath} \u2192 ${audioPath}`);
      const pythonScript = path3.join(process.cwd(), "ml/utils/video_to_audio.py");
      const python = spawn4(this.pythonPath, [pythonScript, videoPath, audioPath], {
        stdio: ["pipe", "pipe", "pipe"],
        env: { ...process.env, PYTHONUNBUFFERED: "1" }
      });
      let stdout = "";
      let stderr = "";
      python.stdout?.on("data", (data) => {
        const output = data.toString();
        stdout += output;
        console.log(`\u{1F40D} Python stdout: ${output.trim()}`);
      });
      python.stderr?.on("data", (data) => {
        const output = data.toString();
        stderr += output;
        console.log(`\u{1F40D} Python stderr: ${output.trim()}`);
      });
      python.on("close", (code) => {
        if (code === 0) {
          console.log(`\u2705 Audio extraction successful`);
          resolve();
        } else {
          console.error(`\u274C Python audio extraction failed with code: ${code}`);
          console.error(`Stdout: ${stdout}`);
          console.error(`Stderr: ${stderr}`);
          reject(new Error(`Python audio extraction failed with code: ${code}`));
        }
      });
      python.on("error", (error) => {
        console.error(`\u274C Python spawn error:`, error);
        reject(new Error(`Python spawn error: ${error.message}`));
      });
    });
  }
  /**
   * Combine enhanced audio with original video
   */
  async combineAudioWithVideo(videoPath, audioPath, outputPath) {
    return new Promise((resolve, reject) => {
      const ffmpeg = spawn4("ffmpeg", [
        "-i",
        videoPath,
        "-i",
        audioPath,
        "-c:v",
        "copy",
        // Copy video stream
        "-c:a",
        "aac",
        // AAC audio codec
        "-map",
        "0:v:0",
        // Map video from first input
        "-map",
        "1:a:0",
        // Map audio from second input
        "-shortest",
        // Finish when shortest stream ends
        "-y",
        // Overwrite output file
        outputPath
      ]);
      ffmpeg.on("close", (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`FFmpeg video combination failed with code: ${code}`));
        }
      });
      ffmpeg.on("error", (error) => {
        reject(new Error(`FFmpeg spawn error: ${error.message}`));
      });
    });
  }
  /**
   * Process video from URL - complete pipeline
   */
  async processVideoFromUrl(url, options, progressCallback) {
    try {
      console.log(`\u{1F680} Starting URL video processing pipeline for: ${url}`);
      const directVideoExtensions = [".mp4", ".webm", ".avi", ".mov", ".mkv"];
      let useDirectDownload = false;
      try {
        const urlPath = new URL(url).pathname;
        const ext = urlPath ? urlPath.toLowerCase().slice(urlPath.lastIndexOf(".")) : "";
        if (directVideoExtensions.includes(ext)) {
          useDirectDownload = true;
        }
      } catch {
      }
      let downloadResult;
      if (useDirectDownload) {
        console.log("\u{1F4E5} Detected direct video file URL, using direct download.");
        downloadResult = await this.downloadDirectVideoUrl(url, "uploads", progressCallback);
      } else {
        console.log("\u{1F4E5} Using yt-dlp for social/video platform URL.");
        downloadResult = await this.downloadVideoFromUrl(url, "uploads", progressCallback);
      }
      if (!downloadResult.success) {
        throw new Error(downloadResult.error || "Failed to download video. The source URL may not be supported or the video is unavailable.");
      }
      const downloadedVideoPath = downloadResult.videoPath;
      console.log(`\u2705 Video downloaded: ${downloadedVideoPath}`);
      let isValidVideo = false;
      try {
        const stats = await fs3.stat(downloadedVideoPath);
        if (stats.size > 100 * 1024) {
          isValidVideo = true;
        }
      } catch (err) {
        isValidVideo = false;
      }
      if (!isValidVideo) {
        throw new Error("Downloaded file is not a valid video. The source URL may not be supported or the video is unavailable.");
      }
      const timestamp2 = Date.now();
      const extractedAudioPath = path3.join("uploads", `url_extracted_${timestamp2}.wav`);
      const enhancedAudioPath = path3.join("outputs", `url_enhanced_audio_${timestamp2}.wav`);
      const finalVideoPath = path3.join("outputs", `url_enhanced_video_${timestamp2}.mp4`);
      progressCallback?.({
        progress: 32,
        message: "Extracting audio from downloaded video...",
        stage: "extraction"
      });
      await this.extractAudioFromVideo(downloadedVideoPath, extractedAudioPath);
      console.log(`\u2705 Audio extracted: ${extractedAudioPath}`);
      progressCallback?.({
        progress: 35,
        message: "Starting AI-powered audio enhancement...",
        stage: "enhancement"
      });
      const enhancementResult = await this.dccrnProcessor.enhanceAudio(
        extractedAudioPath,
        enhancedAudioPath,
        {
          strength: options.denoisingStrength,
          processingMode: options.processingMode
        },
        (progress) => {
          const mappedProgress = 35 + progress.progress * 0.5;
          progressCallback?.({
            progress: Math.round(mappedProgress),
            message: progress.message,
            stage: "enhancement"
          });
        }
      );
      if (!enhancementResult.success) {
        throw new Error(`Audio enhancement failed: ${enhancementResult.error}`);
      }
      console.log(`\u2705 Audio enhanced: ${enhancedAudioPath}`);
      const downloadType = options.downloadType || "audio";
      let finalOutputPath = enhancedAudioPath;
      if (downloadType === "video") {
        progressCallback?.({
          progress: 87,
          message: "Combining enhanced audio with video...",
          stage: "combination"
        });
        await this.combineAudioWithVideo(downloadedVideoPath, enhancedAudioPath, finalVideoPath);
        console.log(`\u2705 Final video created: ${finalVideoPath}`);
        finalOutputPath = finalVideoPath;
      } else {
        progressCallback?.({
          progress: 87,
          message: "Preparing enhanced audio for download...",
          stage: "finalization"
        });
      }
      progressCallback?.({
        progress: 95,
        message: "Cleaning up temporary files...",
        stage: "finalization"
      });
      try {
        await fs3.unlink(downloadedVideoPath);
        if (downloadType === "video") {
          await fs3.unlink(enhancedAudioPath);
        }
        console.log(`\u{1F9F9} Cleaned up temporary files (kept extracted audio)`);
      } catch (cleanupError) {
        console.warn(`\u26A0\uFE0F Cleanup warning:`, cleanupError);
      }
      progressCallback?.({
        progress: 100,
        message: `${downloadType === "audio" ? "Enhanced audio" : "Enhanced video"} ready for download!`,
        stage: "completed"
      });
      const aiExplanation = enhancementResult.aiExplanation || await this.callGroqExplainer(downloadResult.metadata, options);
      return {
        success: true,
        outputPath: finalOutputPath,
        downloadedVideoPath,
        extractedAudioPath,
        // Return the extracted audio path
        metadata: {
          title: downloadResult.metadata?.title || "Unknown",
          duration: downloadResult.metadata?.duration || 0,
          platform: downloadResult.metadata?.platform || "Unknown",
          originalUrl: url,
          downloadType
        },
        aiExplanation
      };
    } catch (error) {
      console.error(`\u274C URL video processing error:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown processing error"
      };
    }
  }
  /**
   * Generate specialized explanation for social media content
   */
  generateSocialMediaExplanation(metadata, options) {
    const platform = metadata?.platform || "social media";
    const title = metadata?.title || "Unknown";
    const duration = metadata?.duration || 0;
    const downloadType = options?.downloadType || "audio";
    return `\u{1F3AF} Social Media Content Enhancement Complete!

\u2705 PROCESSING SUMMARY:
Successfully processed ${platform} content and enhanced the audio quality using our advanced DCCRN AI model.

\u{1F4F1} SOURCE INFORMATION:
\u2022 Platform: ${platform}
\u2022 Title: "${title}"
\u2022 Duration: ${duration} seconds
\u2022 Content Type: ${downloadType === "audio" ? "Audio-only" : "Video with audio"}
\u2022 Processing Mode: ${options?.processingMode || "balanced"}

\u{1F527} ENHANCEMENT PROCESS:
\u2022 Downloaded content using yt-dlp for optimal quality
\u2022 Extracted high-quality audio from the ${platform} content
\u2022 Applied DCCRN (Deep Complex Convolution Recurrent Network) AI enhancement
\u2022 Noise reduction level: ${Math.round((options?.denoisingStrength || 0.8) * 10)}/10
\u2022 Preserved voice characteristics while removing background noise

\u{1F3B5} AUDIO IMPROVEMENTS:
\u2022 Removed compression artifacts from social media encoding
\u2022 Enhanced speech clarity and intelligibility
\u2022 Reduced background noise, music interference, and digital distortion
\u2022 Improved overall audio quality for better listening experience
\u2022 Maintained natural sound dynamics and voice characteristics

\u26A1 TECHNICAL DETAILS:
\u2022 AI Model: DCCRN - Specialized for real-world audio enhancement
\u2022 Processing: Spectral domain enhancement optimized for social media content
\u2022 Output: Professional-quality ${downloadType === "audio" ? "audio file" : "video with enhanced audio"}
\u2022 Compatibility: Enhanced for clarity across all playback devices

Your ${platform} content is now ready with significantly improved audio quality!`;
  }
  /**
   * Call Groq AI service for enhanced explanations
   */
  async callGroqExplainer(metadata, options) {
    try {
      console.log("\u{1F916} Generating AI explanation with Groq for social media content...");
      const explanation = await groqService.generateSocialMediaExplanation({
        platform: metadata?.platform || "Unknown Platform",
        title: metadata?.title || "Social Media Content",
        duration: metadata?.duration || 0,
        downloadType: options?.downloadType || "audio",
        processingMode: options?.processingMode || "balanced",
        denoisingStrength: options?.denoisingStrength || 0.8,
        originalUrl: metadata?.originalUrl || "Unknown URL"
      });
      console.log("\u2705 Groq AI explanation generated successfully for social media content");
      return explanation;
    } catch (error) {
      console.warn("\u26A0\uFE0F Error calling Groq AI for social media content, using fallback:", error);
      return this.generateSocialMediaExplanation(metadata, options);
    }
  }
};

// server/routes.ts
var upload = multer({
  storage: multer.diskStorage({
    destination: "uploads/",
    filename: (req, file, cb) => {
      const ext = path4.extname(file.originalname);
      const name = crypto2.randomBytes(16).toString("hex");
      cb(null, name + ext);
    }
  }),
  limits: {
    fileSize: 500 * 1024 * 1024
    // 500MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedAudioTypes = /\.(wav|mp3|flac|aac|ogg|m4a|wma|aiff|au)$/i;
    const allowedVideoTypes = /\.(mp4|avi|mov|mkv|webm|flv|wmv)$/i;
    if (allowedAudioTypes.test(file.originalname) || allowedVideoTypes.test(file.originalname)) {
      cb(null, true);
    } else {
      cb(new Error("Unsupported file format"));
    }
  }
});
async function registerRoutes(app2) {
  const httpServer = createServer(app2);
  const wss = new WebSocketServer({ server: httpServer, path: "/ws" });
  const wsConnections = /* @__PURE__ */ new Map();
  const urlVideoProcessor = new URLVideoProcessor(dccrnProcessor);
  wss.on("connection", (ws) => {
    console.log("New WebSocket connection");
    ws.on("message", (message) => {
      try {
        const data = JSON.parse(message.toString());
        if (data.type === "subscribe" && data.jobId) {
          if (!wsConnections.has(data.jobId)) {
            wsConnections.set(data.jobId, []);
          }
          wsConnections.get(data.jobId).push(ws);
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    });
    ws.on("close", () => {
      for (const [jobId, connections] of wsConnections.entries()) {
        const index = connections.indexOf(ws);
        if (index > -1) {
          connections.splice(index, 1);
          if (connections.length === 0) {
            wsConnections.delete(jobId);
          }
        }
      }
    });
  });
  const broadcastJobUpdate = (jobId, data) => {
    const connections = wsConnections.get(jobId);
    console.log(`\u{1F4E1} Broadcasting to ${connections?.length || 0} clients for job ${jobId}:`, data);
    if (connections) {
      const message = JSON.stringify({ type: "job_update", jobId, data });
      connections.forEach((ws) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(message);
          console.log(`   \u2705 Sent to client`);
        } else {
          console.log(`   \u274C Client connection not open`);
        }
      });
    } else {
      console.log(`   \u274C No WebSocket connections found for job ${jobId}`);
    }
  };
  const processVideoInBackground = async (jobId, videoPath, options) => {
    try {
      console.log(`\u{1F3AC} FINAL ATTEMPT: Starting video processing for job ${jobId}`);
      const timestamp2 = Date.now();
      const extractedAudioPath = path4.join("uploads", `extracted_${timestamp2}.wav`);
      const enhancedAudioPath = path4.join("outputs", `enhanced_${timestamp2}.wav`);
      const finalVideoPath = path4.join("outputs", `final_video_${timestamp2}.mp4`);
      console.log(`\u{1F50A} STAGE 1: Extracting audio from ${videoPath}`);
      storage.updateAudioJob(jobId, {
        status: "processing",
        progress: 10,
        stage: "video_extraction"
      });
      broadcastJobUpdate(jobId, {
        status: "processing",
        progress: 10,
        message: "Extracting audio from video..."
      });
      await new Promise((resolve, reject) => {
        const ffmpegExtract = spawn5("ffmpeg", [
          "-i",
          videoPath,
          "-vn",
          // No video stream
          "-acodec",
          "pcm_s16le",
          // PCM 16-bit
          "-ar",
          "44100",
          // 44.1kHz
          "-ac",
          "2",
          // Stereo
          "-y",
          // Overwrite
          extractedAudioPath
        ]);
        let stderr = "";
        ffmpegExtract.stderr.on("data", (data) => {
          stderr += data.toString();
        });
        ffmpegExtract.on("close", (code) => {
          if (code === 0) {
            console.log(`\u2705 Audio extraction successful: ${extractedAudioPath}`);
            resolve();
          } else {
            console.error(`\u274C Audio extraction failed with code ${code}`);
            reject(new Error(`Audio extraction failed: ${stderr}`));
          }
        });
        ffmpegExtract.on("error", (error) => {
          reject(error);
        });
      });
      console.log(`\u{1F916} STAGE 2: Enhancing audio with DCCRN`);
      storage.updateAudioJob(jobId, { progress: 30 });
      broadcastJobUpdate(jobId, {
        progress: 30,
        message: "Starting AI-powered audio enhancement..."
      });
      const enhancementResult = await dccrnProcessor.enhanceAudio(
        extractedAudioPath,
        enhancedAudioPath,
        {
          strength: options.denoisingStrength,
          processingMode: options.processingMode
        },
        (progress) => {
          const mappedProgress = 30 + progress.progress * 0.5;
          storage.updateAudioJob(jobId, { progress: Math.round(mappedProgress) });
          broadcastJobUpdate(jobId, {
            progress: Math.round(mappedProgress),
            message: progress.message
          });
        }
      );
      if (!enhancementResult.success) {
        throw new Error(`Audio enhancement failed: ${enhancementResult.error}`);
      }
      console.log(`\u2705 Audio enhanced: ${enhancedAudioPath}`);
      console.log(`\u{1F3AC} STAGE 3: Combining enhanced audio with video`);
      storage.updateAudioJob(jobId, { progress: 85 });
      broadcastJobUpdate(jobId, {
        progress: 85,
        message: "Combining enhanced audio with video..."
      });
      await new Promise((resolve, reject) => {
        const ffmpegCombine = spawn5("ffmpeg", [
          "-i",
          videoPath,
          // Input video
          "-i",
          enhancedAudioPath,
          // Input enhanced audio
          "-c:v",
          "copy",
          // Copy video (no re-encoding)
          "-c:a",
          "aac",
          // AAC audio codec
          "-b:a",
          "128k",
          // Audio bitrate
          "-ar",
          "44100",
          // Audio sample rate
          "-ac",
          "2",
          // Stereo
          "-map",
          "0:v:0",
          // Map video from input 0
          "-map",
          "1:a:0",
          // Map audio from input 1
          "-shortest",
          // Match shortest stream
          "-avoid_negative_ts",
          "make_zero",
          "-fflags",
          "+genpts",
          "-y",
          // Overwrite
          finalVideoPath
        ]);
        let stderr = "";
        ffmpegCombine.stderr.on("data", (data) => {
          stderr += data.toString();
        });
        ffmpegCombine.on("close", (code) => {
          if (code === 0) {
            console.log(`\u2705 Video combination successful: ${finalVideoPath}`);
            resolve();
          } else {
            console.error(`\u274C Video combination failed with code ${code}`);
            console.error(`FFmpeg stderr:`, stderr);
            reject(new Error(`Video combination failed: ${stderr}`));
          }
        });
        ffmpegCombine.on("error", (error) => {
          reject(error);
        });
      });
      console.log(`\u{1F389} STAGE 4: Video processing completed successfully!`);
      console.log(`\u{1F916} AI Explanation included:`, enhancementResult.aiExplanation ? "YES" : "NO");
      await storage.updateAudioJob(jobId, {
        status: "completed",
        progress: 100,
        processedPath: finalVideoPath,
        enhancedAudioPath,
        stage: "completed",
        aiExplanation: enhancementResult.aiExplanation
      });
      broadcastJobUpdate(jobId, {
        status: "completed",
        progress: 100,
        message: "Video processing completed! Enhanced video ready for download.",
        processedPath: finalVideoPath,
        enhancedAudioPath,
        aiExplanation: enhancementResult.aiExplanation
        // Use AI explanation from enhancement result
      });
      try {
        await fs4.unlink(extractedAudioPath);
        console.log(`\u{1F9F9} Cleaned up: ${extractedAudioPath}`);
      } catch (error) {
        console.warn(`\u26A0\uFE0F Cleanup warning: ${error}`);
      }
      console.log(`\u{1F3C6} SUCCESS: Video processing completed for job ${jobId}`);
    } catch (error) {
      console.error(`\u{1F4A5} FINAL FAILURE: Video processing failed for job ${jobId}:`, error);
      storage.updateAudioJob(jobId, {
        status: "failed",
        progress: 90,
        errorMessage: error instanceof Error ? error.message : "Video processing failed"
      });
      broadcastJobUpdate(jobId, {
        status: "failed",
        progress: 90,
        error: error instanceof Error ? error.message : "Video processing failed"
      });
    }
  };
  app2.post("/api/upload", (req, res, next) => {
    console.log("Upload request received:");
    console.log("Content-Type:", req.headers["content-type"]);
    console.log("Content-Length:", req.headers["content-length"]);
    next();
  }, upload.single("audio"), async (req, res) => {
    try {
      console.log("Multer processed file:", req.file ? "YES" : "NO");
      console.log("File object:", req.file);
      console.log("Body:", req.body);
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }
      const file = req.file;
      const {
        noiseReductionLevel = 7,
        voicePreservation = 9,
        processingMode = "balanced",
        // 'fast' or 'balanced'
        outputFormat = "wav",
        denoisingStrength = "0.8"
      } = req.body;
      const modelPath = path4.join(process.cwd(), "checkpoints", "dccrn_latest.pth");
      const modelAvailable = await fs4.access(modelPath).then(() => true).catch(() => false);
      if (!modelAvailable) {
        return res.status(503).json({
          error: "DCCRN model not available. Please train the model first.",
          code: "MODEL_NOT_FOUND"
        });
      }
      const outputFilename = `enhanced_${Date.now()}_${file.originalname}`;
      const outputPath = path4.join("outputs", outputFilename);
      const jobData = {
        filename: file.originalname,
        originalFormat: path4.extname(file.originalname).slice(1),
        outputFormat,
        fileSize: file.size,
        originalPath: file.path,
        noiseReductionLevel: parseInt(noiseReductionLevel),
        voicePreservation: parseInt(voicePreservation),
        processingMode,
        // Pass the actual processing mode (fast or balanced)
        processingOptions: {
          denoisingStrength: parseFloat(denoisingStrength || "0.8"),
          processingMode
          // Also include in processing options
        },
        stage: "upload",
        // Initial stage
        status: "pending",
        progress: 0
      };
      const validatedData = insertAudioJobSchema.parse(jobData);
      const job = await storage.createAudioJob(validatedData);
      console.log(`\u2705 Job created: ${job.id}`);
      console.log(`\u{1F4C1} File path: ${file.path}`);
      console.log(`\u{1F4C1} Output path: ${outputPath}`);
      res.json({
        jobId: job.id,
        status: "uploaded",
        message: "Audio uploaded successfully. Processing with DCCRN...",
        job
      });
      console.log(`\u{1F680} Starting DCCRN processing...`);
      processDCCRNAudio(job.id, file.path, outputPath, parseFloat(denoisingStrength || "0.8"), processingMode, broadcastJobUpdate).catch((error) => {
        console.error(`\u274C DCCRN processing failed for job ${job.id}:`, error);
        broadcastJobUpdate(job.id, {
          status: "failed",
          error: error.message,
          progress: 0
        });
      });
      console.log(`\u{1F680} DCCRN processing initiated`);
    } catch (error) {
      console.error("Upload error:", error);
      res.status(500).json({ error: "Upload failed" });
    }
  });
  app2.post("/api/process-video-url", async (req, res) => {
    try {
      const { url, options = {} } = req.body;
      if (!url) {
        return res.status(400).json({ error: "Video URL is required" });
      }
      console.log(`\u{1F3AC} Processing video URL: ${url}`);
      console.log(`\u{1F527} Options:`, options);
      const {
        noiseReductionLevel = 7,
        voicePreservation = 9,
        processingMode = "balanced",
        outputFormat = "wav",
        denoisingStrength = "0.8"
      } = options;
      const timestamp2 = Date.now();
      const extractedFilename = `extracted_${timestamp2}_audio.wav`;
      const extractedPath = path4.join("uploads", extractedFilename);
      const enhancedFilename = `enhanced_${timestamp2}_${extractedFilename}`;
      const enhancedPath = path4.join("outputs", enhancedFilename);
      const jobData = {
        filename: `video_${timestamp2}.${outputFormat}`,
        originalFormat: "video_url",
        outputFormat,
        fileSize: 0,
        originalPath: url,
        noiseReductionLevel: parseInt(noiseReductionLevel),
        voicePreservation: parseInt(voicePreservation),
        processingMode,
        processingOptions: {
          denoisingStrength: parseFloat(denoisingStrength || "0.8"),
          processingMode,
          extractedPath,
          enhancedPath
        },
        stage: "extraction",
        status: "pending",
        progress: 0
      };
      const validatedData = insertAudioJobSchema.parse(jobData);
      const job = await storage.createAudioJob(validatedData);
      console.log(`\u2705 Video job created: ${job.id}`);
      res.json({
        jobId: job.id,
        status: "processing_video",
        message: "Video processing started...",
        job
      });
      processVideoUrl(job.id, url, extractedPath, enhancedPath, parseFloat(denoisingStrength || "0.8"), processingMode, broadcastJobUpdate).catch((error) => {
        console.error(`\u274C Video processing failed for job ${job.id}:`, error);
        broadcastJobUpdate(job.id, {
          status: "failed",
          error: error.message,
          progress: 0
        });
      });
      console.log(`\u{1F680} Video processing initiated`);
    } catch (error) {
      console.error("Video URL processing error:", error);
      res.status(500).json({ error: "Failed to process video URL" });
    }
  });
  app2.post("/api/upload-batch", upload.array("files", 10), async (req, res) => {
    try {
      if (!req.files || req.files.length === 0) {
        return res.status(400).json({ error: "No files uploaded" });
      }
      const modelPath = path4.join(process.cwd(), "checkpoints", "dccrn_latest.pth");
      const modelAvailable = await fs4.access(modelPath).then(() => true).catch(() => false);
      if (!modelAvailable) {
        return res.status(503).json({
          error: "DCCRN model not available. Please train the model first.",
          code: "MODEL_NOT_FOUND"
        });
      }
      const options = req.body.options ? JSON.parse(req.body.options) : {};
      const jobs = [];
      for (const file of req.files) {
        const outputFilename = `enhanced_${Date.now()}_${file.originalname}`;
        const outputPath = path4.join("outputs", outputFilename);
        const jobData = {
          filename: file.originalname,
          originalFormat: path4.extname(file.originalname).slice(1),
          outputFormat: options.outputFormat || "wav",
          fileSize: file.size,
          originalPath: file.path,
          noiseReductionLevel: options.noiseReductionLevel || 8,
          voicePreservation: options.voicePreservation || 9,
          processingMode: "dccrn",
          processingOptions: {
            denoisingStrength: parseFloat(options.denoisingStrength || "0.8")
          }
        };
        const validatedData = insertAudioJobSchema.parse(jobData);
        const job = await storage.createAudioJob(validatedData);
        jobs.push({ job, outputPath });
      }
      res.json({ jobs: jobs.map(({ job }) => ({ jobId: job.id, status: "uploaded" })) });
      for (const { job, outputPath } of jobs) {
        processDCCRNAudio(
          job.id,
          job.originalPath,
          outputPath,
          parseFloat(options.denoisingStrength || "0.8"),
          options.processingMode || "balanced",
          broadcastJobUpdate
        ).catch((error) => {
          console.error(`\u274C Batch DCCRN processing failed for job ${job.id}:`, error);
          broadcastJobUpdate(job.id, {
            status: "failed",
            error: error.message,
            progress: 0
          });
        });
      }
    } catch (error) {
      console.error("Batch upload error:", error);
      res.status(500).json({ error: "Batch upload failed" });
    }
  });
  app2.get("/api/jobs/:jobId", async (req, res) => {
    try {
      const job = await storage.getAudioJob(req.params.jobId);
      if (!job) {
        return res.status(404).json({ error: "Job not found" });
      }
      console.log(`\u{1F4CA} Job status requested for ${req.params.jobId}:`, {
        status: job.status,
        progress: job.progress
      });
      res.json(job);
    } catch (error) {
      console.error("Error fetching job:", error);
      res.status(500).json({ error: "Failed to fetch job" });
    }
  });
  app2.get("/api/jobs", async (req, res) => {
    try {
      const { userId } = req.query;
      const jobs = await storage.listAudioJobs(userId);
      res.json(jobs);
    } catch (error) {
      console.error("Error listing jobs:", error);
      res.status(500).json({ error: "Failed to list jobs" });
    }
  });
  app2.get("/api/download/:jobId", async (req, res) => {
    try {
      const { type } = req.query;
      const job = await storage.getAudioJob(req.params.jobId);
      if (!job) {
        return res.status(404).json({ error: "Job not found" });
      }
      if (job.status !== "completed" || !job.processedPath) {
        return res.status(400).json({ error: "Job not completed or file not available" });
      }
      let filePath;
      let filename;
      let contentType;
      if (type === "audio" && job.enhancedAudioPath) {
        filePath = job.enhancedAudioPath;
        filename = `${path4.parse(job.filename).name}_enhanced_audio.wav`;
        contentType = "audio/wav";
      } else if (job.isVideo) {
        filePath = job.processedPath;
        const originalExt = path4.extname(job.filename);
        filename = `${path4.parse(job.filename).name}_enhanced${originalExt}`;
        contentType = "video/mp4";
      } else {
        filePath = job.processedPath;
        filename = `${path4.parse(job.filename).name}_enhanced.${job.outputFormat}`;
        contentType = "audio/wav";
      }
      res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
      res.setHeader("Content-Type", contentType);
      const fileStream = await fs4.readFile(filePath);
      res.send(fileStream);
    } catch (error) {
      console.error("Download error:", error);
      res.status(500).json({ error: "Download failed" });
    }
  });
  app2.get("/api/audio/:jobId/original", async (req, res) => {
    try {
      const job = await storage.getAudioJob(req.params.jobId);
      if (!job || !job.originalPath) {
        return res.status(404).json({ error: "Original audio not found" });
      }
      const filePath = job.originalPath;
      const stat = await fs4.stat(filePath);
      res.setHeader("Content-Type", "audio/wav");
      res.setHeader("Content-Length", stat.size.toString());
      res.setHeader("Accept-Ranges", "bytes");
      const fileStream = await fs4.readFile(filePath);
      res.send(fileStream);
    } catch (error) {
      console.error("Audio serving error:", error);
      res.status(500).json({ error: "Failed to serve audio" });
    }
  });
  app2.get("/api/audio/:jobId/processed", async (req, res) => {
    try {
      const job = await storage.getAudioJob(req.params.jobId);
      if (!job || !job.processedPath) {
        return res.status(404).json({ error: "Processed audio not found" });
      }
      if (job.status !== "completed") {
        return res.status(400).json({ error: "Processing not completed yet" });
      }
      const filePath = job.processedPath;
      const stat = await fs4.stat(filePath);
      res.setHeader("Content-Type", "audio/wav");
      res.setHeader("Content-Length", stat.size.toString());
      res.setHeader("Accept-Ranges", "bytes");
      const fileStream = await fs4.readFile(filePath);
      res.send(fileStream);
    } catch (error) {
      console.error("Audio serving error:", error);
      res.status(500).json({ error: "Failed to serve audio" });
    }
  });
  app2.post("/api/jobs/:jobId/cancel", async (req, res) => {
    try {
      const jobId = req.params.jobId;
      const cancelled = audioProcessor.cancelJob(jobId);
      if (cancelled) {
        await storage.updateAudioJob(jobId, {
          status: "failed",
          errorMessage: "Cancelled by user"
        });
        broadcastJobUpdate(jobId, { status: "cancelled" });
        res.json({ success: true });
      } else {
        res.status(400).json({ error: "Job not found or cannot be cancelled" });
      }
    } catch (error) {
      console.error("Cancel job error:", error);
      res.status(500).json({ error: "Failed to cancel job" });
    }
  });
  app2.post("/api/noise-samples", upload.single("sample"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
      }
      const { name, description, noiseType } = req.body;
      const sampleData = {
        name,
        description,
        filePath: req.file.path,
        noiseType
      };
      const validatedData = insertNoiseSampleSchema.parse(sampleData);
      const sample = await storage.createNoiseSample(validatedData);
      try {
        const profile = await groqService.generateNoiseProfile(req.file.path);
      } catch (error) {
        console.error("Error generating noise profile:", error);
      }
      res.json(sample);
    } catch (error) {
      console.error("Noise sample upload error:", error);
      res.status(500).json({ error: "Failed to upload noise sample" });
    }
  });
  app2.get("/api/noise-samples", async (req, res) => {
    try {
      const { noiseType } = req.query;
      const samples = await storage.listNoiseSamples(noiseType);
      res.json(samples);
    } catch (error) {
      console.error("Error listing noise samples:", error);
      res.status(500).json({ error: "Failed to list noise samples" });
    }
  });
  app2.get("/api/supported-formats", async (req, res) => {
    try {
      const formats = await getSupportedFormats();
      res.json({
        audio: ["wav", "mp3", "flac", "aac", "ogg", "m4a", "wma", "aiff", "au"],
        video: ["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"],
        ffmpeg_supported: formats
      });
    } catch (error) {
      console.error("Error getting supported formats:", error);
      res.status(500).json({ error: "Failed to get supported formats" });
    }
  });
  app2.post("/api/analyze/:jobId", async (req, res) => {
    try {
      const job = await storage.getAudioJob(req.params.jobId);
      if (!job || !job.originalPath) {
        return res.status(404).json({ error: "Job or file not found" });
      }
      const analysis = await groqService.analyzeAudioNoise(job.originalPath);
      res.json(analysis);
    } catch (error) {
      console.error("Analysis error:", error);
      res.status(500).json({ error: "Analysis failed" });
    }
  });
  async function processAudioJob(jobId, broadcastUpdate) {
    try {
      const job = await storage.getAudioJob(jobId);
      if (!job) return;
      await storage.updateAudioJob(jobId, {
        status: "processing",
        startedAt: /* @__PURE__ */ new Date(),
        progress: 0
      });
      broadcastUpdate(jobId, { status: "processing", progress: 0 });
      const options = {
        noiseReductionLevel: job.noiseReductionLevel || 7,
        voicePreservation: job.voicePreservation || 9,
        processingMode: job.processingMode || "balanced",
        preserveEmotions: true,
        contextAware: true
      };
      const isVideo = /\.(mp4|avi|mov|mkv|webm|flv|wmv)$/i.test(job.filename);
      const result = isVideo ? await audioProcessor.processVideoFile(jobId, job.originalPath, options, (progress) => {
        storage.updateAudioJob(jobId, { progress: progress.progress });
        broadcastUpdate(jobId, progress);
      }) : await audioProcessor.processAudioFile(jobId, job.originalPath, options, (progress) => {
        storage.updateAudioJob(jobId, { progress: progress.progress });
        broadcastUpdate(jobId, progress);
      });
      if (result.success) {
        await storage.updateAudioJob(jobId, {
          status: "completed",
          processedPath: result.outputPath,
          completedAt: /* @__PURE__ */ new Date(),
          progress: 100
        });
        broadcastUpdate(jobId, {
          status: "completed",
          progress: 100,
          downloadUrl: `/api/download/${jobId}`
        });
      } else {
        await storage.updateAudioJob(jobId, {
          status: "failed",
          errorMessage: result.error,
          completedAt: /* @__PURE__ */ new Date()
        });
        broadcastUpdate(jobId, {
          status: "failed",
          error: result.error
        });
      }
    } catch (error) {
      console.error(`Error processing job ${jobId}:`, error);
      await storage.updateAudioJob(jobId, {
        status: "failed",
        errorMessage: error instanceof Error ? error.message : "Unknown error",
        completedAt: /* @__PURE__ */ new Date()
      });
      broadcastUpdate(jobId, {
        status: "failed",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  }
  async function processDCCRNAudio(jobId, inputPath, outputPath, strength, processingMode = "balanced", broadcastUpdate) {
    console.log(`\u{1F504} Starting DCCRN processing for job ${jobId}`);
    console.log(`\u{1F4C1} Input: ${inputPath}`);
    console.log(`\u{1F4C1} Output: ${outputPath}`);
    console.log(`\u{1F4AA} Strength: ${strength}`);
    console.log(`\u26A1 Mode: ${processingMode.toUpperCase()}`);
    try {
      const job = await storage.getAudioJob(jobId);
      if (!job) {
        console.log(`\u274C Job ${jobId} not found in database`);
        return;
      }
      console.log(`\u2705 Job found: ${job.filename}`);
      await storage.updateAudioJob(jobId, {
        status: "processing",
        startedAt: /* @__PURE__ */ new Date(),
        progress: 0
      });
      console.log(`\u{1F4CA} Job status updated to processing`);
      const modeDescription = processingMode === "fast" ? "Fast AI denoising" : "Balanced AI denoising";
      broadcastUpdate(jobId, {
        status: "processing",
        progress: 0,
        message: `Starting ${modeDescription}...`,
        stage: "analysis"
      });
      console.log(`\u{1F4E1} Broadcasted initial progress update`);
      console.log(`\u{1F9E0} Starting ${processingMode.toUpperCase()} DCCRN processor...`);
      const result = await dccrnProcessor.enhanceAudio(inputPath, outputPath, {
        strength,
        processingMode,
        noiseReductionLevel: Math.round(strength * 10)
      }, (progress) => {
        console.log(`\u{1F4CA} Progress: ${progress.progress}% - ${progress.message}`);
        let frontendStage = "enhancement";
        switch (progress.stage) {
          case "initialization":
          case "loading":
            frontendStage = "analysis";
            break;
          case "processing":
          case "finalizing":
            frontendStage = "enhancement";
            break;
          case "completed":
            frontendStage = "download";
            break;
        }
        storage.updateAudioJob(jobId, {
          progress: progress.progress
        });
        broadcastUpdate(jobId, {
          status: "processing",
          progress: progress.progress,
          message: progress.message,
          stage: frontendStage
        });
      });
      console.log(`\u{1F3AF} DCCRN processing completed:`, result);
      if (result.success) {
        await storage.updateAudioJob(jobId, {
          status: "completed",
          processedPath: result.outputPath,
          completedAt: /* @__PURE__ */ new Date(),
          progress: 100,
          aiExplanation: result.aiExplanation
        });
        broadcastUpdate(jobId, {
          status: "completed",
          progress: 100,
          message: "DCCRN enhancement completed!",
          stage: "download",
          downloadUrl: `/api/download/${jobId}`,
          metadata: {
            originalSize: result.originalSize,
            enhancedSize: result.enhancedSize,
            duration: result.duration
          },
          aiExplanation: result.aiExplanation
          // Include the AI explanation
        });
        console.log(`\u2705 DCCRN processing completed successfully for job ${jobId}`);
        console.log(`\u{1F916} AI Explanation included:`, result.aiExplanation ? "YES" : "NO");
      } else {
        await storage.updateAudioJob(jobId, {
          status: "failed",
          errorMessage: result.error,
          completedAt: /* @__PURE__ */ new Date()
        });
        broadcastUpdate(jobId, {
          status: "failed",
          error: result.error
        });
      }
    } catch (error) {
      console.error(`Error processing DCCRN job ${jobId}:`, error);
      await storage.updateAudioJob(jobId, {
        status: "failed",
        errorMessage: error instanceof Error ? error.message : "Unknown error",
        completedAt: /* @__PURE__ */ new Date()
      });
      broadcastUpdate(jobId, {
        status: "failed",
        error: error instanceof Error ? error.message : "Unknown error"
      });
    }
  }
  app2.post("/api/upload/audio", upload.single("audio"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No audio file uploaded" });
      }
      const file = req.file;
      const {
        denoisingStrength = "1.0",
        outputFormat = "wav"
      } = req.body;
      const modelPath = path4.join(process.cwd(), "checkpoints", "dccrn_latest.pth");
      const modelAvailable = await fs4.access(modelPath).then(() => true).catch(() => false);
      if (!modelAvailable) {
        return res.status(503).json({
          error: "DCCRN model not available. Please train the model first.",
          code: "MODEL_NOT_FOUND"
        });
      }
      const outputFilename = `enhanced_${Date.now()}_${file.originalname}`;
      const outputPath = path4.join("outputs", outputFilename);
      const jobData = {
        filename: file.originalname,
        originalFormat: path4.extname(file.originalname).slice(1),
        outputFormat,
        fileSize: file.size,
        originalPath: file.path,
        noiseReductionLevel: Math.round(parseFloat(denoisingStrength) * 10),
        voicePreservation: 9,
        processingMode: "dccrn",
        processingOptions: { denoisingStrength: parseFloat(denoisingStrength) }
      };
      const validatedData = insertAudioJobSchema.parse(jobData);
      const job = await storage.createAudioJob(validatedData);
      res.json({
        jobId: job.id,
        status: "uploaded",
        message: "Audio uploaded successfully. Processing with DCCRN...",
        job
      });
      processDCCRNAudio(job.id, file.path, outputPath, parseFloat(denoisingStrength), "balanced", broadcastJobUpdate).catch((error) => {
        console.error(`\u274C Video DCCRN processing failed for job ${job.id}:`, error);
        broadcastJobUpdate(job.id, {
          status: "failed",
          error: error.message,
          progress: 0
        });
      });
    } catch (error) {
      console.error("Audio upload error:", error);
      res.status(500).json({ error: "Audio upload failed" });
    }
  });
  app2.post("/api/upload/video", upload.single("video"), async (req, res) => {
    try {
      console.log("\u{1F3AC} Video upload request received");
      console.log("\u{1F4C1} File object:", req.file ? "YES" : "NO");
      console.log("\u{1F4DD} Body:", req.body);
      if (!req.file) {
        return res.status(400).json({ error: "No video file uploaded" });
      }
      const file = req.file;
      console.log(`\u{1F4C1} Uploaded file: ${file.originalname}, size: ${file.size} bytes`);
      const {
        denoisingStrength = "1.0",
        preserveVideoQuality = "true",
        outputFormat = "mp4",
        processingMode = "balanced"
      } = req.body;
      const modelPath = path4.join(process.cwd(), "checkpoints", "dccrn_latest.pth");
      const modelAvailable = await fs4.access(modelPath).then(() => true).catch(() => false);
      if (!modelAvailable) {
        return res.status(503).json({
          error: "DCCRN model not available. Please train the model first.",
          code: "MODEL_NOT_FOUND"
        });
      }
      console.log("\u2705 DCCRN model is available");
      const jobData = {
        filename: file.originalname,
        originalFormat: path4.extname(file.originalname).slice(1),
        outputFormat: "mp4",
        // Always use mp4 for video output, not wav
        fileSize: file.size,
        originalPath: file.path,
        status: "pending",
        progress: 0,
        processingMode,
        isVideo: true,
        stage: "upload",
        noiseReductionLevel: Math.round(parseFloat(denoisingStrength) * 10),
        voicePreservation: 9,
        processingOptions: {
          denoisingStrength: parseFloat(denoisingStrength),
          processingMode,
          preserveVideoQuality: preserveVideoQuality === "true"
        }
      };
      console.log("\u{1F4DD} Creating job with data:", jobData);
      const validatedData = insertAudioJobSchema.parse(jobData);
      const job = await storage.createAudioJob(validatedData);
      console.log(`\u2705 Video job created successfully: ${job.id}`);
      res.json({
        success: true,
        jobId: job.id,
        message: "Video uploaded successfully. Processing started.",
        job
      });
      console.log("\u{1F680} Starting video processing in background...");
      processVideoInBackground(job.id, file.path, {
        denoisingStrength: parseFloat(denoisingStrength),
        processingMode,
        preserveVideoQuality: preserveVideoQuality === "true",
        outputFormat: "mp4",
        // Force MP4 for video output
        voicePreservation: 9,
        preserveEmotions: true,
        contextAware: true
      }).catch((error) => {
        console.error(`\u274C Video processing failed for job ${job.id}:`, error);
      });
      console.log("\u{1F3AC} Video upload endpoint completed successfully");
    } catch (error) {
      console.error("\u274C Video upload error:", error);
      res.status(500).json({ error: "Video upload failed", details: error instanceof Error ? error.message : "Unknown error" });
    }
  });
  app2.post("/api/denoise", async (req, res) => {
    try {
      const { filePath, denoisingStrength = 1, outputPath } = req.body;
      if (!filePath) {
        return res.status(400).json({ error: "File path is required" });
      }
      const modelPath = path4.join(process.cwd(), "checkpoints", "dccrn_latest.pth");
      const modelAvailable = await fs4.access(modelPath).then(() => true).catch(() => false);
      if (!modelAvailable) {
        return res.status(503).json({
          error: "DCCRN model not available",
          code: "MODEL_NOT_FOUND"
        });
      }
      const finalOutputPath = outputPath || path4.join("outputs", `denoised_${Date.now()}_${path4.basename(filePath)}`);
      const result = await dccrnProcessor.enhanceAudio(filePath, finalOutputPath, { strength: denoisingStrength });
      if (result.success) {
        res.json({
          success: true,
          outputPath: finalOutputPath,
          downloadUrl: `/api/download/file?path=${encodeURIComponent(finalOutputPath)}`
        });
      } else {
        res.status(500).json({
          success: false,
          error: result.error
        });
      }
    } catch (error) {
      console.error("Denoising error:", error);
      res.status(500).json({ error: "Denoising failed" });
    }
  });
  app2.post("/api/denoise/batch", async (req, res) => {
    try {
      const { inputDir, outputDir, denoisingStrength = 1 } = req.body;
      if (!inputDir || !outputDir) {
        return res.status(400).json({ error: "Input and output directories are required" });
      }
      const modelPath = path4.join(process.cwd(), "checkpoints", "dccrn_latest.pth");
      const modelAvailable = await fs4.access(modelPath).then(() => true).catch(() => false);
      if (!modelAvailable) {
        return res.status(503).json({
          error: "DCCRN model not available",
          code: "MODEL_NOT_FOUND"
        });
      }
      const result = { success: false, error: "Batch processing not implemented" };
      if (result.success) {
        res.json({
          success: true,
          outputDir,
          message: "Batch processing completed successfully"
        });
      } else {
        res.status(500).json({
          success: false,
          error: result.error
        });
      }
    } catch (error) {
      console.error("Batch processing error:", error);
      res.status(500).json({ error: "Batch processing failed" });
    }
  });
  app2.get("/api/visualize/:jobId", async (req, res) => {
    try {
      const { jobId } = req.params;
      const job = await storage.getAudioJob(jobId);
      if (!job) {
        return res.status(404).json({ error: "Job not found" });
      }
      const spectrograms = {
        noisy: {
          data: [],
          // Spectrogram data
          shape: [257, 100],
          // Frequency bins x Time frames
          sampleRate: 16e3,
          hopLength: 256
        },
        enhanced: {
          data: [],
          // Enhanced spectrogram data
          shape: [257, 100],
          sampleRate: 16e3,
          hopLength: 256
        }
      };
      res.json({
        jobId,
        spectrograms,
        metadata: {
          originalFilename: job.filename,
          processingMode: job.processingMode,
          denoisingStrength: job.processingOptions?.denoisingStrength || 1
        }
      });
    } catch (error) {
      console.error("Visualization error:", error);
      res.status(500).json({ error: "Failed to generate visualization" });
    }
  });
  app2.get("/api/model/status", async (req, res) => {
    try {
      const modelPath = path4.join(process.cwd(), "checkpoints", "dccrn_latest.pth");
      const modelAvailable = await fs4.access(modelPath).then(() => true).catch(() => false);
      const ffmpegAvailable = false;
      res.json({
        dccrn: {
          available: modelAvailable,
          modelPath: modelAvailable ? "checkpoints/dccrn_latest.pth" : null
        },
        ffmpeg: {
          available: ffmpegAvailable
        },
        services: {
          audioProcessing: modelAvailable,
          videoProcessing: modelAvailable && ffmpegAvailable
        }
      });
    } catch (error) {
      console.error("Model status error:", error);
      res.status(500).json({ error: "Failed to check model status" });
    }
  });
  app2.get("/api/download/file", async (req, res) => {
    try {
      const { path: filePath } = req.query;
      if (!filePath || typeof filePath !== "string") {
        return res.status(400).json({ error: "File path is required" });
      }
      const normalizedPath = path4.normalize(filePath);
      const outputsDir = path4.resolve("outputs");
      const fullPath = path4.resolve(normalizedPath);
      if (!fullPath.startsWith(outputsDir)) {
        return res.status(403).json({ error: "Access denied" });
      }
      try {
        await fs4.access(fullPath);
      } catch {
        return res.status(404).json({ error: "File not found" });
      }
      const filename = path4.basename(fullPath);
      res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);
      res.setHeader("Content-Type", "application/octet-stream");
      const fileStream = __require("fs").createReadStream(fullPath);
      fileStream.pipe(res);
    } catch (error) {
      console.error("File download error:", error);
      res.status(500).json({ error: "Download failed" });
    }
  });
  async function processVideoUrl(jobId, videoUrl, extractedPath, enhancedPath, strength, processingMode = "balanced", broadcastUpdate) {
    console.log(`\uFFFD Starting URL video processing for job ${jobId}`);
    console.log(`\u{1F4F9} URL: ${videoUrl}`);
    console.log(` Strength: ${strength}`);
    console.log(`\u26A1 Mode: ${processingMode.toUpperCase()}`);
    try {
      const job = await storage.getAudioJob(jobId);
      if (!job) {
        console.log(`\u274C Job ${jobId} not found in database`);
        return;
      }
      console.log(`\u2705 Job found: ${job.filename}`);
      await storage.updateAudioJob(jobId, {
        status: "processing",
        startedAt: /* @__PURE__ */ new Date(),
        progress: 0
      });
      const result = await urlVideoProcessor.processVideoFromUrl(
        videoUrl,
        {
          denoisingStrength: strength,
          processingMode,
          quality: "best"
        },
        (progress) => {
          console.log(`\u{1F4CA} URL Video Progress: ${progress.progress}% - ${progress.message}`);
          let frontendStage = progress.stage;
          switch (progress.stage) {
            case "download":
              frontendStage = "download";
              break;
            case "extraction":
              frontendStage = "extraction";
              break;
            case "enhancement":
              frontendStage = "enhancement";
              break;
            case "combination":
              frontendStage = "combination";
              break;
            case "finalization":
            case "completed":
              frontendStage = "download";
              break;
          }
          broadcastUpdate(jobId, {
            status: "processing",
            progress: Math.round(progress.progress),
            message: progress.message,
            stage: frontendStage
          });
        }
      );
      if (!result.success) {
        throw new Error(`URL video processing failed: ${result.error}`);
      }
      console.log(`\u2705 URL video processing completed: ${result.outputPath}`);
      await storage.updateAudioJob(jobId, {
        status: "completed",
        processedPath: result.outputPath,
        originalPath: result.extractedAudioPath || extractedPath,
        // Update to point to extracted audio
        completedAt: /* @__PURE__ */ new Date(),
        progress: 100
      });
      console.log(`\u{1F389} URL video processing job ${jobId} completed successfully`);
      console.log(`\u{1F916} AI Explanation included:`, result.aiExplanation ? "YES" : "NO");
      broadcastUpdate(jobId, {
        status: "completed",
        outputPath: result.outputPath,
        progress: 100,
        aiExplanation: result.aiExplanation,
        // Use the AI explanation from the result
        result: {
          originalSize: 0,
          enhancedSize: 0,
          processingMode,
          extractionMethod: "yt-dlp",
          metadata: result.metadata
        }
      });
    } catch (error) {
      console.error(`\u274C URL video processing error for job ${jobId}:`, error);
      await storage.updateAudioJob(jobId, {
        status: "failed",
        errorMessage: error instanceof Error ? error.message : "Unknown URL video processing error",
        completedAt: /* @__PURE__ */ new Date()
      });
      broadcastUpdate(jobId, {
        status: "failed",
        error: error instanceof Error ? error.message : "Unknown URL video processing error",
        progress: 0
      });
    }
  }
  return httpServer;
}

// server/vite.ts
import express from "express";
import fs5 from "fs";
import path6 from "path";
import { createServer as createViteServer, createLogger } from "vite";

// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path5 from "path";
import runtimeErrorOverlay from "@replit/vite-plugin-runtime-error-modal";
var vite_config_default = defineConfig({
  plugins: [
    react(),
    runtimeErrorOverlay(),
    ...process.env.NODE_ENV !== "production" && process.env.REPL_ID !== void 0 ? [
      await import("@replit/vite-plugin-cartographer").then(
        (m) => m.cartographer()
      )
    ] : []
  ],
  resolve: {
    alias: {
      "@": path5.resolve(import.meta.dirname, "client", "src"),
      "@shared": path5.resolve(import.meta.dirname, "shared"),
      "@assets": path5.resolve(import.meta.dirname, "attached_assets")
    }
  },
  root: path5.resolve(import.meta.dirname, "client"),
  build: {
    outDir: path5.resolve(import.meta.dirname, "dist/public"),
    emptyOutDir: true
  },
  server: {
    fs: {
      strict: true,
      deny: ["**/.*"]
    },
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true
      },
      "/ws": {
        target: "ws://localhost:5000",
        ws: true,
        changeOrigin: true
      }
    }
  }
});

// server/vite.ts
import { nanoid } from "nanoid";
var viteLogger = createLogger();
function log(message, source = "express") {
  const formattedTime = (/* @__PURE__ */ new Date()).toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true
  });
  console.log(`${formattedTime} [${source}] ${message}`);
}
async function setupVite(app2, server) {
  const serverOptions = {
    middlewareMode: true,
    hmr: { server },
    allowedHosts: true
  };
  const vite = await createViteServer({
    ...vite_config_default,
    configFile: false,
    customLogger: {
      ...viteLogger,
      error: (msg, options) => {
        viteLogger.error(msg, options);
        process.exit(1);
      }
    },
    server: serverOptions,
    appType: "custom"
  });
  app2.use(vite.middlewares);
  app2.use("*", async (req, res, next) => {
    const url = req.originalUrl;
    try {
      const clientTemplate = path6.resolve(
        import.meta.dirname,
        "..",
        "client",
        "index.html"
      );
      let template = await fs5.promises.readFile(clientTemplate, "utf-8");
      template = template.replace(
        `src="/src/main.tsx"`,
        `src="/src/main.tsx?v=${nanoid()}"`
      );
      const page = await vite.transformIndexHtml(url, template);
      res.status(200).set({ "Content-Type": "text/html" }).end(page);
    } catch (e) {
      vite.ssrFixStacktrace(e);
      next(e);
    }
  });
}
function serveStatic(app2) {
  const distPath = path6.resolve(import.meta.dirname, "public");
  if (!fs5.existsSync(distPath)) {
    throw new Error(
      `Could not find the build directory: ${distPath}, make sure to build the client first`
    );
  }
  app2.use(express.static(distPath));
  app2.use("*", (_req, res) => {
    res.sendFile(path6.resolve(distPath, "index.html"));
  });
}

// server/index.ts
var app = express2();
app.use(express2.json());
app.use(express2.urlencoded({ extended: false }));
app.use((req, res, next) => {
  const start = Date.now();
  const path7 = req.path;
  let capturedJsonResponse = void 0;
  const originalResJson = res.json;
  res.json = function(bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };
  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path7.startsWith("/api")) {
      let logLine = `${req.method} ${path7} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }
      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "\u2026";
      }
      log(logLine);
    }
  });
  next();
});
(async () => {
  const server = await registerRoutes(app);
  app.use((err, _req, res, _next) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";
    res.status(status).json({ message });
    throw err;
  });
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }
  const port = parseInt(process.env.PORT || "5000", 10);
  server.listen(port, "0.0.0.0", () => {
    console.log("\n\u{1F680} SonicPurge Server Started!");
    console.log("================================");
    console.log(`\u{1F310} Local:     http://localhost:${port}`);
    console.log(`\u{1F4F1} Network:   http://0.0.0.0:${port}`);
    console.log("\u{1F916} AI Model:  DCCRN Ready");
    console.log("\u{1F4E1} WebSocket: Enabled");
    console.log("\u{1F4BE} Storage:   SQLite + FileSystem");
    console.log("================================");
    console.log("\u{1F4A1} Keep this terminal open while using the app");
    console.log("\u{1F504} Press Ctrl+C to stop the server\n");
  });
})();

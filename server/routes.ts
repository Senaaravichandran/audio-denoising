import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import multer from "multer";
import path from "path";
import crypto from "crypto";
import { promises as fs } from "fs";
import { spawn } from "child_process";
import { storage } from "./storage";
import { audioProcessor } from "./services/audioProcessor";
import { dccrnProcessor } from "./services/dccrnProcessor";
import { groqService } from "./services/groqService";
// import { videoProcessingService } from "./services/videoProcessingService";
// import { dccrnService } from "./services/dccrnService";
import { insertAudioJobSchema, insertNoiseSampleSchema } from "@shared/schema";
import { getAudioMetadata, getSupportedFormats, extractAudioFromVideo, combineAudioWithVideo } from "./utils/ffmpeg";
import { URLVideoProcessor } from "./services/urlVideoProcessor";

// Configure multer for file uploads
const upload = multer({
  storage: multer.diskStorage({
    destination: 'uploads/',
    filename: (req, file, cb) => {
      // Generate a unique filename with original extension
      const ext = path.extname(file.originalname);
      const name = crypto.randomBytes(16).toString('hex');
      cb(null, name + ext);
    }
  }),
  limits: {
    fileSize: 500 * 1024 * 1024, // 500MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedAudioTypes = /\.(wav|mp3|flac|aac|ogg|m4a|wma|aiff|au)$/i;
    const allowedVideoTypes = /\.(mp4|avi|mov|mkv|webm|flv|wmv)$/i;
    
    if (allowedAudioTypes.test(file.originalname) || allowedVideoTypes.test(file.originalname)) {
      cb(null, true);
    } else {
      cb(new Error('Unsupported file format'));
    }
  }
});

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  
  // WebSocket server for real-time updates
  const wss = new WebSocketServer({ server: httpServer, path: '/ws' });
  
  // Store WebSocket connections by job ID
  const wsConnections = new Map<string, WebSocket[]>();
  
  // Initialize URL Video Processor
  const urlVideoProcessor = new URLVideoProcessor(dccrnProcessor);
  
  wss.on('connection', (ws) => {
    console.log('New WebSocket connection');
    
    ws.on('message', (message) => {
      try {
        const data = JSON.parse(message.toString());
        
        if (data.type === 'subscribe' && data.jobId) {
          // Subscribe to job updates
          if (!wsConnections.has(data.jobId)) {
            wsConnections.set(data.jobId, []);
          }
          wsConnections.get(data.jobId)!.push(ws);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });
    
    ws.on('close', () => {
      // Remove connection from all subscriptions
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
  
  // Helper function to broadcast job updates
  const broadcastJobUpdate = (jobId: string, data: any) => {
    const connections = wsConnections.get(jobId);
    console.log(`📡 Broadcasting to ${connections?.length || 0} clients for job ${jobId}:`, data);
    
    if (connections) {
      const message = JSON.stringify({ type: 'job_update', jobId, data });
      connections.forEach(ws => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(message);
          console.log(`   ✅ Sent to client`);
        } else {
          console.log(`   ❌ Client connection not open`);
        }
      });
    } else {
      console.log(`   ❌ No WebSocket connections found for job ${jobId}`);
    }
  };

  // Video processing function - COMPLETELY REWRITTEN
  const processVideoInBackground = async (
    jobId: string,
    videoPath: string,
    options: {
      denoisingStrength: number;
      processingMode: 'fast' | 'balanced';
      preserveVideoQuality: boolean;
      outputFormat: string;
      voicePreservation: number;
      preserveEmotions: boolean;
      contextAware: boolean;
    }
  ) => {
    try {
      console.log(`🎬 FINAL ATTEMPT: Starting video processing for job ${jobId}`);
      
      // Generate proper output paths with timestamps
      const timestamp = Date.now();
      const extractedAudioPath = path.join('uploads', `extracted_${timestamp}.wav`);
      const enhancedAudioPath = path.join('outputs', `enhanced_${timestamp}.wav`);
      const finalVideoPath = path.join('outputs', `final_video_${timestamp}.mp4`);

      // STAGE 1: Extract audio from video using simple FFmpeg
      console.log(`🔊 STAGE 1: Extracting audio from ${videoPath}`);
      storage.updateAudioJob(jobId, { 
        status: 'processing', 
        progress: 10, 
        stage: 'video_extraction' 
      });
      broadcastJobUpdate(jobId, { 
        status: 'processing', 
        progress: 10, 
        message: 'Extracting audio from video...' 
      });

      await new Promise<void>((resolve, reject) => {
        const ffmpegExtract = spawn('ffmpeg', [
          '-i', videoPath,
          '-vn',                    // No video stream
          '-acodec', 'pcm_s16le',  // PCM 16-bit
          '-ar', '44100',          // 44.1kHz
          '-ac', '2',              // Stereo
          '-y',                    // Overwrite
          extractedAudioPath
        ]);

        let stderr = '';
        ffmpegExtract.stderr.on('data', (data) => {
          stderr += data.toString();
        });

        ffmpegExtract.on('close', (code) => {
          if (code === 0) {
            console.log(`✅ Audio extraction successful: ${extractedAudioPath}`);
            resolve();
          } else {
            console.error(`❌ Audio extraction failed with code ${code}`);
            reject(new Error(`Audio extraction failed: ${stderr}`));
          }
        });

        ffmpegExtract.on('error', (error) => {
          reject(error);
        });
      });

      // STAGE 2: Enhance audio with DCCRN
      console.log(`🤖 STAGE 2: Enhancing audio with DCCRN`);
      storage.updateAudioJob(jobId, { progress: 30 });
      broadcastJobUpdate(jobId, { 
        progress: 30, 
        message: 'Starting AI-powered audio enhancement...' 
      });

      const enhancementResult = await dccrnProcessor.enhanceAudio(
        extractedAudioPath,
        enhancedAudioPath,
        {
          strength: options.denoisingStrength,
          processingMode: options.processingMode
        },
        (progress) => {
          const mappedProgress = 30 + (progress.progress * 0.5);
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

      console.log(`✅ Audio enhanced: ${enhancedAudioPath}`);

      // STAGE 3: Combine enhanced audio with video
      console.log(`🎬 STAGE 3: Combining enhanced audio with video`);
      storage.updateAudioJob(jobId, { progress: 85 });
      broadcastJobUpdate(jobId, { 
        progress: 85, 
        message: 'Combining enhanced audio with video...' 
      });

      await new Promise<void>((resolve, reject) => {
        const ffmpegCombine = spawn('ffmpeg', [
          '-i', videoPath,          // Input video
          '-i', enhancedAudioPath,  // Input enhanced audio
          '-c:v', 'copy',           // Copy video (no re-encoding)
          '-c:a', 'aac',            // AAC audio codec
          '-b:a', '128k',           // Audio bitrate
          '-ar', '44100',           // Audio sample rate
          '-ac', '2',               // Stereo
          '-map', '0:v:0',          // Map video from input 0
          '-map', '1:a:0',          // Map audio from input 1
          '-shortest',              // Match shortest stream
          '-avoid_negative_ts', 'make_zero',
          '-fflags', '+genpts',
          '-y',                     // Overwrite
          finalVideoPath
        ]);

        let stderr = '';
        ffmpegCombine.stderr.on('data', (data) => {
          stderr += data.toString();
        });

        ffmpegCombine.on('close', (code) => {
          if (code === 0) {
            console.log(`✅ Video combination successful: ${finalVideoPath}`);
            resolve();
          } else {
            console.error(`❌ Video combination failed with code ${code}`);
            console.error(`FFmpeg stderr:`, stderr);
            reject(new Error(`Video combination failed: ${stderr}`));
          }
        });

        ffmpegCombine.on('error', (error) => {
          reject(error);
        });
      });

      // STAGE 4: Complete
      console.log(`🎉 STAGE 4: Video processing completed successfully!`);
      console.log(`🤖 AI Explanation included:`, enhancementResult.aiExplanation ? 'YES' : 'NO');
      await storage.updateAudioJob(jobId, {
        status: 'completed',
        progress: 100,
        processedPath: finalVideoPath,
        enhancedAudioPath: enhancedAudioPath,
        stage: 'completed',
        aiExplanation: enhancementResult.aiExplanation
      });

      broadcastJobUpdate(jobId, {
        status: 'completed',
        progress: 100,
        message: 'Video processing completed! Enhanced video ready for download.',
        processedPath: finalVideoPath,
        enhancedAudioPath: enhancedAudioPath,
        aiExplanation: enhancementResult.aiExplanation // Use AI explanation from enhancement result
      });

      // Cleanup temporary extracted audio file
      try {
        await fs.unlink(extractedAudioPath);
        console.log(`🧹 Cleaned up: ${extractedAudioPath}`);
      } catch (error) {
        console.warn(`⚠️ Cleanup warning: ${error}`);
      }

      console.log(`🏆 SUCCESS: Video processing completed for job ${jobId}`);

    } catch (error) {
      console.error(`💥 FINAL FAILURE: Video processing failed for job ${jobId}:`, error);
      
      storage.updateAudioJob(jobId, {
        status: 'failed',
        progress: 90,
        errorMessage: error instanceof Error ? error.message : 'Video processing failed'
      });

      broadcastJobUpdate(jobId, {
        status: 'failed',
        progress: 90,
        error: error instanceof Error ? error.message : 'Video processing failed'
      });
    }
  };

  // File upload endpoint - Updated to use DCCRN
  app.post('/api/upload', (req, res, next) => {
    console.log('Upload request received:');
    console.log('Content-Type:', req.headers['content-type']);
    console.log('Content-Length:', req.headers['content-length']);
    next();
  }, upload.single('audio'), async (req, res) => {
    try {
      console.log('Multer processed file:', req.file ? 'YES' : 'NO');
      console.log('File object:', req.file);
      console.log('Body:', req.body);
      
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      const file = req.file;

      const {
        noiseReductionLevel = 7,
        voicePreservation = 9,
        processingMode = 'balanced', // 'fast' or 'balanced'
        outputFormat = 'wav',
        denoisingStrength = '0.8'
      } = req.body;

      // Generate output path
      const outputFilename = `enhanced_${Date.now()}_${file.originalname}`;
      const outputPath = path.join('outputs', outputFilename);

      // Create audio job
      const jobData = {
        filename: file.originalname,
        originalFormat: path.extname(file.originalname).slice(1),
        outputFormat,
        fileSize: file.size,
        originalPath: file.path,
        noiseReductionLevel: parseInt(noiseReductionLevel),
        voicePreservation: parseInt(voicePreservation),
        processingMode, // Pass the actual processing mode (fast or balanced)
        processingOptions: { 
          denoisingStrength: parseFloat(denoisingStrength || '0.8'),
          processingMode // Also include in processing options
        },
        stage: 'upload', // Initial stage
        status: 'pending',
        progress: 0
      };

      const validatedData = insertAudioJobSchema.parse(jobData);
      const job = await storage.createAudioJob(validatedData);

      console.log(`✅ Job created: ${job.id}`);
      console.log(`📁 File path: ${file.path}`);
      console.log(`📁 Output path: ${outputPath}`);

      res.json({ 
        jobId: job.id, 
        status: 'uploaded', 
        message: 'Audio uploaded successfully. Processing with DCCRN...',
        job 
      });

      console.log(`🚀 Starting DCCRN processing...`);
      // Start DCCRN processing asynchronously with proper error handling
      processDCCRNAudio(job.id, file.path, outputPath, parseFloat(denoisingStrength || '0.8'), processingMode, broadcastJobUpdate)
        .catch(error => {
          console.error(`❌ DCCRN processing failed for job ${job.id}:`, error);
          broadcastJobUpdate(job.id, { 
            status: 'failed', 
            error: error.message,
            progress: 0 
          });
        });
      console.log(`🚀 DCCRN processing initiated`);

    } catch (error) {
      console.error('Upload error:', error);
      res.status(500).json({ error: 'Upload failed' });
    }
  });

  // Video URL processing endpoint
  app.post('/api/process-video-url', async (req, res) => {
    try {
      const { url, options = {} } = req.body;
      
      if (!url) {
        return res.status(400).json({ error: 'Video URL is required' });
      }

      console.log(`🎬 Processing video URL: ${url}`);
      console.log(`🔧 Options:`, options);

      // Extract processing options
      const {
        noiseReductionLevel = 7,
        voicePreservation = 9,
        processingMode = 'balanced',
        outputFormat = 'wav',
        denoisingStrength = '0.8'
      } = options;

      // Generate output paths
      const timestamp = Date.now();
      const extractedFilename = `extracted_${timestamp}_audio.wav`;
      const extractedPath = path.join('uploads', extractedFilename);
      const enhancedFilename = `enhanced_${timestamp}_${extractedFilename}`;
      const enhancedPath = path.join('outputs', enhancedFilename);
      
      // Create audio job
      const jobData = {
        filename: `video_${timestamp}.${outputFormat}`,
        originalFormat: 'video_url',
        outputFormat,
        fileSize: 0,
        originalPath: url,
        noiseReductionLevel: parseInt(noiseReductionLevel),
        voicePreservation: parseInt(voicePreservation),
        processingMode,
        processingOptions: { 
          denoisingStrength: parseFloat(denoisingStrength || '0.8'),
          processingMode,
          extractedPath,
          enhancedPath
        },
        stage: 'extraction',
        status: 'pending',
        progress: 0
      };

      const validatedData = insertAudioJobSchema.parse(jobData);
      const job = await storage.createAudioJob(validatedData);

      console.log(`✅ Video job created: ${job.id}`);

      res.json({ 
        jobId: job.id, 
        status: 'processing_video', 
        message: 'Video processing started...',
        job 
      });

      // Start video processing asynchronously
      processVideoUrl(job.id, url, extractedPath, enhancedPath, parseFloat(denoisingStrength || '0.8'), processingMode, broadcastJobUpdate)
        .catch(error => {
          console.error(`❌ Video processing failed for job ${job.id}:`, error);
          broadcastJobUpdate(job.id, { 
            status: 'failed', 
            error: error.message,
            progress: 0 
          });
        });
      console.log(`🚀 Video processing initiated`);

    } catch (error) {
      console.error('Video URL processing error:', error);
      res.status(500).json({ error: 'Failed to process video URL' });
    }
  });

  // Batch upload endpoint - Updated to use DCCRN
  app.post('/api/upload-batch', upload.array('files', 10), async (req, res) => {
    try {
      if (!req.files || req.files.length === 0) {
        return res.status(400).json({ error: 'No files uploaded' });
      }

      const options = req.body.options ? JSON.parse(req.body.options) : {};
      const jobs = [];

      for (const file of req.files as Express.Multer.File[]) {
        // Generate output path for each file
        const outputFilename = `enhanced_${Date.now()}_${file.originalname}`;
        const outputPath = path.join('outputs', outputFilename);

        const jobData = {
          filename: file.originalname,
          originalFormat: path.extname(file.originalname).slice(1),
          outputFormat: options.outputFormat || 'wav',
          fileSize: file.size,
          originalPath: file.path,
          noiseReductionLevel: options.noiseReductionLevel || 8,
          voicePreservation: options.voicePreservation || 9,
          processingMode: 'dccrn',
          processingOptions: { 
            denoisingStrength: parseFloat(options.denoisingStrength || '0.8')
          },
        };

        const validatedData = insertAudioJobSchema.parse(jobData);
        const job = await storage.createAudioJob(validatedData);
        jobs.push({ job, outputPath });
      }

      res.json({ jobs: jobs.map(({ job }) => ({ jobId: job.id, status: 'uploaded' })) });

      // Process batch asynchronously with DCCRN
      for (const { job, outputPath } of jobs) {
        processDCCRNAudio(
          job.id, 
          job.originalPath!, 
          outputPath, 
          parseFloat(options.denoisingStrength || '0.8'),
          options.processingMode || 'balanced',
          broadcastJobUpdate
        ).catch(error => {
          console.error(`❌ Batch DCCRN processing failed for job ${job.id}:`, error);
          broadcastJobUpdate(job.id, { 
            status: 'failed', 
            error: error.message,
            progress: 0 
          });
        });
      }

    } catch (error) {
      console.error('Batch upload error:', error);
      res.status(500).json({ error: 'Batch upload failed' });
    }
  });

  // Job status endpoint
  app.get('/api/jobs/:jobId', async (req, res) => {
    try {
      const job = await storage.getAudioJob(req.params.jobId);
      if (!job) {
        return res.status(404).json({ error: 'Job not found' });
      }
      
      console.log(`📊 Job status requested for ${req.params.jobId}:`, {
        status: job.status,
        progress: job.progress
      });
      
      res.json(job);
    } catch (error) {
      console.error('Error fetching job:', error);
      res.status(500).json({ error: 'Failed to fetch job' });
    }
  });

  // List jobs endpoint
  app.get('/api/jobs', async (req, res) => {
    try {
      const { userId } = req.query;
      const jobs = await storage.listAudioJobs(userId as string);
      res.json(jobs);
    } catch (error) {
      console.error('Error listing jobs:', error);
      res.status(500).json({ error: 'Failed to list jobs' });
    }
  });

  // Download processed file endpoint
  app.get('/api/download/:jobId', async (req, res) => {
    try {
      const { type } = req.query; // 'video' or 'audio' (default)
      const job = await storage.getAudioJob(req.params.jobId);
      
      if (!job) {
        return res.status(404).json({ error: 'Job not found' });
      }

      if (job.status !== 'completed' || !job.processedPath) {
        return res.status(400).json({ error: 'Job not completed or file not available' });
      }

      let filePath: string;
      let filename: string;
      let contentType: string;

      if (type === 'audio' && job.enhancedAudioPath) {
        // Download enhanced audio only
        filePath = job.enhancedAudioPath;
        filename = `${path.parse(job.filename).name}_enhanced_audio.wav`;
        contentType = 'audio/wav';
      } else if (job.isVideo) {
        // Download enhanced video (default for video jobs)
        filePath = job.processedPath;
        const originalExt = path.extname(job.filename);
        filename = `${path.parse(job.filename).name}_enhanced${originalExt}`;
        contentType = 'video/mp4';
      } else {
        // Download enhanced audio (default for audio jobs)
        filePath = job.processedPath;
        filename = `${path.parse(job.filename).name}_enhanced.${job.outputFormat}`;
        contentType = 'audio/wav';
      }

      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      res.setHeader('Content-Type', contentType);
      
      const fileStream = await fs.readFile(filePath);
      res.send(fileStream);

    } catch (error) {
      console.error('Download error:', error);
      res.status(500).json({ error: 'Download failed' });
    }
  });

  // Serve audio files for playback in the browser
  app.get('/api/audio/:jobId/original', async (req, res) => {
    try {
      const job = await storage.getAudioJob(req.params.jobId);
      
      if (!job || !job.originalPath) {
        return res.status(404).json({ error: 'Original audio not found' });
      }

      const filePath = job.originalPath;
      const stat = await fs.stat(filePath);
      
      res.setHeader('Content-Type', 'audio/wav');
      res.setHeader('Content-Length', stat.size.toString());
      res.setHeader('Accept-Ranges', 'bytes');
      
      const fileStream = await fs.readFile(filePath);
      res.send(fileStream);

    } catch (error) {
      console.error('Audio serving error:', error);
      res.status(500).json({ error: 'Failed to serve audio' });
    }
  });

  app.get('/api/audio/:jobId/processed', async (req, res) => {
    try {
      const job = await storage.getAudioJob(req.params.jobId);
      
      if (!job || !job.processedPath) {
        return res.status(404).json({ error: 'Processed audio not found' });
      }

      if (job.status !== 'completed') {
        return res.status(400).json({ error: 'Processing not completed yet' });
      }

      const filePath = job.processedPath;
      const stat = await fs.stat(filePath);
      
      res.setHeader('Content-Type', 'audio/wav');
      res.setHeader('Content-Length', stat.size.toString());
      res.setHeader('Accept-Ranges', 'bytes');
      
      const fileStream = await fs.readFile(filePath);
      res.send(fileStream);

    } catch (error) {
      console.error('Audio serving error:', error);
      res.status(500).json({ error: 'Failed to serve audio' });
    }
  });

  // Cancel job endpoint
  app.post('/api/jobs/:jobId/cancel', async (req, res) => {
    try {
      const jobId = req.params.jobId;
      const cancelled = audioProcessor.cancelJob(jobId);
      
      if (cancelled) {
        await storage.updateAudioJob(jobId, { 
          status: 'failed', 
          errorMessage: 'Cancelled by user' 
        });
        
        broadcastJobUpdate(jobId, { status: 'cancelled' });
        res.json({ success: true });
      } else {
        res.status(400).json({ error: 'Job not found or cannot be cancelled' });
      }
    } catch (error) {
      console.error('Cancel job error:', error);
      res.status(500).json({ error: 'Failed to cancel job' });
    }
  });

  // Noise sample upload endpoint
  app.post('/api/noise-samples', upload.single('sample'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
      }

      const { name, description, noiseType } = req.body;
      
      const sampleData = {
        name,
        description,
        filePath: req.file.path,
        noiseType,
      };

      const validatedData = insertNoiseSampleSchema.parse(sampleData);
      const sample = await storage.createNoiseSample(validatedData);

      // Generate noise profile using Groq
      try {
        const profile = await groqService.generateNoiseProfile(req.file.path);
        // Store profile in the processing options or separate table
      } catch (error) {
        console.error('Error generating noise profile:', error);
      }

      res.json(sample);

    } catch (error) {
      console.error('Noise sample upload error:', error);
      res.status(500).json({ error: 'Failed to upload noise sample' });
    }
  });

  // List noise samples endpoint
  app.get('/api/noise-samples', async (req, res) => {
    try {
      const { noiseType } = req.query;
      const samples = await storage.listNoiseSamples(noiseType as string);
      res.json(samples);
    } catch (error) {
      console.error('Error listing noise samples:', error);
      res.status(500).json({ error: 'Failed to list noise samples' });
    }
  });

  // Supported formats endpoint
  app.get('/api/supported-formats', async (req, res) => {
    try {
      const formats = await getSupportedFormats();
      res.json({ 
        audio: ['wav', 'mp3', 'flac', 'aac', 'ogg', 'm4a', 'wma', 'aiff', 'au'],
        video: ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'],
        ffmpeg_supported: formats
      });
    } catch (error) {
      console.error('Error getting supported formats:', error);
      res.status(500).json({ error: 'Failed to get supported formats' });
    }
  });

  // Audio analysis endpoint
  app.post('/api/analyze/:jobId', async (req, res) => {
    try {
      const job = await storage.getAudioJob(req.params.jobId);
      
      if (!job || !job.originalPath) {
        return res.status(404).json({ error: 'Job or file not found' });
      }

      const analysis = await groqService.analyzeAudioNoise(job.originalPath);
      res.json(analysis);

    } catch (error) {
      console.error('Analysis error:', error);
      res.status(500).json({ error: 'Analysis failed' });
    }
  });

  // Async processing function
  async function processAudioJob(jobId: string, broadcastUpdate: (jobId: string, data: any) => void) {
    try {
      const job = await storage.getAudioJob(jobId);
      if (!job) return;

      await storage.updateAudioJob(jobId, { 
        status: 'processing', 
        startedAt: new Date(),
        progress: 0 
      });

      broadcastUpdate(jobId, { status: 'processing', progress: 0 });

      const options = {
        noiseReductionLevel: job.noiseReductionLevel || 7,
        voicePreservation: job.voicePreservation || 9,
        processingMode: job.processingMode || 'balanced',
        preserveEmotions: true,
        contextAware: true,
      };

      const isVideo = /\.(mp4|avi|mov|mkv|webm|flv|wmv)$/i.test(job.filename);
      
      const result = isVideo 
        ? await audioProcessor.processVideoFile(jobId, job.originalPath!, options, (progress) => {
            storage.updateAudioJob(jobId, { progress: progress.progress });
            broadcastUpdate(jobId, progress);
          })
        : await audioProcessor.processAudioFile(jobId, job.originalPath!, options, (progress) => {
            storage.updateAudioJob(jobId, { progress: progress.progress });
            broadcastUpdate(jobId, progress);
          });

      if (result.success) {
        await storage.updateAudioJob(jobId, {
          status: 'completed',
          processedPath: result.outputPath,
          completedAt: new Date(),
          progress: 100
        });

        broadcastUpdate(jobId, { 
          status: 'completed', 
          progress: 100,
          downloadUrl: `/api/download/${jobId}`
        });
      } else {
        await storage.updateAudioJob(jobId, {
          status: 'failed',
          errorMessage: result.error,
          completedAt: new Date()
        });

        broadcastUpdate(jobId, { 
          status: 'failed', 
          error: result.error 
        });
      }

    } catch (error) {
      console.error(`Error processing job ${jobId}:`, error);
      
      await storage.updateAudioJob(jobId, {
        status: 'failed',
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
        completedAt: new Date()
      });

      broadcastUpdate(jobId, { 
        status: 'failed', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  }

  // DCCRN-specific processing function
  async function processDCCRNAudio(
    jobId: string, 
    inputPath: string, 
    outputPath: string, 
    strength: number,
    processingMode: string = 'balanced',
    broadcastUpdate: (jobId: string, data: any) => void
  ) {
    console.log(`🔄 Starting DCCRN processing for job ${jobId}`);
    console.log(`📁 Input: ${inputPath}`);
    console.log(`📁 Output: ${outputPath}`);
    console.log(`💪 Strength: ${strength}`);
    console.log(`⚡ Mode: ${processingMode.toUpperCase()}`);
    
    try {
      const job = await storage.getAudioJob(jobId);
      if (!job) {
        console.log(`❌ Job ${jobId} not found in database`);
        return;
      }

      console.log(`✅ Job found: ${job.filename}`);

      await storage.updateAudioJob(jobId, { 
        status: 'processing', 
        startedAt: new Date(),
        progress: 0
      });

      console.log(`📊 Job status updated to processing`);

      const modeDescription = processingMode === 'fast' ? 'Fast AI denoising' : 'Balanced AI denoising';
      broadcastUpdate(jobId, { 
        status: 'processing', 
        progress: 0,
        message: `Starting ${modeDescription}...`,
        stage: 'analysis'
      });

      console.log(`📡 Broadcasted initial progress update`);

      // Process with DCCRN
      console.log(`🧠 Starting ${processingMode.toUpperCase()} DCCRN processor...`);
      const result = await dccrnProcessor.enhanceAudio(inputPath, outputPath, { 
        strength, 
        processingMode: processingMode as 'fast' | 'balanced',
        noiseReductionLevel: Math.round(strength * 10)
      }, (progress: any) => {
        console.log(`📊 Progress: ${progress.progress}% - ${progress.message}`);
        
        // Map DCCRN stages to frontend stages
        let frontendStage = 'enhancement'; // default
        switch (progress.stage) {
          case 'initialization':
          case 'loading':
            frontendStage = 'analysis';
            break;
          case 'processing':
          case 'finalizing':
            frontendStage = 'enhancement';
            break;
          case 'completed':
            frontendStage = 'download';
            break;
        }
        
        storage.updateAudioJob(jobId, { 
          progress: progress.progress
        });
        broadcastUpdate(jobId, {
          status: 'processing',
          progress: progress.progress,
          message: progress.message,
          stage: frontendStage
        });
      });

      console.log(`🎯 DCCRN processing completed:`, result);

      if (result.success) {
        await storage.updateAudioJob(jobId, {
          status: 'completed',
          processedPath: result.outputPath,
          completedAt: new Date(),
          progress: 100,
          aiExplanation: result.aiExplanation
        });

        broadcastUpdate(jobId, {
          status: 'completed', 
          progress: 100,
          message: 'DCCRN enhancement completed!',
          stage: 'download',
          downloadUrl: `/api/download/${jobId}`,
          metadata: {
            originalSize: result.originalSize,
            enhancedSize: result.enhancedSize,
            duration: result.duration
          },
          aiExplanation: result.aiExplanation // Include the AI explanation
        });

        console.log(`✅ DCCRN processing completed successfully for job ${jobId}`);
        console.log(`🤖 AI Explanation included:`, result.aiExplanation ? 'YES' : 'NO');
      } else {
        await storage.updateAudioJob(jobId, {
          status: 'failed',
          errorMessage: result.error,
          completedAt: new Date()
        });

        broadcastUpdate(jobId, { 
          status: 'failed', 
          error: result.error 
        });
      }

    } catch (error) {
      console.error(`Error processing DCCRN job ${jobId}:`, error);
      
      await storage.updateAudioJob(jobId, {
        status: 'failed',
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
        completedAt: new Date()
      });

      broadcastUpdate(jobId, { 
        status: 'failed', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  }

  // Audio upload and processing with DCCRN
  app.post('/api/upload/audio', upload.single('audio'), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No audio file uploaded' });
      }

      const file = req.file;

      const {
        denoisingStrength = '1.0',
        outputFormat = 'wav'
      } = req.body;

      // Generate output path
      const outputFilename = `enhanced_${Date.now()}_${file.originalname}`;
      const outputPath = path.join('outputs', outputFilename);

      // Create job for tracking
      const jobData = {
        filename: file.originalname,
        originalFormat: path.extname(file.originalname).slice(1),
        outputFormat,
        fileSize: file.size,
        originalPath: file.path,
        noiseReductionLevel: Math.round(parseFloat(denoisingStrength) * 10),
        voicePreservation: 9,
        processingMode: 'dccrn',
        processingOptions: { denoisingStrength: parseFloat(denoisingStrength) },
      };

      const validatedData = insertAudioJobSchema.parse(jobData);
      const job = await storage.createAudioJob(validatedData);

      res.json({ 
        jobId: job.id, 
        status: 'uploaded', 
        message: 'Audio uploaded successfully. Processing with DCCRN...',
        job 
      });

      // Start DCCRN processing asynchronously with proper error handling
      processDCCRNAudio(job.id, file.path, outputPath, parseFloat(denoisingStrength), 'balanced', broadcastJobUpdate)
        .catch(error => {
          console.error(`❌ Video DCCRN processing failed for job ${job.id}:`, error);
          broadcastJobUpdate(job.id, { 
            status: 'failed', 
            error: error.message,
            progress: 0 
          });
        });

    } catch (error) {
      console.error('Audio upload error:', error);
      res.status(500).json({ error: 'Audio upload failed' });
    }
  });

  // Video upload and processing
  app.post('/api/upload/video', upload.single('video'), async (req, res) => {
    try {
      console.log('🎬 Video upload request received');
      console.log('📁 File object:', req.file ? 'YES' : 'NO');
      console.log('📝 Body:', req.body);

      if (!req.file) {
        return res.status(400).json({ error: 'No video file uploaded' });
      }

      const file = req.file;
      console.log(`📁 Uploaded file: ${file.originalname}, size: ${file.size} bytes`);

      const {
        denoisingStrength = '1.0',
        preserveVideoQuality = 'true',
        outputFormat = 'mp4',
        processingMode = 'balanced'
      } = req.body;

      console.log('✅ Starting video processing');

      // Create video processing job using the correct schema structure
      const jobData = {
        filename: file.originalname,
        originalFormat: path.extname(file.originalname).slice(1),
        outputFormat: 'mp4', // Always use mp4 for video output, not wav
        fileSize: file.size,
        originalPath: file.path,
        status: 'pending' as const,
        progress: 0,
        processingMode: processingMode as 'fast' | 'balanced',
        isVideo: true,
        stage: 'upload',
        noiseReductionLevel: Math.round(parseFloat(denoisingStrength) * 10),
        voicePreservation: 9,
        processingOptions: { 
          denoisingStrength: parseFloat(denoisingStrength),
          processingMode,
          preserveVideoQuality: preserveVideoQuality === 'true'
        }
      };

      console.log('📝 Creating job with data:', jobData);

      // Validate and create the job
      const validatedData = insertAudioJobSchema.parse(jobData);
      const job = await storage.createAudioJob(validatedData);

      console.log(`✅ Video job created successfully: ${job.id}`);

      // Send immediate response
      res.json({
        success: true,
        jobId: job.id,
        message: 'Video uploaded successfully. Processing started.',
        job
      });

      console.log('🚀 Starting video processing in background...');

      // Process video in background
      processVideoInBackground(job.id, file.path, {
        denoisingStrength: parseFloat(denoisingStrength),
        processingMode: processingMode as 'fast' | 'balanced',
        preserveVideoQuality: preserveVideoQuality === 'true',
        outputFormat: 'mp4', // Force MP4 for video output
        voicePreservation: 9,
        preserveEmotions: true,
        contextAware: true,
      }).catch(error => {
        console.error(`❌ Video processing failed for job ${job.id}:`, error);
      });

      console.log('🎬 Video upload endpoint completed successfully');

    } catch (error) {
      console.error('❌ Video upload error:', error);
      res.status(500).json({ error: 'Video upload failed', details: error instanceof Error ? error.message : 'Unknown error' });
    }
  });

  // Audio denoising endpoint (for existing audio files)
  app.post('/api/denoise', async (req, res) => {
    try {
      const { filePath, denoisingStrength = 1.0, outputPath } = req.body;

      if (!filePath) {
        return res.status(400).json({ error: 'File path is required' });
      }

      // Generate output path if not provided
      const finalOutputPath = outputPath || path.join('outputs', `denoised_${Date.now()}_${path.basename(filePath)}`);
      
      // Run DCCRN inference
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
      console.error('Denoising error:', error);
      res.status(500).json({ error: 'Denoising failed' });
    }
  });

  // Batch processing endpoint
  app.post('/api/denoise/batch', async (req, res) => {
    try {
      const { inputDir, outputDir, denoisingStrength = 1.0 } = req.body;

      if (!inputDir || !outputDir) {
        return res.status(400).json({ error: 'Input and output directories are required' });
      }

      // Start batch processing - placeholder implementation
      const result = { success: false, error: 'Batch processing not implemented' };
      
      if (result.success) {
        res.json({
          success: true,
          outputDir,
          message: 'Batch processing completed successfully'
        });
      } else {
        res.status(500).json({
          success: false,
          error: result.error
        });
      }

    } catch (error) {
      console.error('Batch processing error:', error);
      res.status(500).json({ error: 'Batch processing failed' });
    }
  });

  // Visualization endpoint - get spectrograms
  app.get('/api/visualize/:jobId', async (req, res) => {
    try {
      const { jobId } = req.params;
      
      // Get job details
      const job = await storage.getAudioJob(jobId);
      if (!job) {
        return res.status(404).json({ error: 'Job not found' });
      }

      // For now, return placeholder data
      // In a full implementation, you'd generate actual spectrograms
      const spectrograms = {
        noisy: {
          data: [], // Spectrogram data
          shape: [257, 100], // Frequency bins x Time frames
          sampleRate: 16000,
          hopLength: 256
        },
        enhanced: {
          data: [], // Enhanced spectrogram data
          shape: [257, 100],
          sampleRate: 16000,
          hopLength: 256
        }
      };

      res.json({
        jobId,
        spectrograms,
        metadata: {
          originalFilename: job.filename,
          processingMode: job.processingMode,
          denoisingStrength: (job.processingOptions as any)?.denoisingStrength || 1.0
        }
      });

    } catch (error) {
      console.error('Visualization error:', error);
      res.status(500).json({ error: 'Failed to generate visualization' });
    }
  });

  // Model status endpoint
  app.get('/api/model/status', async (req, res) => {
    try {
      const modelPath = path.join(process.cwd(), 'checkpoints', 'dccrn_latest.pth');
      const modelAvailable = await fs.access(modelPath).then(() => true).catch(() => false);
      // const ffmpegAvailable = await videoProcessingService.isFFmpegAvailable();
      const ffmpegAvailable = false; // Temporarily disabled
      
      res.json({
        dccrn: {
          available: modelAvailable,
          modelPath: modelAvailable ? 'checkpoints/dccrn_latest.pth' : null
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
      console.error('Model status error:', error);
      res.status(500).json({ error: 'Failed to check model status' });
    }
  });

  // File download endpoint for processed files
  app.get('/api/download/file', async (req, res) => {
    try {
      const { path: filePath } = req.query;

      if (!filePath || typeof filePath !== 'string') {
        return res.status(400).json({ error: 'File path is required' });
      }

      // Security check - ensure path is within outputs directory
      const normalizedPath = path.normalize(filePath);
      const outputsDir = path.resolve('outputs');
      const fullPath = path.resolve(normalizedPath);

      if (!fullPath.startsWith(outputsDir)) {
        return res.status(403).json({ error: 'Access denied' });
      }

      // Check if file exists
      try {
        await fs.access(fullPath);
      } catch {
        return res.status(404).json({ error: 'File not found' });
      }

      // Set appropriate headers
      const filename = path.basename(fullPath);
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      res.setHeader('Content-Type', 'application/octet-stream');

      // Stream file
      const fileStream = require('fs').createReadStream(fullPath);
      fileStream.pipe(res);

    } catch (error) {
      console.error('File download error:', error);
      res.status(500).json({ error: 'Download failed' });
    }
  });

  // Video processing function - temporarily disabled
  /*
  async function processVideoWithDCCRN(
    jobId: string, 
    inputPath: string, 
    outputPath: string, 
    denoisingStrength: number,
    preserveVideoQuality: boolean,
    broadcastUpdate: (jobId: string, data: any) => void
  ) {
    try {
      await storage.updateAudioJob(jobId, {
        status: 'processing',
        startedAt: new Date()
      });

      broadcastUpdate(jobId, { 
        status: 'processing', 
        progress: 10,
        message: 'Starting video processing...'
      });

      // Ensure output directory exists
      await fs.mkdir(path.dirname(outputPath), { recursive: true });

      broadcastUpdate(jobId, { 
        status: 'processing', 
        progress: 30,
        message: 'Processing video with DCCRN...'
      });

      // Process video
      const result = await videoProcessingService.processVideo({
        inputPath,
        outputPath,
        denoisingStrength,
        preserveVideoQuality
      });

      if (result.success) {
        await storage.updateAudioJob(jobId, {
          status: 'completed',
          processedPath: outputPath,
          completedAt: new Date()
        });

        broadcastUpdate(jobId, { 
          status: 'completed', 
          progress: 100,
          downloadUrl: `/api/download/file?path=${encodeURIComponent(outputPath)}`,
          message: 'Video processing completed successfully!',
          metadata: result.metadata
        });
      } else {
        await storage.updateAudioJob(jobId, {
          status: 'failed',
          errorMessage: result.error,
          completedAt: new Date()
        });

        broadcastUpdate(jobId, { 
          status: 'failed', 
          error: result.error 
        });
      }

    } catch (error) {
      console.error(`Error processing video job ${jobId}:`, error);
      
      await storage.updateAudioJob(jobId, {
        status: 'failed',
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
        completedAt: new Date()
      });

      broadcastUpdate(jobId, { 
        status: 'failed', 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  }
  */

  // Video URL processing function - Updated to use URLVideoProcessor
  async function processVideoUrl(
    jobId: string,
    videoUrl: string,
    extractedPath: string,
    enhancedPath: string,
    strength: number,
    processingMode: string = 'balanced',
    broadcastUpdate: (jobId: string, data: any) => void
  ) {
    console.log(`� Starting URL video processing for job ${jobId}`);
    console.log(`📹 URL: ${videoUrl}`);
    console.log(` Strength: ${strength}`);
    console.log(`⚡ Mode: ${processingMode.toUpperCase()}`);

    try {
      const job = await storage.getAudioJob(jobId);
      if (!job) {
        console.log(`❌ Job ${jobId} not found in database`);
        return;
      }

      console.log(`✅ Job found: ${job.filename}`);

      // Update job status to processing
      await storage.updateAudioJob(jobId, { 
        status: 'processing', 
        startedAt: new Date(),
        progress: 0
      });

      // Use URLVideoProcessor for complete pipeline
      const result = await urlVideoProcessor.processVideoFromUrl(
        videoUrl,
        {
          denoisingStrength: strength,
          processingMode: processingMode as 'fast' | 'balanced',
          quality: 'best'
        },
        (progress) => {
          console.log(`📊 URL Video Progress: ${progress.progress}% - ${progress.message}`);
          
          // Map stages to frontend
          let frontendStage = progress.stage;
          switch (progress.stage) {
            case 'download':
              frontendStage = 'download';
              break;
            case 'extraction':
              frontendStage = 'extraction';
              break;
            case 'enhancement':
              frontendStage = 'enhancement';
              break;
            case 'combination':
              frontendStage = 'combination';
              break;
            case 'finalization':
            case 'completed':
              frontendStage = 'download';
              break;
          }
          
          broadcastUpdate(jobId, {
            status: 'processing',
            progress: Math.round(progress.progress),
            message: progress.message,
            stage: frontendStage
          });
        }
      );

      if (!result.success) {
        throw new Error(`URL video processing failed: ${result.error}`);
      }

      console.log(`✅ URL video processing completed: ${result.outputPath}`);

      // Final update - mark as completed and update originalPath to extracted audio
      await storage.updateAudioJob(jobId, {
        status: 'completed',
        processedPath: result.outputPath!,
        originalPath: result.extractedAudioPath || extractedPath, // Update to point to extracted audio
        aiExplanation: result.aiExplanation, // Store AI explanation in database
        completedAt: new Date(),
        progress: 100
      });

      console.log(`🎉 URL video processing job ${jobId} completed successfully`);
      console.log(`🤖 AI Explanation included:`, result.aiExplanation ? 'YES' : 'NO');

      broadcastUpdate(jobId, { 
        status: 'completed',
        outputPath: result.outputPath,
        progress: 100,
        aiExplanation: result.aiExplanation, // Use the AI explanation from the result
        result: {
          originalSize: 0,
          enhancedSize: 0,
          processingMode,
          extractionMethod: 'yt-dlp',
          metadata: result.metadata
        }
      });

    } catch (error) {
      console.error(`❌ URL video processing error for job ${jobId}:`, error);
      
      await storage.updateAudioJob(jobId, {
        status: 'failed',
        errorMessage: error instanceof Error ? error.message : 'Unknown URL video processing error',
        completedAt: new Date()
      });

      broadcastUpdate(jobId, { 
        status: 'failed', 
        error: error instanceof Error ? error.message : 'Unknown URL video processing error',
        progress: 0 
      });
    }
  }

  return httpServer;
}

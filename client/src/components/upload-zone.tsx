import { useState, useRef, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { 
  Upload, 
  Video, 
  FileAudio, 
  Link,
  X,
  Music,
  Loader2
} from "lucide-react";

interface UploadZoneProps {
  onJobCreated: (jobId: string) => void;
}

interface FileWithPreview extends File {
  preview?: string;
}

export default function UploadZone({ onJobCreated }: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<FileWithPreview[]>([]);
  const [videoUrl, setVideoUrl] = useState("");
  const [processingOptions, setProcessingOptions] = useState({
    noiseReductionLevel: 7,
    voicePreservation: 9,
    processingMode: "balanced", // "fast" or "balanced"
    outputFormat: "wav",
    downloadType: "audio" // "audio" or "video" - for URL processing
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      // Detect if this is a video file
      const file = formData.get('audio') as File;
      const isVideoFile = file && isVideo(file.name);
      
      const endpoint = isVideoFile ? "/api/upload/video" : "/api/upload";
      const fieldName = isVideoFile ? "video" : "audio";
      
      // Rename the form field for video uploads
      if (isVideoFile) {
        formData.delete('audio');
        formData.append(fieldName, file);
      }
      
      const response = await apiRequest("POST", endpoint, formData);
      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Upload Successful",
        description: "Your file has been uploaded and processing will begin shortly.",
      });
      onJobCreated(data.jobId);
      setSelectedFiles([]);
    },
    onError: (error) => {
      toast({
        title: "Upload Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const videoUrlMutation = useMutation({
    mutationFn: async (data: { url: string; options: any }) => {
      const response = await apiRequest("POST", "/api/process-video-url", data);
      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Video Processing Started",
        description: "Audio extraction from video URL has begun.",
      });
      onJobCreated(data.jobId);
      setVideoUrl("");
    },
    onError: (error) => {
      toast({
        title: "Video Processing Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const batchUploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await apiRequest("POST", "/api/upload-batch", formData);
      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Batch Upload Successful",
        description: `${data.jobs.length} files uploaded and processing started.`,
      });
      // For batch uploads, we'll use the first job ID
      if (data.jobs.length > 0) {
        onJobCreated(data.jobs[0].jobId);
      }
      setSelectedFiles([]);
    },
    onError: (error) => {
      toast({
        title: "Batch Upload Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files) as FileWithPreview[];
    setSelectedFiles(prev => [...prev, ...files]);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files) as FileWithPreview[];
      setSelectedFiles(prev => [...prev, ...files]);
    }
  }, []);

  const removeFile = useCallback((index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  }, []);

  const handleUpload = useCallback(() => {
    if (selectedFiles.length === 0) return;

    const formData = new FormData();
    
    if (selectedFiles.length === 1) {
      // Single file upload
      formData.append("audio", selectedFiles[0]);
      Object.entries(processingOptions).forEach(([key, value]) => {
        formData.append(key, value.toString());
      });
      uploadMutation.mutate(formData);
    } else {
      // Batch upload
      selectedFiles.forEach(file => {
        formData.append("files", file);
      });
      formData.append("options", JSON.stringify(processingOptions));
      batchUploadMutation.mutate(formData);
    }
  }, [selectedFiles, processingOptions, uploadMutation, batchUploadMutation]);

  const handleVideoUrlSubmit = useCallback(() => {
    if (!videoUrl.trim()) return;

    videoUrlMutation.mutate({
      url: videoUrl,
      options: processingOptions
    });
  }, [videoUrl, processingOptions, videoUrlMutation]);

  const isLoading = uploadMutation.isPending || videoUrlMutation.isPending || batchUploadMutation.isPending;

  const getSupportedFormats = () => {
    return {
      audio: ["WAV", "MP3", "FLAC", "AAC", "OGG", "M4A", "WMA", "AIFF", "AU"],
      video: ["MP4", "AVI", "MOV", "MKV", "WebM", "FLV", "WMV"]
    };
  };

  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    const audioFormats = ['wav', 'mp3', 'flac', 'aac', 'ogg', 'm4a', 'wma', 'aiff', 'au'];
    const videoFormats = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'];
    
    if (audioFormats.includes(ext || '')) return <FileAudio className="text-neon-green" size={20} />;
    if (videoFormats.includes(ext || '')) return <Video className="text-electric-blue" size={20} />;
    return <Music className="text-gray-400" size={20} />;
  };

  const isVideo = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    const videoFormats = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'];
    return videoFormats.includes(ext || '');
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const supportedFormats = getSupportedFormats();

  return (
    <div className="space-y-8">
      {/* Main Upload Zone */}
      <Card 
        className={`glass-panel border-2 border-dashed transition-colors cursor-pointer ${
          isDragOver 
            ? 'border-neon-green bg-neon-green/10' 
            : 'border-neon-green/50 hover:border-neon-green'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        data-testid="upload-zone"
      >
        <CardContent className="p-8 text-center">
          <div className="w-20 h-20 bg-neon-green/20 rounded-full flex items-center justify-center mx-auto mb-6">
            <Upload className="text-neon-green" size={40} />
          </div>
          <h3 className="text-2xl font-semibold mb-4">Drop Your Audio Files Here</h3>
          <p className="text-gray-400 mb-6">
            Supports all major audio and video formats
          </p>
          
          {/* Format Badges */}
          <div className="space-y-3 mb-6">
            <div>
              <p className="text-sm text-gray-500 mb-2">Audio Formats:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {supportedFormats.audio.map(format => (
                  <Badge key={format} variant="outline" className="border-neon-green/30 text-neon-green">
                    {format}
                  </Badge>
                ))}
              </div>
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-2">Video Formats:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {supportedFormats.video.map(format => (
                  <Badge key={format} variant="outline" className="border-electric-blue/30 text-electric-blue">
                    {format}
                  </Badge>
                ))}
              </div>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              className="bg-neon-green text-dark-teal hover:bg-neon-green/90"
              disabled={isLoading}
              data-testid="button-browse-files"
            >
              {isLoading ? (
                <Loader2 className="mr-2 animate-spin" size={16} />
              ) : (
                <FileAudio className="mr-2" size={16} />
              )}
              Browse Files
            </Button>
            <Button 
              variant="outline" 
              className="border-electric-blue text-electric-blue hover:bg-electric-blue/10"
              disabled={isLoading}
              data-testid="button-extract-video"
            >
              <Video className="mr-2" size={16} />
              Extract from Video
            </Button>
          </div>
          
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".wav,.mp3,.flac,.aac,.ogg,.m4a,.wma,.aiff,.au,.mp4,.avi,.mov,.mkv,.webm,.flv,.wmv"
            onChange={handleFileSelect}
            className="hidden"
            data-testid="file-input"
          />
        </CardContent>
      </Card>

      {/* Selected Files */}
      {selectedFiles.length > 0 && (
        <Card className="glass-panel">
          <CardContent className="p-6">
            <h4 className="text-lg font-semibold mb-4 flex items-center">
              <FileAudio className="mr-2 text-neon-green" size={20} />
              Selected Files ({selectedFiles.length})
            </h4>
            <div className="space-y-3 mb-6">
              {selectedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    {getFileIcon(file.name)}
                    <div>
                      <p className="font-medium text-sm">{file.name}</p>
                      <p className="text-xs text-gray-400">{formatFileSize(file.size)}</p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(index)}
                    className="text-gray-400 hover:text-red-400"
                    data-testid={`button-remove-file-${index}`}
                  >
                    <X size={16} />
                  </Button>
                </div>
              ))}
            </div>
            <Button 
              onClick={handleUpload} 
              className="w-full bg-neon-green text-dark-teal hover:bg-neon-green/90"
              disabled={isLoading}
              data-testid="button-start-processing"
            >
              {isLoading ? (
                <Loader2 className="mr-2 animate-spin" size={16} />
              ) : (
                <Upload className="mr-2" size={16} />
              )}
              {selectedFiles.length === 1 ? 'Start Processing' : `Process ${selectedFiles.length} Files`}
            </Button>
          </CardContent>
        </Card>
      )}

      <Separator className="bg-border" />

      {/* Video URL Input */}
      <Card className="glass-panel">
        <CardContent className="p-6">
          <h4 className="text-lg font-semibold mb-4 flex items-center">
            <Link className="mr-2 text-electric-blue" size={20} />
            Extract Audio from Video URL
          </h4>
          <div className="flex items-center space-x-4">
            <Input
              type="text"
              placeholder="Paste video URL (YouTube, Vimeo, etc.) or upload video file..."
              value={videoUrl}
              onChange={(e) => setVideoUrl(e.target.value)}
              className="flex-1 bg-slate-800/50 border-gray-600 focus:border-electric-blue"
              data-testid="input-video-url"
            />
            <Button 
              onClick={handleVideoUrlSubmit}
              className="bg-electric-blue text-white hover:bg-electric-blue/90"
              disabled={!videoUrl.trim() || isLoading}
              data-testid="button-extract-audio"
            >
              {isLoading ? (
                <Loader2 className="mr-2 animate-spin" size={16} />
              ) : (
                <Video className="mr-2" size={16} />
              )}
              Extract Audio
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Processing Options */}
      <Card className="glass-panel">
        <CardContent className="p-6">
          <h4 className="text-lg font-semibold mb-4">Processing Options</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
            <div>
              <label className="block text-sm font-medium mb-2">Noise Reduction Level</label>
              <Input
                type="range"
                min="1"
                max="10"
                value={processingOptions.noiseReductionLevel}
                onChange={(e) => setProcessingOptions(prev => ({
                  ...prev,
                  noiseReductionLevel: parseInt(e.target.value)
                }))}
                className="w-full accent-neon-green"
                data-testid="slider-noise-reduction"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Gentle</span>
                <span className="text-neon-green font-medium">{processingOptions.noiseReductionLevel}</span>
                <span>Maximum</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Voice Preservation</label>
              <Input
                type="range"
                min="1"
                max="10"
                value={processingOptions.voicePreservation}
                onChange={(e) => setProcessingOptions(prev => ({
                  ...prev,
                  voicePreservation: parseInt(e.target.value)
                }))}
                className="w-full accent-electric-blue"
                data-testid="slider-voice-preservation"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>Basic</span>
                <span className="text-electric-blue font-medium">{processingOptions.voicePreservation}</span>
                <span>Max</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Processing Speed</label>
              <select 
                value={processingOptions.processingMode}
                onChange={(e) => setProcessingOptions(prev => ({
                  ...prev,
                  processingMode: e.target.value
                }))}
                className="w-full bg-slate-800/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-neon-green focus:outline-none"
                data-testid="select-processing-mode"
              >
                <option value="balanced">🎯 Balanced Quality (Recommended)</option>
                <option value="fast">⚡ Fast Processing (Quick Results)</option>
              </select>
              <p className="text-xs text-gray-400 mt-1">
                {processingOptions.processingMode === 'fast' 
                  ? 'Single-stage processing for quicker results' 
                  : '3-stage processing for optimal speech preservation'
                }
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Output Format</label>
              <select 
                value={processingOptions.outputFormat}
                onChange={(e) => setProcessingOptions(prev => ({
                  ...prev,
                  outputFormat: e.target.value
                }))}
                className="w-full bg-slate-800/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-neon-green focus:outline-none"
                data-testid="select-output-format"
              >
                <option value="wav">WAV (Uncompressed)</option>
                <option value="mp3">MP3 (320kbps)</option>
                <option value="flac">FLAC (Lossless)</option>
                <option value="aac">AAC (256kbps)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Download Type (For URLs)</label>
              <select 
                value={processingOptions.downloadType}
                onChange={(e) => setProcessingOptions(prev => ({
                  ...prev,
                  downloadType: e.target.value
                }))}
                className="w-full bg-slate-800/50 border border-gray-600 rounded-lg px-3 py-2 text-white focus:border-electric-blue focus:outline-none"
                data-testid="select-download-type"
              >
                <option value="audio">🎵 Audio Only (Enhanced Audio)</option>
                <option value="video">🎬 Video + Enhanced Audio</option>
              </select>
              <p className="text-xs text-gray-400 mt-1">
                {processingOptions.downloadType === 'audio' 
                  ? 'Download only the enhanced audio file' 
                  : 'Download video with enhanced audio merged back'
                }
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

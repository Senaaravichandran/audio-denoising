import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { useWebSocket } from "@/hooks/use-websocket";
import { 
  CheckCircle, 
  Brain, 
  Cog, 
  Download,
  Clock,
  TrendingUp,
  Volume2,
  Zap,
  X
} from "lucide-react";

interface ProcessingStatusProps {
  jobId: string;
  onComplete: () => void;
  onReset?: () => void; // Add reset callback
}

interface JobStatus {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  filename: string;
  stage?: 'analysis' | 'enhancement' | 'conversion' | 'completed';
  message?: string;
  analysisResult?: any;
  timeRemaining?: number;
}

export default function ProcessingStatus({ jobId, onComplete, onReset }: ProcessingStatusProps) {
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [stats, setStats] = useState({
    noiseReduction: 0,
    voiceClarity: 0,
    processingSpeed: 0
  });

  // Fetch initial job status
  const { data: initialJob } = useQuery<JobStatus>({
    queryKey: ['/api/jobs', jobId],
    refetchInterval: (query) => {
      // Stop refetching when job is completed
      return query.state.data?.status === 'completed' ? false : 1000;
    },
  });

  // Initialize job status when data is fetched
  useEffect(() => {
    if (initialJob && !jobStatus) {
      setJobStatus(initialJob);
    }
  }, [initialJob, jobStatus]);

  // WebSocket connection for real-time updates
  const { lastMessage, sendMessage } = useWebSocket('/ws');

  useEffect(() => {
    if (initialJob) {
      setJobStatus(initialJob as JobStatus);
    }
  }, [initialJob]);

  useEffect(() => {
    if (jobId) {
      // Subscribe to job updates
      sendMessage(JSON.stringify({ type: 'subscribe', jobId }));
    }
  }, [jobId, sendMessage]);

  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage.data);
        console.log('📡 WebSocket message received:', data);
        
        if (data.type === 'job_update' && data.jobId === jobId) {
          console.log('🔄 Updating job status:', data.data);
          setJobStatus(prev => prev ? { ...prev, ...data.data } : null);
          
          // Update stats if analysis result is available
          if (data.data.analysisResult) {
            const analysis = data.data.analysisResult;
            setStats({
              noiseReduction: Math.round((1 - analysis.noiseLevel) * 100),
              voiceClarity: Math.round(analysis.snrRatio),
              processingSpeed: 3.2 // Simulated speed
            });
          }
          
          // Call completion callback if job is completed
          if (data.data.status === 'completed') {
            console.log('🎉 Job completed, calling onComplete callback');
            setTimeout(() => {
              onComplete();
            }, 1000); // Small delay for visual effect
          }
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    }
  }, [lastMessage, jobId, onComplete]);

  if (!jobStatus) {
    return (
      <Card className="glass-panel">
        <CardContent className="p-8 text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-neon-green mx-auto mb-4"></div>
          <h3 className="text-xl font-semibold mb-2">Initializing Processing</h3>
          <p className="text-gray-400">Connecting to processing engine...</p>
          <div className="mt-4 flex items-center justify-center space-x-2">
            <Brain className="text-neon-green animate-pulse" size={16} />
            <span className="text-sm text-gray-500">AI Model Loading</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getStageIcon = (stage: string, isActive: boolean, isCompleted: boolean) => {
    const className = isCompleted ? "text-neon-green" : isActive ? "text-electric-blue" : "text-gray-400";
    const size = 24;
    
    switch (stage) {
      case 'upload':
        return <CheckCircle className={className} size={size} />;
      case 'analysis':
        return <Brain className={className} size={size} />;
      case 'enhancement':
        return <Cog className={className} size={size} />;
      case 'download':
        return <Download className={className} size={size} />;
      default:
        return <Clock className={className} size={size} />;
    }
  };

  const getStageStatus = (stage: string) => {
    if (!jobStatus.stage) return { isActive: false, isCompleted: false };
    
    const stages = ['upload', 'analysis', 'enhancement', 'download'];
    const currentIndex = stages.indexOf(jobStatus.stage);
    const stageIndex = stages.indexOf(stage);
    
    return {
      isActive: stageIndex === currentIndex,
      isCompleted: stageIndex < currentIndex || jobStatus.status === 'completed'
    };
  };

  const formatTimeRemaining = (seconds?: number) => {
    if (!seconds) return 'Calculating...';
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-8">
      {/* Processing Pipeline */}
      <Card className="glass-panel">
        <CardContent className="p-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {/* Upload Status */}
            <div className="text-center">
              <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 transition-all duration-300 ${
                getStageStatus('upload').isCompleted ? 'bg-neon-green/20 pulse-glow' : 'bg-gray-700'
              }`}>
                {getStageIcon('upload', getStageStatus('upload').isActive, getStageStatus('upload').isCompleted)}
              </div>
              <h4 className={`font-semibold transition-colors ${getStageStatus('upload').isCompleted ? 'text-neon-green' : 'text-gray-400'}`}>
                Upload Complete
              </h4>
              <p className="text-sm text-gray-400 mt-2" data-testid="text-filename">
                {jobStatus.filename}
              </p>
            </div>

            {/* Analysis Status */}
            <div className="text-center">
              <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 transition-all duration-300 ${
                getStageStatus('analysis').isActive ? 'bg-electric-blue/20 pulse-glow' : 
                getStageStatus('analysis').isCompleted ? 'bg-neon-green/20 pulse-glow' : 'bg-gray-700'
              }`}>
                {getStageStatus('analysis').isActive && (
                  <Brain className="text-electric-blue animate-pulse" size={24} />
                ) || getStageIcon('analysis', getStageStatus('analysis').isActive, getStageStatus('analysis').isCompleted)}
              </div>
              <h4 className={`font-semibold transition-colors ${
                getStageStatus('analysis').isCompleted ? 'text-neon-green' : 
                getStageStatus('analysis').isActive ? 'text-electric-blue' : 'text-gray-400'
              }`}>
                AI Analysis
              </h4>
              <p className="text-sm text-gray-400 mt-2">
                {getStageStatus('analysis').isCompleted ? 'Analysis complete' : 
                 getStageStatus('analysis').isActive ? 'Detecting noise patterns...' : 'Pending'}
              </p>
              {getStageStatus('analysis').isActive && (
                <div className="w-full bg-gray-700 rounded-full h-2 mt-3">
                  <div className="bg-electric-blue h-2 rounded-full transition-all duration-500" style={{ width: `${jobStatus.progress}%` }}></div>
                </div>
              )}
            </div>

            {/* Processing Status */}
            <div className="text-center">
              <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 transition-all duration-300 ${
                getStageStatus('enhancement').isActive ? 'bg-electric-blue/20 pulse-glow' : 
                getStageStatus('enhancement').isCompleted ? 'bg-neon-green/20 pulse-glow' : 'bg-gray-700'
              }`}>
                {getStageStatus('enhancement').isActive && (
                  <Cog className="text-electric-blue animate-spin" size={24} />
                ) || getStageIcon('enhancement', getStageStatus('enhancement').isActive, getStageStatus('enhancement').isCompleted)}
              </div>
              <h4 className={`font-semibold transition-colors ${
                getStageStatus('enhancement').isCompleted ? 'text-neon-green' : 
                getStageStatus('enhancement').isActive ? 'text-electric-blue' : 'text-gray-400'
              }`}>
                AI Enhancement
              </h4>
              <p className="text-sm text-gray-400 mt-2">
                {getStageStatus('enhancement').isCompleted ? 'Enhancement complete' : 
                 getStageStatus('enhancement').isActive ? jobStatus.message || 'Enhancing audio...' : 'Pending'}
              </p>
              {getStageStatus('enhancement').isActive && (
                <p className="text-xs text-gray-500 mt-1">
                  ETA: {formatTimeRemaining(jobStatus.timeRemaining)}
                </p>
              )}
            </div>

            {/* Download Status */}
            <div className="text-center">
              <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 transition-all duration-300 ${
                jobStatus.status === 'completed' ? 'bg-neon-green/20 pulse-glow' : 'bg-gray-700'
              }`}>
                {getStageIcon('download', jobStatus.status === 'completed', jobStatus.status === 'completed')}
              </div>
              <h4 className={`font-semibold transition-colors ${jobStatus.status === 'completed' ? 'text-neon-green' : 'text-gray-400'}`}>
                Ready for Download
              </h4>
              <p className="text-sm text-gray-400 mt-2">
                {jobStatus.status === 'completed' ? 'Download available' : 'Pending...'}
              </p>
            </div>
          </div>

          {/* Overall Progress Bar */}
          <div className="mt-8">
            <div className="flex justify-between items-center mb-3">
              <span className="text-sm font-medium text-gray-200">Overall Progress</span>
              <span className="text-sm font-semibold text-neon-green" data-testid="text-progress">{jobStatus.progress}%</span>
            </div>
            <Progress value={jobStatus.progress} className="h-4 bg-gray-700" />
            {jobStatus.message && (
              <p className="text-sm text-gray-400 mt-3 text-center" data-testid="text-status-message">{jobStatus.message}</p>
            )}
            
            {/* Current Stage Indicator */}
            <div className="mt-4 text-center">
              <p className="text-xs text-gray-500 uppercase tracking-wider">Current Stage</p>
              <p className="text-sm text-electric-blue font-medium mt-1">
                {jobStatus.stage === 'analysis' ? 'Analyzing Audio Patterns' : 
                 jobStatus.stage === 'enhancement' ? 'AI Enhancement in Progress' :
                 jobStatus.stage === 'download' ? 'Finalizing Enhanced Audio' : 
                 'Processing...'}
              </p>
            </div>
            
            {/* Download Button - Show when completed */}
            {jobStatus.status === 'completed' && (
              <div className="mt-6 text-center space-y-3">
                <Button 
                  onClick={() => window.open(`/api/download/${jobStatus.id}`, '_blank')}
                  className="bg-neon-green text-dark-teal hover:bg-neon-green/90 px-8 py-3 text-lg font-semibold mr-4"
                >
                  <Download className="mr-2" size={20} />
                  Download Enhanced Audio
                </Button>
                
                {onReset && (
                  <Button 
                    onClick={onReset}
                    variant="outline"
                    className="border-neon-green/30 text-neon-green hover:bg-neon-green/10 px-6 py-3"
                  >
                    Process Another Audio
                  </Button>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Live Processing Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="glass-panel text-center">
          <CardContent className="p-6">
            <Volume2 className="mx-auto mb-3 text-neon-green" size={32} />
            <h5 className="text-neon-green font-semibold">Noise Reduction</h5>
            <p className="text-2xl font-bold mt-2" data-testid="text-noise-reduction">{stats.noiseReduction}%</p>
            <Badge variant="outline" className="mt-2 border-neon-green/30 text-neon-green">
              Excellent
            </Badge>
          </CardContent>
        </Card>

        <Card className="glass-panel text-center">
          <CardContent className="p-6">
            <TrendingUp className="mx-auto mb-3 text-electric-blue" size={32} />
            <h5 className="text-electric-blue font-semibold">Voice Clarity</h5>
            <p className="text-2xl font-bold mt-2" data-testid="text-voice-clarity">+{stats.voiceClarity} dB</p>
            <Badge variant="outline" className="mt-2 border-electric-blue/30 text-electric-blue">
              Enhanced
            </Badge>
          </CardContent>
        </Card>

        <Card className="glass-panel text-center">
          <CardContent className="p-6">
            <Zap className="mx-auto mb-3 text-purple-400" size={32} />
            <h5 className="text-purple-400 font-semibold">Processing Speed</h5>
            <p className="text-2xl font-bold mt-2" data-testid="text-processing-speed">{stats.processingSpeed}x RT</p>
            <Badge variant="outline" className="mt-2 border-purple-400/30 text-purple-400">
              Real-time
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Error State */}
      {jobStatus.status === 'failed' && (
        <Card className="glass-panel border-red-400/30">
          <CardContent className="p-6 text-center">
            <div className="w-16 h-16 bg-red-400/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <X className="text-red-400" size={32} />
            </div>
            <h4 className="text-xl font-semibold text-red-400 mb-2">Processing Failed</h4>
            <p className="text-gray-400">{jobStatus.message || 'An error occurred during processing'}</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

import { useState } from "react";
import UploadZone from "@/components/upload-zone";
import AudioComparison from "@/components/audio-comparison";
import ProcessingStatus from "@/components/processing-status";
import AdvancedControls from "@/components/advanced-controls";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { 
  AudioWaveform, 
  Upload, 
  Video, 
  Settings, 
  Download,
  Brain,
  Zap,
  Shield,
  Cpu,
  CheckCircle
} from "lucide-react";

export default function Home() {
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [processingStage, setProcessingStage] = useState<'upload' | 'processing' | 'completed'>('upload');
  const [completedJobData, setCompletedJobData] = useState<any>(null);

  const handleJobCreated = (jobId: string) => {
    console.log('🚀 Job created:', jobId);
    setCurrentJobId(jobId);
    setProcessingStage('processing');
    setCompletedJobData(null); // Reset previous job data
  };

  const handleJobCompleted = () => {
    console.log('✅ Job completed, switching to completed stage');
    setProcessingStage('completed');
  };

  const handleReset = () => {
    console.log('🔄 Resetting to upload stage');
    setCurrentJobId(null);
    setProcessingStage('upload');
    setCompletedJobData(null);
  };

  return (
    <div className="min-h-screen relative">
      {/* Background with AI Neural Network Pattern */}
      <div className="fixed inset-0 z-0 bg-neural-network opacity-90">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900/80 via-slate-800/60 to-slate-900/80"></div>
      </div>

      {/* Header */}
      <header className="relative z-10 px-6 py-4 glass-panel border-b border-neon-green/30">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-neon-green rounded-lg flex items-center justify-center">
              <AudioWaveform className="text-dark-teal" size={24} />
            </div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-neon-green to-electric-blue bg-clip-text text-transparent">
              SonicPurge
            </h1>
          </div>
          <nav className="hidden md:flex space-x-6">
            <a href="#upload" className="text-gray-300 hover:text-neon-green transition-colors" data-testid="nav-upload">
              Upload
            </a>
            <a href="#process" className="text-gray-300 hover:text-neon-green transition-colors" data-testid="nav-process">
              Process
            </a>
            <a href="#compare" className="text-gray-300 hover:text-neon-green transition-colors" data-testid="nav-compare">
              Compare
            </a>
            <a href="#advanced" className="text-gray-300 hover:text-neon-green transition-colors" data-testid="nav-advanced">
              Advanced
            </a>
          </nav>
          <Button className="bg-neon-green text-dark-teal hover:bg-neon-green/90" data-testid="button-export">
            <Download className="mr-2" size={16} />
            Export
          </Button>
        </div>
      </header>

      {/* Hero Section - Always visible */}
      <section id="upload" className="relative z-10 px-6 py-16">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-5xl font-bold mb-6 leading-tight">
            Professional AI-Powered<br />
            <span className="bg-gradient-to-r from-neon-green to-electric-blue bg-clip-text text-transparent">
              Audio Enhancement
            </span>
          </h2>
          <p className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto">
            Transform your audio with cutting-edge machine learning. Remove noise, enhance clarity, 
            and extract audio from videos with professional-grade results in seconds.
          </p>

          {/* Feature Highlights */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
            <Card className="glass-panel border-neon-green/20 hover:border-neon-green/50 transition-colors">
              <CardContent className="p-6 text-center">
                <Brain className="mx-auto mb-4 text-neon-green" size={32} />
                <h3 className="font-semibold mb-2">AI-Powered</h3>
                <p className="text-sm text-gray-400">Advanced ML algorithms for superior noise reduction</p>
              </CardContent>
            </Card>
            <Card className="glass-panel border-electric-blue/20 hover:border-electric-blue/50 transition-colors">
              <CardContent className="p-6 text-center">
                <Zap className="mx-auto mb-4 text-electric-blue" size={32} />
                <h3 className="font-semibold mb-2">Ultra-Fast</h3>
                <p className="text-sm text-gray-400">Real-time processing with optimized performance</p>
              </CardContent>
            </Card>
            <Card className="glass-panel border-purple-400/20 hover:border-purple-400/50 transition-colors">
              <CardContent className="p-6 text-center">
                <Shield className="mx-auto mb-4 text-purple-400" size={32} />
                <h3 className="font-semibold mb-2">Voice Preservation</h3>
                <p className="text-sm text-gray-400">Maintains speech quality while removing noise</p>
              </CardContent>
            </Card>
            <Card className="glass-panel border-orange-400/20 hover:border-orange-400/50 transition-colors">
              <CardContent className="p-6 text-center">
                <Cpu className="mx-auto mb-4 text-orange-400" size={32} />
                <h3 className="font-semibold mb-2">Universal Support</h3>
                <p className="text-sm text-gray-400">All audio and video formats supported</p>
              </CardContent>
            </Card>
          </div>

          {/* Upload Component - Always visible */}
          {processingStage === 'upload' && <UploadZone onJobCreated={handleJobCreated} />}
          
          {/* Show upload success state when processing */}
          {(processingStage === 'processing' || processingStage === 'completed') && (
            <Card className="glass-panel border-neon-green/30 max-w-2xl mx-auto">
              <CardContent className="p-6 text-center">
                <div className="w-16 h-16 bg-neon-green/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <CheckCircle className="text-neon-green" size={32} />
                </div>
                <h3 className="text-xl font-semibold text-neon-green mb-2">Upload Successful!</h3>
                <p className="text-gray-400">Your audio is being processed with AI enhancement</p>
              </CardContent>
            </Card>
          )}
        </div>
      </section>

      {/* Processing Status Section - Shows when processing */}
      {currentJobId && processingStage === 'processing' && (
        <section id="process" className="relative z-10 px-6 py-12">
          <div className="absolute inset-0 bg-gradient-to-r from-slate-900/50 to-slate-800/30"></div>
          <div className="max-w-6xl mx-auto relative">
            <div className="text-center mb-8">
              <h3 className="text-3xl font-bold bg-gradient-to-r from-neon-green to-electric-blue bg-clip-text text-transparent mb-4">
                🤖 AI Processing in Progress
              </h3>
              <p className="text-gray-300">Watch your audio transform in real-time with our advanced DCCRN model</p>
            </div>
            <ProcessingStatus jobId={currentJobId} onComplete={handleJobCompleted} onReset={handleReset} />
          </div>
        </section>
      )}

      {/* Audio Comparison Section - Shows when completed */}
      {currentJobId && processingStage === 'completed' && (
        <section id="compare" className="relative z-10 px-6 py-16">
          <div className="absolute inset-0 bg-gradient-to-br from-green-900/20 to-blue-900/20"></div>
          <div className="max-w-7xl mx-auto relative">
            <div className="text-center mb-12">
              <h3 className="text-4xl font-bold bg-gradient-to-r from-neon-green to-electric-blue bg-clip-text text-transparent mb-4">
                ✨ Enhancement Complete!
              </h3>
              <p className="text-xl text-gray-300">Compare your original and enhanced audio files</p>
            </div>
            <AudioComparison jobId={currentJobId} />
            
            {/* Reset Button for completed section */}
            <div className="text-center mt-8">
              <Button 
                onClick={handleReset}
                variant="outline"
                className="border-neon-green/30 text-neon-green hover:bg-neon-green/10 px-8 py-3"
              >
                Process Another Audio File
              </Button>
            </div>
          </div>
        </section>
      )}

      {/* Advanced Features Section */}
      <section id="advanced" className="relative z-10 px-6 py-16 bg-frequency-spectrum">
        <div className="absolute inset-0 opacity-25">
          <div className="absolute inset-0 bg-gradient-to-r from-slate-900/80 to-slate-800/70"></div>
        </div>
        <div className="max-w-7xl mx-auto relative">
          <h3 className="text-4xl font-bold text-center mb-4">Advanced AI Features</h3>
          <p className="text-xl text-gray-300 text-center mb-12 max-w-3xl mx-auto">
            Harness the power of cutting-edge machine learning algorithms for professional-grade audio enhancement
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
            {/* Real-time Processing */}
            <Card className="glass-panel hover:bg-white/10 transition-colors">
              <CardContent className="p-6">
                <div className="w-16 h-16 bg-neon-green/20 rounded-xl flex items-center justify-center mb-6">
                  <Zap className="text-neon-green" size={32} />
                </div>
                <h4 className="text-xl font-semibold mb-4">Real-time Processing</h4>
                <p className="text-gray-400 mb-4">
                  Ultra-fast noise reduction for live audio streams with zero-latency AI processing.
                </p>
                <div className="space-y-2">
                  <Badge variant="outline" className="border-neon-green/30 text-neon-green">
                    Live microphone enhancement
                  </Badge>
                  <Badge variant="outline" className="border-neon-green/30 text-neon-green">
                    Video call optimization
                  </Badge>
                  <Badge variant="outline" className="border-neon-green/30 text-neon-green">
                    Streaming audio cleanup
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Batch Processing */}
            <Card className="glass-panel hover:bg-white/10 transition-colors">
              <CardContent className="p-6">
                <div className="w-16 h-16 bg-electric-blue/20 rounded-xl flex items-center justify-center mb-6">
                  <Upload className="text-electric-blue" size={32} />
                </div>
                <h4 className="text-xl font-semibold mb-4">Batch Processing</h4>
                <p className="text-gray-400 mb-4">
                  Process multiple files simultaneously with intelligent queue management.
                </p>
                <div className="space-y-2">
                  <Badge variant="outline" className="border-electric-blue/30 text-electric-blue">
                    Drag & drop multiple files
                  </Badge>
                  <Badge variant="outline" className="border-electric-blue/30 text-electric-blue">
                    Priority queue system
                  </Badge>
                  <Badge variant="outline" className="border-electric-blue/30 text-electric-blue">
                    Progress tracking
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Voice Isolation */}
            <Card className="glass-panel hover:bg-white/10 transition-colors">
              <CardContent className="p-6">
                <div className="w-16 h-16 bg-purple-500/20 rounded-xl flex items-center justify-center mb-6">
                  <Brain className="text-purple-400" size={32} />
                </div>
                <h4 className="text-xl font-semibold mb-4">Voice Isolation</h4>
                <p className="text-gray-400 mb-4">
                  Advanced speaker separation and voice focus technology for multi-speaker recordings.
                </p>
                <div className="space-y-2">
                  <Badge variant="outline" className="border-purple-400/30 text-purple-400">
                    Multi-speaker separation
                  </Badge>
                  <Badge variant="outline" className="border-purple-400/30 text-purple-400">
                    Voice focus enhancement
                  </Badge>
                  <Badge variant="outline" className="border-purple-400/30 text-purple-400">
                    Meeting optimization
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Video Audio Extraction */}
            <Card className="glass-panel hover:bg-white/10 transition-colors">
              <CardContent className="p-6">
                <div className="w-16 h-16 bg-yellow-500/20 rounded-xl flex items-center justify-center mb-6">
                  <Video className="text-yellow-400" size={32} />
                </div>
                <h4 className="text-xl font-semibold mb-4">Video Audio Extraction</h4>
                <p className="text-gray-400 mb-4">
                  Extract and enhance audio from any video format with automatic processing.
                </p>
                <div className="space-y-2">
                  <Badge variant="outline" className="border-yellow-400/30 text-yellow-400">
                    Universal video support
                  </Badge>
                  <Badge variant="outline" className="border-yellow-400/30 text-yellow-400">
                    URL processing
                  </Badge>
                  <Badge variant="outline" className="border-yellow-400/30 text-yellow-400">
                    Auto enhancement
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Long File Support */}
            <Card className="glass-panel hover:bg-white/10 transition-colors">
              <CardContent className="p-6">
                <div className="w-16 h-16 bg-orange-500/20 rounded-xl flex items-center justify-center mb-6">
                  <Settings className="text-orange-400" size={32} />
                </div>
                <h4 className="text-xl font-semibold mb-4">Long Audio Support</h4>
                <p className="text-gray-400 mb-4">
                  Handle files up to several hours with intelligent chunked processing.
                </p>
                <div className="space-y-2">
                  <Badge variant="outline" className="border-orange-400/30 text-orange-400">
                    Progressive processing
                  </Badge>
                  <Badge variant="outline" className="border-orange-400/30 text-orange-400">
                    Memory optimization
                  </Badge>
                  <Badge variant="outline" className="border-orange-400/30 text-orange-400">
                    Resume capability
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Custom Learning */}
            <Card className="glass-panel hover:bg-white/10 transition-colors">
              <CardContent className="p-6">
                <div className="w-16 h-16 bg-pink-500/20 rounded-xl flex items-center justify-center mb-6">
                  <Brain className="text-pink-400" size={32} />
                </div>
                <h4 className="text-xl font-semibold mb-4">Custom Learning</h4>
                <p className="text-gray-400 mb-4">
                  Train the AI on your specific noise samples for personalized enhancement.
                </p>
                <div className="space-y-2">
                  <Badge variant="outline" className="border-pink-400/30 text-pink-400">
                    Upload noise samples
                  </Badge>
                  <Badge variant="outline" className="border-pink-400/30 text-pink-400">
                    AI model fine-tuning
                  </Badge>
                  <Badge variant="outline" className="border-pink-400/30 text-pink-400">
                    Personalized profiles
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>

          <Separator className="my-12 bg-border" />

          {/* Advanced Controls */}
          <AdvancedControls />
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 px-6 py-12 border-t border-gray-800">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-8 h-8 bg-neon-green rounded-lg flex items-center justify-center">
                  <AudioWaveform className="text-dark-teal" size={16} />
                </div>
                <span className="text-xl font-bold">SonicPurge</span>
              </div>
              <p className="text-gray-400 text-sm">
                Professional AI-powered audio enhancement for creators, studios, and professionals worldwide.
              </p>
            </div>
            <div>
              <h5 className="font-semibold mb-4">Features</h5>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>Real-time Processing</li>
                <li>Batch Upload</li>
                <li>Voice Isolation</li>
                <li>Video Audio Extraction</li>
              </ul>
            </div>
            <div>
              <h5 className="font-semibold mb-4">Support</h5>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>Documentation</li>
                <li>API Reference</li>
                <li>Contact Support</li>
                <li>System Status</li>
              </ul>
            </div>
            <div>
              <h5 className="font-semibold mb-4">Connect</h5>
              <div className="flex space-x-4">
                <Button variant="outline" size="icon" className="border-gray-700 hover:border-neon-green hover:text-neon-green">
                  <span className="sr-only">Twitter</span>
                  X
                </Button>
                <Button variant="outline" size="icon" className="border-gray-700 hover:border-neon-green hover:text-neon-green">
                  <span className="sr-only">GitHub</span>
                  GH
                </Button>
                <Button variant="outline" size="icon" className="border-gray-700 hover:border-neon-green hover:text-neon-green">
                  <span className="sr-only">Discord</span>
                  DC
                </Button>
              </div>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-sm text-gray-400">
            <p>&copy; 2024 SonicPurge. Powered by advanced machine learning technology.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

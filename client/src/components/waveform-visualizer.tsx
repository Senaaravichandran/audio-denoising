import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface WaveformVisualizerProps {
  audioUrl?: string;
  color?: string;
  height?: number;
  isActive?: boolean;
  currentTime?: number;
  duration?: number;
  onSeek?: (time: number) => void;
  className?: string;
  'data-testid'?: string;
}

export default function WaveformVisualizer({
  audioUrl,
  color = "#00E68A",
  height = 128,
  isActive = false,
  currentTime = 0,
  duration = 0,
  onSeek,
  className,
  'data-testid': testId,
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Generate mock waveform data for demonstration
  // In a real app, this would use Web Audio API to analyze actual audio
  useEffect(() => {
    if (audioUrl) {
      setIsLoading(true);
      // Simulate loading time
      setTimeout(() => {
        // Generate realistic waveform data
        const samples = 200;
        const data = Array.from({ length: samples }, (_, i) => {
          // Create a more realistic waveform pattern
          const baseAmplitude = Math.sin(i * 0.1) * 0.3;
          const noise = (Math.random() - 0.5) * 0.4;
          const envelope = Math.exp(-Math.abs(i - samples / 2) / (samples / 4));
          return Math.max(0, Math.min(1, baseAmplitude + noise * envelope + 0.1));
        });
        setWaveformData(data);
        setIsLoading(false);
      }, 500);
    } else {
      // Generate default waveform for placeholder
      const samples = 200;
      const data = Array.from({ length: samples }, () => Math.random() * 0.8 + 0.1);
      setWaveformData(data);
    }
  }, [audioUrl]);

  // Draw waveform on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || waveformData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height: canvasHeight } = canvas;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, canvasHeight);

    // Calculate bar width
    const barWidth = width / waveformData.length;
    const centerY = canvasHeight / 2;

    // Calculate progress position
    const progressX = duration > 0 ? (currentTime / duration) * width : 0;

    // Draw waveform bars
    waveformData.forEach((amplitude, index) => {
      const x = index * barWidth;
      const barHeight = amplitude * centerY * 0.9;
      
      // Determine color based on progress
      const isPastProgress = x <= progressX;
      ctx.fillStyle = isPastProgress 
        ? color 
        : color + '40'; // Add transparency for unplayed portions

      // Draw upper and lower bars (mirrored)
      ctx.fillRect(x, centerY - barHeight, barWidth - 1, barHeight);
      ctx.fillRect(x, centerY, barWidth - 1, barHeight);
    });

    // Draw progress line
    if (duration > 0) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(progressX, 0);
      ctx.lineTo(progressX, canvasHeight);
      ctx.stroke();
    }

    // Add pulsing effect if active
    if (isActive) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.globalCompositeOperation = 'lighter';
      
      // Redraw progress area with glow
      waveformData.forEach((amplitude, index) => {
        const x = index * barWidth;
        const barHeight = amplitude * centerY * 0.9;
        
        if (x <= progressX) {
          ctx.fillStyle = color + '80';
          ctx.fillRect(x, centerY - barHeight, barWidth - 1, barHeight);
          ctx.fillRect(x, centerY, barWidth - 1, barHeight);
        }
      });
      
      ctx.globalCompositeOperation = 'source-over';
      ctx.shadowBlur = 0;
    }
  }, [waveformData, color, currentTime, duration, isActive]);

  // Handle canvas click for seeking
  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek || duration === 0) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const clickPosition = x / canvas.width;
    const seekTime = clickPosition * duration;
    
    onSeek(seekTime);
  };

  // Resize canvas to match container
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const resizeCanvas = () => {
      const { width } = container.getBoundingClientRect();
      canvas.width = width;
      canvas.height = height;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    return () => window.removeEventListener('resize', resizeCanvas);
  }, [height]);

  if (isLoading) {
    return (
      <div 
        ref={containerRef}
        className={cn("relative bg-slate-900 rounded-lg flex items-center justify-center", className)}
        style={{ height }}
        data-testid={testId}
      >
        <div className="flex items-center space-x-2 text-gray-400">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-current"></div>
          <span className="text-sm">Analyzing audio...</span>
        </div>
      </div>
    );
  }

  return (
    <div 
      ref={containerRef}
      className={cn("relative bg-slate-900 rounded-lg overflow-hidden", className)}
      style={{ height }}
      data-testid={testId}
    >
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="w-full h-full cursor-pointer"
        style={{ height }}
      />
      
      {/* Overlay gradient effect */}
      <div 
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `linear-gradient(90deg, transparent 0%, ${color}20 50%, transparent 100%)`,
          opacity: isActive ? 0.6 : 0.2,
          transition: 'opacity 0.3s ease'
        }}
      />
      
      {/* Time markers */}
      {duration > 0 && (
        <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-gray-500 px-2 py-1">
          <span>0:00</span>
          <span>
            {Math.floor(duration / 60)}:{(duration % 60).toFixed(0).padStart(2, '0')}
          </span>
        </div>
      )}
    </div>
  );
}

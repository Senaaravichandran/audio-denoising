import { useState, useRef, useCallback, useEffect } from 'react';

interface UseAudioPlayerOptions {
  autoPlay?: boolean;
  loop?: boolean;
  volume?: number;
  onEnded?: () => void;
  onTimeUpdate?: (currentTime: number) => void;
  onDurationChange?: (duration: number) => void;
  onPlay?: () => void;
  onPause?: () => void;
  onLoadStart?: () => void;
  onLoadedData?: () => void;
  onError?: (error: Event) => void;
}

export function useAudioPlayer(options: UseAudioPlayerOptions = {}) {
  const {
    autoPlay = false,
    loop = false,
    volume = 1,
    onEnded,
    onTimeUpdate,
    onDurationChange,
    onPlay,
    onPause,
    onLoadStart,
    onLoadedData,
    onError,
  } = options;

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<Event | null>(null);
  const [isMuted, setIsMuted] = useState(false);
  const [currentVolume, setCurrentVolume] = useState(volume);

  // Initialize audio element
  useEffect(() => {
    const audio = new Audio();
    audio.autoplay = autoPlay;
    audio.loop = loop;
    audio.volume = volume;
    audioRef.current = audio;

    // Event listeners
    const handlePlay = () => {
      setIsPlaying(true);
      setIsPaused(false);
      onPlay?.();
    };

    const handlePause = () => {
      setIsPlaying(false);
      setIsPaused(true);
      onPause?.();
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setIsPaused(false);
      onEnded?.();
    };

    const handleTimeUpdate = () => {
      const time = audio.currentTime;
      setCurrentTime(time);
      onTimeUpdate?.(time);
    };

    const handleDurationChange = () => {
      const dur = audio.duration;
      setDuration(isNaN(dur) ? 0 : dur);
      onDurationChange?.(dur);
    };

    const handleLoadStart = () => {
      setIsLoading(true);
      setError(null);
      onLoadStart?.();
    };

    const handleLoadedData = () => {
      setIsLoading(false);
      setIsLoaded(true);
      onLoadedData?.();
    };

    const handleError = (event: Event) => {
      setIsLoading(false);
      setError(event);
      onError?.(event);
    };

    const handleVolumeChange = () => {
      setCurrentVolume(audio.volume);
      setIsMuted(audio.muted);
    };

    // Attach event listeners
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('durationchange', handleDurationChange);
    audio.addEventListener('loadstart', handleLoadStart);
    audio.addEventListener('loadeddata', handleLoadedData);
    audio.addEventListener('error', handleError);
    audio.addEventListener('volumechange', handleVolumeChange);

    return () => {
      // Cleanup event listeners
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('durationchange', handleDurationChange);
      audio.removeEventListener('loadstart', handleLoadStart);
      audio.removeEventListener('loadeddata', handleLoadedData);
      audio.removeEventListener('error', handleError);
      audio.removeEventListener('volumechange', handleVolumeChange);

      // Cleanup audio element
      audio.pause();
      audio.src = '';
      audio.load();
    };
  }, []);

  const load = useCallback((src: string) => {
    if (audioRef.current) {
      audioRef.current.src = src;
      audioRef.current.load();
      setIsLoaded(false);
      setCurrentTime(0);
      setDuration(0);
    }
  }, []);

  const play = useCallback(async () => {
    if (audioRef.current) {
      try {
        await audioRef.current.play();
      } catch (err) {
        console.error('Failed to play audio:', err);
      }
    }
  }, []);

  const pause = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
    }
  }, []);

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  }, []);

  const setCurrentTimeHandler = useCallback((time: number) => {
    if (audioRef.current && isFinite(time)) {
      audioRef.current.currentTime = Math.max(0, Math.min(time, duration));
    }
  }, [duration]);

  const setVolume = useCallback((vol: number) => {
    if (audioRef.current) {
      const clampedVolume = Math.max(0, Math.min(1, vol));
      audioRef.current.volume = clampedVolume;
    }
  }, []);

  const toggleMute = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.muted = !audioRef.current.muted;
    }
  }, []);

  const mute = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.muted = true;
    }
  }, []);

  const unmute = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.muted = false;
    }
  }, []);

  const seek = useCallback((percentage: number) => {
    if (audioRef.current && duration > 0) {
      const time = (percentage / 100) * duration;
      setCurrentTimeHandler(time);
    }
  }, [duration, setCurrentTimeHandler]);

  const skipForward = useCallback((seconds: number = 10) => {
    if (audioRef.current) {
      const newTime = Math.min(currentTime + seconds, duration);
      setCurrentTimeHandler(newTime);
    }
  }, [currentTime, duration, setCurrentTimeHandler]);

  const skipBackward = useCallback((seconds: number = 10) => {
    if (audioRef.current) {
      const newTime = Math.max(currentTime - seconds, 0);
      setCurrentTimeHandler(newTime);
    }
  }, [currentTime, setCurrentTimeHandler]);

  const getCurrentPercentage = useCallback(() => {
    return duration > 0 ? (currentTime / duration) * 100 : 0;
  }, [currentTime, duration]);

  const getFormattedTime = useCallback((time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, []);

  return {
    // Audio element ref
    audioRef,
    
    // State
    isPlaying,
    isPaused,
    isLoading,
    isLoaded,
    duration,
    currentTime,
    error,
    isMuted,
    volume: currentVolume,
    
    // Controls
    load,
    play,
    pause,
    stop,
    setCurrentTime: setCurrentTimeHandler,
    setVolume,
    toggleMute,
    mute,
    unmute,
    seek,
    skipForward,
    skipBackward,
    
    // Utilities
    getCurrentPercentage,
    getFormattedTime,
    
    // Computed properties
    isEnded: !isPlaying && !isPaused && currentTime === duration && duration > 0,
    hasError: error !== null,
    progressPercentage: getCurrentPercentage(),
    formattedCurrentTime: getFormattedTime(currentTime),
    formattedDuration: getFormattedTime(duration),
  };
}

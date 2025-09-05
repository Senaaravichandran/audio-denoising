import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { 
  Settings, 
  Upload, 
  Brain, 
  Sliders,
  FileAudio,
  Loader2,
  Save,
  RotateCcw,
  Mic,
  Music,
  MessageSquare,
  Users
} from "lucide-react";

interface NoiseProfile {
  id: string;
  name: string;
  noiseType: string;
  description?: string;
}

export default function AdvancedControls() {
  const [settings, setSettings] = useState({
    // Noise Reduction Settings
    noiseReductionLevel: 7,
    voicePreservation: 9,
    processingMode: 'balanced',
    outputFormat: 'wav',
    
    // Advanced Options
    adaptiveProcessing: true,
    emotionPreservation: true,
    contextAwareProcessing: true,
    dynamicNoiseHandling: true,
    multiSpeakerMode: false,
    realTimeMode: false,
    
    // Quality Settings
    sampleRate: 44100,
    bitDepth: 16,
    channels: 2,
    
    // Environment Profiles
    environmentProfile: 'auto',
    customNoiseProfile: '',
  });

  const [noiseProfiles, setNoiseProfiles] = useState<NoiseProfile[]>([
    { id: '1', name: 'Office Environment', noiseType: 'office', description: 'Keyboard typing, air conditioning, conversations' },
    { id: '2', name: 'Outdoor Recording', noiseType: 'outdoor', description: 'Wind, traffic, environmental sounds' },
    { id: '3', name: 'Home Studio', noiseType: 'home', description: 'Room tone, appliances, neighbors' },
    { id: '4', name: 'Meeting Room', noiseType: 'meeting', description: 'HVAC, projector fan, multiple speakers' },
  ]);

  const [isUploading, setIsUploading] = useState(false);
  const { toast } = useToast();

  // Mutation for uploading noise samples
  const uploadNoiseSample = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await apiRequest("POST", "/api/noise-samples", formData);
      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Noise Sample Uploaded",
        description: "Your custom noise profile has been created successfully.",
      });
      setNoiseProfiles(prev => [...prev, data]);
    },
    onError: (error) => {
      toast({
        title: "Upload Failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleNoiseProfileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('sample', file);
    formData.append('name', `Custom Profile - ${file.name}`);
    formData.append('description', 'User uploaded noise sample');
    formData.append('noiseType', 'custom');

    uploadNoiseSample.mutate(formData);
    setIsUploading(false);
  };

  const handleSettingChange = (key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const resetToDefaults = () => {
    setSettings({
      noiseReductionLevel: 7,
      voicePreservation: 9,
      processingMode: 'balanced',
      outputFormat: 'wav',
      adaptiveProcessing: true,
      emotionPreservation: true,
      contextAwareProcessing: true,
      dynamicNoiseHandling: true,
      multiSpeakerMode: false,
      realTimeMode: false,
      sampleRate: 44100,
      bitDepth: 16,
      channels: 2,
      environmentProfile: 'auto',
      customNoiseProfile: '',
    });
    toast({
      title: "Settings Reset",
      description: "All settings have been reset to default values.",
    });
  };

  const saveSettings = () => {
    // In a real app, this would save to backend
    localStorage.setItem('audioEnhancementSettings', JSON.stringify(settings));
    toast({
      title: "Settings Saved",
      description: "Your preferences have been saved successfully.",
    });
  };

  return (
    <Card className="glass-panel">
      <CardContent className="p-8">
        <div className="flex items-center justify-between mb-8">
          <h4 className="text-2xl font-semibold flex items-center">
            <Settings className="mr-3 text-neon-green" size={28} />
            AI Enhancement Controls
          </h4>
          <div className="flex space-x-2">
            <Button
              variant="outline"
              onClick={resetToDefaults}
              className="border-gray-600 hover:border-orange-400 hover:text-orange-400"
              data-testid="button-reset-settings"
            >
              <RotateCcw className="mr-2" size={16} />
              Reset
            </Button>
            <Button
              onClick={saveSettings}
              className="bg-neon-green text-dark-teal hover:bg-neon-green/90"
              data-testid="button-save-settings"
            >
              <Save className="mr-2" size={16} />
              Save Settings
            </Button>
          </div>
        </div>

        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-slate-800">
            <TabsTrigger value="basic" className="data-[state=active]:bg-neon-green data-[state=active]:text-dark-teal">
              Basic
            </TabsTrigger>
            <TabsTrigger value="advanced" className="data-[state=active]:bg-neon-green data-[state=active]:text-dark-teal">
              Advanced
            </TabsTrigger>
            <TabsTrigger value="profiles" className="data-[state=active]:bg-neon-green data-[state=active]:text-dark-teal">
              Profiles
            </TabsTrigger>
            <TabsTrigger value="output" className="data-[state=active]:bg-neon-green data-[state=active]:text-dark-teal">
              Output
            </TabsTrigger>
          </TabsList>

          {/* Basic Settings */}
          <TabsContent value="basic" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="noise-reduction">Noise Reduction Intensity</Label>
                  <div className="mt-2">
                    <Input
                      id="noise-reduction"
                      type="range"
                      min="1"
                      max="10"
                      value={settings.noiseReductionLevel}
                      onChange={(e) => handleSettingChange('noiseReductionLevel', parseInt(e.target.value))}
                      className="w-full accent-neon-green"
                      data-testid="slider-noise-reduction-advanced"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>Gentle</span>
                      <span className="text-neon-green font-medium">{settings.noiseReductionLevel}</span>
                      <span>Maximum</span>
                    </div>
                  </div>
                </div>

                <div>
                  <Label htmlFor="voice-preservation">Voice Preservation</Label>
                  <div className="mt-2">
                    <Input
                      id="voice-preservation"
                      type="range"
                      min="1"
                      max="10"
                      value={settings.voicePreservation}
                      onChange={(e) => handleSettingChange('voicePreservation', parseInt(e.target.value))}
                      className="w-full accent-electric-blue"
                      data-testid="slider-voice-preservation-advanced"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>Basic</span>
                      <span className="text-electric-blue font-medium">{settings.voicePreservation}</span>
                      <span>Maximum</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <Label>Processing Mode</Label>
                  <Select 
                    value={settings.processingMode} 
                    onValueChange={(value) => handleSettingChange('processingMode', value)}
                  >
                    <SelectTrigger className="bg-slate-800 border-gray-600" data-testid="select-processing-mode-advanced">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-gray-600">
                      <SelectItem value="balanced">
                        <div className="flex items-center">
                          <Sliders className="mr-2" size={16} />
                          Balanced (Default)
                        </div>
                      </SelectItem>
                      <SelectItem value="voice-focus">
                        <div className="flex items-center">
                          <Mic className="mr-2" size={16} />
                          Voice Focus
                        </div>
                      </SelectItem>
                      <SelectItem value="music-enhance">
                        <div className="flex items-center">
                          <Music className="mr-2" size={16} />
                          Music Enhancement
                        </div>
                      </SelectItem>
                      <SelectItem value="podcast-optimize">
                        <div className="flex items-center">
                          <MessageSquare className="mr-2" size={16} />
                          Podcast Optimize
                        </div>
                      </SelectItem>
                      <SelectItem value="meeting-cleanup">
                        <div className="flex items-center">
                          <Users className="mr-2" size={16} />
                          Meeting Cleanup
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Output Format</Label>
                  <Select 
                    value={settings.outputFormat} 
                    onValueChange={(value) => handleSettingChange('outputFormat', value)}
                  >
                    <SelectTrigger className="bg-slate-800 border-gray-600" data-testid="select-output-format-advanced">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-gray-600">
                      <SelectItem value="wav">WAV (Uncompressed)</SelectItem>
                      <SelectItem value="mp3">MP3 (320kbps)</SelectItem>
                      <SelectItem value="flac">FLAC (Lossless)</SelectItem>
                      <SelectItem value="aac">AAC (256kbps)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Advanced Settings */}
          <TabsContent value="advanced" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-6">
                <h5 className="text-lg font-semibold text-electric-blue">AI Processing Options</h5>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Adaptive Processing</Label>
                      <p className="text-sm text-gray-400">Automatically adjust settings based on audio content</p>
                    </div>
                    <Switch
                      checked={settings.adaptiveProcessing}
                      onCheckedChange={(value) => handleSettingChange('adaptiveProcessing', value)}
                      data-testid="switch-adaptive-processing"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Emotion Preservation</Label>
                      <p className="text-sm text-gray-400">Maintain emotional tones and inflections</p>
                    </div>
                    <Switch
                      checked={settings.emotionPreservation}
                      onCheckedChange={(value) => handleSettingChange('emotionPreservation', value)}
                      data-testid="switch-emotion-preservation"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Context-Aware Processing</Label>
                      <p className="text-sm text-gray-400">Preserve important sounds like alarms, applause</p>
                    </div>
                    <Switch
                      checked={settings.contextAwareProcessing}
                      onCheckedChange={(value) => handleSettingChange('contextAwareProcessing', value)}
                      data-testid="switch-context-aware"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Dynamic Noise Handling</Label>
                      <p className="text-sm text-gray-400">Adapt to changing noise conditions</p>
                    </div>
                    <Switch
                      checked={settings.dynamicNoiseHandling}
                      onCheckedChange={(value) => handleSettingChange('dynamicNoiseHandling', value)}
                      data-testid="switch-dynamic-noise"
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <h5 className="text-lg font-semibold text-purple-400">Special Modes</h5>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Multi-Speaker Mode</Label>
                      <p className="text-sm text-gray-400">Optimize for recordings with multiple speakers</p>
                    </div>
                    <Switch
                      checked={settings.multiSpeakerMode}
                      onCheckedChange={(value) => handleSettingChange('multiSpeakerMode', value)}
                      data-testid="switch-multi-speaker"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Real-Time Mode</Label>
                      <p className="text-sm text-gray-400">Enable for live audio processing</p>
                    </div>
                    <Switch
                      checked={settings.realTimeMode}
                      onCheckedChange={(value) => handleSettingChange('realTimeMode', value)}
                      data-testid="switch-real-time"
                    />
                  </div>
                </div>

                <Separator className="bg-gray-700" />

                <div className="space-y-4">
                  <h6 className="font-medium text-gray-300">Audio Quality Settings</h6>
                  
                  <div>
                    <Label>Sample Rate (Hz)</Label>
                    <Select 
                      value={settings.sampleRate.toString()} 
                      onValueChange={(value) => handleSettingChange('sampleRate', parseInt(value))}
                    >
                      <SelectTrigger className="bg-slate-800 border-gray-600" data-testid="select-sample-rate">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-slate-800 border-gray-600">
                        <SelectItem value="22050">22.05 kHz</SelectItem>
                        <SelectItem value="44100">44.1 kHz (CD Quality)</SelectItem>
                        <SelectItem value="48000">48 kHz (Studio)</SelectItem>
                        <SelectItem value="96000">96 kHz (Hi-Res)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label>Bit Depth</Label>
                    <Select 
                      value={settings.bitDepth.toString()} 
                      onValueChange={(value) => handleSettingChange('bitDepth', parseInt(value))}
                    >
                      <SelectTrigger className="bg-slate-800 border-gray-600" data-testid="select-bit-depth">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-slate-800 border-gray-600">
                        <SelectItem value="16">16-bit</SelectItem>
                        <SelectItem value="24">24-bit</SelectItem>
                        <SelectItem value="32">32-bit</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Environment Profiles */}
          <TabsContent value="profiles" className="space-y-6">
            <div className="space-y-6">
              <div>
                <h5 className="text-lg font-semibold text-neon-green mb-4">Environment Profiles</h5>
                <p className="text-gray-400 mb-6">
                  Select or create custom noise profiles to optimize processing for specific environments.
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  {noiseProfiles.map((profile) => (
                    <Card 
                      key={profile.id} 
                      className={`glass-panel cursor-pointer transition-colors hover:border-neon-green/50 ${
                        settings.customNoiseProfile === profile.id ? 'border-neon-green' : 'border-gray-600'
                      }`}
                      onClick={() => handleSettingChange('customNoiseProfile', profile.id)}
                      data-testid={`profile-card-${profile.id}`}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h6 className="font-medium">{profile.name}</h6>
                          <Badge variant="outline" className="border-electric-blue/30 text-electric-blue">
                            {profile.noiseType}
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-400">{profile.description}</p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>

              <Separator className="bg-gray-700" />

              <div>
                <h6 className="font-medium mb-4 flex items-center">
                  <Upload className="mr-2 text-neon-green" size={20} />
                  Upload Custom Noise Sample
                </h6>
                <p className="text-sm text-gray-400 mb-4">
                  Upload a clean noise sample to create a custom profile for your specific environment.
                </p>
                
                <div className="flex items-center space-x-4">
                  <div className="relative">
                    <Input
                      type="file"
                      accept=".wav,.mp3,.flac,.aac"
                      onChange={handleNoiseProfileUpload}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      data-testid="input-noise-sample"
                    />
                    <Button 
                      variant="outline" 
                      className="border-neon-green text-neon-green hover:bg-neon-green/10"
                      disabled={isUploading}
                    >
                      {isUploading ? (
                        <Loader2 className="mr-2 animate-spin" size={16} />
                      ) : (
                        <FileAudio className="mr-2" size={16} />
                      )}
                      Select Noise Sample
                    </Button>
                  </div>
                  <p className="text-xs text-gray-500">
                    Supported: WAV, MP3, FLAC, AAC (max 10MB)
                  </p>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Output Settings */}
          <TabsContent value="output" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-6">
                <h5 className="text-lg font-semibold text-electric-blue">File Output Settings</h5>
                
                <div className="space-y-4">
                  <div>
                    <Label>Output Format</Label>
                    <Select 
                      value={settings.outputFormat} 
                      onValueChange={(value) => handleSettingChange('outputFormat', value)}
                    >
                      <SelectTrigger className="bg-slate-800 border-gray-600" data-testid="select-output-format-final">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-slate-800 border-gray-600">
                        <SelectItem value="wav">
                          <div>
                            <p className="font-medium">WAV (Uncompressed)</p>
                            <p className="text-xs text-gray-400">Best quality, larger file size</p>
                          </div>
                        </SelectItem>
                        <SelectItem value="flac">
                          <div>
                            <p className="font-medium">FLAC (Lossless)</p>
                            <p className="text-xs text-gray-400">Compressed but lossless</p>
                          </div>
                        </SelectItem>
                        <SelectItem value="mp3">
                          <div>
                            <p className="font-medium">MP3 (320kbps)</p>
                            <p className="text-xs text-gray-400">Good quality, smaller size</p>
                          </div>
                        </SelectItem>
                        <SelectItem value="aac">
                          <div>
                            <p className="font-medium">AAC (256kbps)</p>
                            <p className="text-xs text-gray-400">Optimized for mobile devices</p>
                          </div>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label>Channels</Label>
                    <Select 
                      value={settings.channels.toString()} 
                      onValueChange={(value) => handleSettingChange('channels', parseInt(value))}
                    >
                      <SelectTrigger className="bg-slate-800 border-gray-600" data-testid="select-channels">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-slate-800 border-gray-600">
                        <SelectItem value="1">Mono</SelectItem>
                        <SelectItem value="2">Stereo</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <h5 className="text-lg font-semibold text-purple-400">Processing Summary</h5>
                
                <div className="glass-panel p-4 space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Noise Reduction:</span>
                    <span className="text-neon-green font-medium">{settings.noiseReductionLevel}/10</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Voice Preservation:</span>
                    <span className="text-electric-blue font-medium">{settings.voicePreservation}/10</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Processing Mode:</span>
                    <Badge variant="outline" className="border-purple-400/30 text-purple-400">
                      {settings.processingMode.replace('-', ' ')}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Output Format:</span>
                    <span className="text-white font-medium">{settings.outputFormat.toUpperCase()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Quality:</span>
                    <span className="text-white font-medium">
                      {settings.sampleRate/1000}kHz / {settings.bitDepth}-bit
                    </span>
                  </div>
                </div>

                <div className="space-y-3">
                  <h6 className="font-medium text-gray-300">Active Features</h6>
                  <div className="flex flex-wrap gap-2">
                    {settings.adaptiveProcessing && (
                      <Badge className="bg-neon-green/20 text-neon-green border-neon-green/30">
                        Adaptive Processing
                      </Badge>
                    )}
                    {settings.emotionPreservation && (
                      <Badge className="bg-electric-blue/20 text-electric-blue border-electric-blue/30">
                        Emotion Preservation
                      </Badge>
                    )}
                    {settings.contextAwareProcessing && (
                      <Badge className="bg-purple-400/20 text-purple-400 border-purple-400/30">
                        Context Aware
                      </Badge>
                    )}
                    {settings.multiSpeakerMode && (
                      <Badge className="bg-orange-400/20 text-orange-400 border-orange-400/30">
                        Multi-Speaker
                      </Badge>
                    )}
                    {settings.realTimeMode && (
                      <Badge className="bg-red-400/20 text-red-400 border-red-400/30">
                        Real-Time
                      </Badge>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

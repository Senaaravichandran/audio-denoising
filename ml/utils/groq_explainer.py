#!/usr/bin/env python3
"""
Groq AI Service for AudioClarity
Provides intelligent explanations of audio enhancement processes
"""

import os
import sys
import json
from typing import Dict, Any, Optional
from pathlib import Path

class GroqExplainer:
    """AI-powered explanation service using Groq"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            print("Warning: GROQ_API_KEY not found. Set environment variable or pass api_key parameter.")
            self.client = None
            return
            
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.model = "llama-3.1-8b-instant"  # Fast and efficient model
            print("Groq AI service initialized successfully")
        except ImportError:
            print("Error: groq package not installed. Run: pip install groq")
            self.client = None
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            self.client = None
    
    def generate_enhancement_explanation(self, enhancement_data: Dict[str, Any]) -> str:
        """
        Generate detailed explanation of audio enhancement process
        
        Args:
            enhancement_data: Dictionary containing enhancement details
            
        Returns:
            str: Detailed AI-generated explanation
        """
        if not self.client:
            return self._fallback_explanation(enhancement_data)
        
        try:
            # Prepare the prompt with enhancement details
            prompt = self._create_enhancement_prompt(enhancement_data)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert audio engineer and AI specialist. Explain audio enhancement processes in a clear, detailed, and professional manner."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            print("AI explanation generated successfully")
            return explanation
            
        except Exception as e:
            print(f"Error generating AI explanation: {e}")
            return self._fallback_explanation(enhancement_data)
    
    def _create_enhancement_prompt(self, data: Dict[str, Any]) -> str:
        """Create a detailed prompt for the AI model"""
        
        prompt = f"""
AudioClarity Enhancement Process Analysis

Please provide a detailed, professional explanation of the audio enhancement process that was just completed. Here are the technical details:

PROCESSING DETAILS:
- Input Source: {data.get('source_type', 'Unknown')}
- Original File: {data.get('original_filename', 'N/A')}
- Processing Mode: {data.get('processing_mode', 'Standard')}
- Noise Reduction Level: {data.get('noise_reduction_level', 'N/A')}/10
- Voice Preservation: {data.get('voice_preservation', 'N/A')}/10
- Output Format: {data.get('output_format', 'WAV')}
- Processing Time: {data.get('processing_time', 'N/A')} seconds
- AI Model Used: {data.get('ai_model', 'DCCRN (Deep Complex Convolution Recurrent Network)')}

FILE INFORMATION:
- Original Size: {data.get('original_size', 'N/A')}
- Enhanced Size: {data.get('enhanced_size', 'N/A')}
- Sample Rate: {data.get('sample_rate', '16000')} Hz
- Duration: {data.get('duration', 'N/A')} seconds

ENHANCEMENT STAGES:
{self._format_stages(data.get('stages', []))}

Please explain:
1. What exactly was done to enhance this audio
2. How the DCCRN AI model processed the audio
3. What specific improvements were made (noise reduction, clarity, etc.)
4. Why the chosen settings were optimal for this type of audio
5. What the user can expect from the enhanced audio quality

Make the explanation clear, informative, and professional. Focus on the technical achievements and quality improvements.
"""
        return prompt
    
    def _format_stages(self, stages: list) -> str:
        """Format processing stages for the prompt"""
        if not stages:
            return "- Standard audio enhancement pipeline completed"
        
        formatted = []
        for i, stage in enumerate(stages, 1):
            formatted.append(f"- Stage {i}: {stage}")
        return "\n".join(formatted)
    
    def _fallback_explanation(self, data: Dict[str, Any]) -> str:
        """Provide fallback explanation when AI is not available"""
        source = data.get('source_type', 'audio file')
        mode = data.get('processing_mode', 'standard')
        noise_level = data.get('noise_reduction_level', 'moderate')
        
        return f"""
🎯 AudioClarity Enhancement Complete!

✅ PROCESSING SUMMARY:
Your {source} has been successfully enhanced using our advanced DCCRN (Deep Complex Convolution Recurrent Network) AI model. Here's what was accomplished:

🔧 ENHANCEMENT PROCESS:
• Applied {mode} processing mode for optimal quality
• Noise reduction level: {noise_level}/10 - Removed background noise, hums, and distortions
• Voice preservation: Maintained natural speech characteristics
• AI-powered spectral analysis and reconstruction

🎵 AUDIO IMPROVEMENTS:
• Significantly reduced background noise and interference
• Enhanced speech clarity and intelligibility  
• Improved overall audio quality and listening experience
• Preserved original audio dynamics and natural sound

⚡ TECHNICAL DETAILS:
• AI Model: DCCRN - State-of-the-art audio enhancement
• Processing: Real-time spectral domain enhancement
• Output: High-quality {data.get('output_format', 'WAV')} file
• Sample Rate: {data.get('sample_rate', '16000')} Hz for optimal clarity

Your enhanced audio is now ready with professional-grade quality improvements!
"""

    def generate_social_media_explanation(self, metadata: Dict[str, Any]) -> str:
        """
        Generate specialized explanation for social media content enhancement
        
        Args:
            metadata: Dictionary containing social media content details
            
        Returns:
            str: Detailed AI-generated explanation for social media processing
        """
        if not self.client:
            return self._fallback_social_media_explanation(metadata)
        
        try:
            prompt = self._create_social_media_prompt(metadata)
            
            # Call Groq AI for social media explanation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert in social media content processing and AI audio enhancement. Explain how content from platforms like YouTube, TikTok, Instagram, etc. is processed and enhanced using advanced AI models. Be informative, professional, and engaging."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Groq AI error for social media: {e}")
            return self._fallback_social_media_explanation(metadata)
    
    def _create_social_media_prompt(self, data: Dict[str, Any]) -> str:
        """Create a specialized prompt for social media content"""
        
        platform = data.get('platform', 'social media')
        title = data.get('title', 'Unknown')
        duration = data.get('duration', 0)
        download_type = data.get('download_type', 'audio')
        processing_mode = data.get('processing_mode', 'balanced')
        denoising_strength = data.get('denoising_strength', 0.8)
        
        prompt = f"""
AudioClarity Social Media Enhancement Analysis

Please provide a comprehensive, engaging explanation of the social media content processing that was just completed. Make it informative and professional.

CONTENT DETAILS:
- Platform: {platform}
- Title: "{title}"
- Duration: {duration} seconds
- Download Type: {download_type}
- Processing Mode: {processing_mode}
- Noise Reduction Level: {round(denoising_strength * 10)}/10

PROCESSING PIPELINE:
1. Downloaded content using yt-dlp for optimal quality from {platform}
2. Extracted audio from the {download_type} content using moviepy
3. Applied DCCRN AI enhancement with {processing_mode} mode
4. Applied noise reduction at level {round(denoising_strength * 10)}/10
5. {"Combined enhanced audio with original video" if download_type == 'video' else "Prepared enhanced audio for download"}

Please explain:
1. How we successfully processed this {platform} content
2. What makes social media audio enhancement challenging and how we solved it
3. The specific improvements made by the DCCRN AI model
4. How the chosen settings optimize quality for this type of content
5. What quality improvements the user can expect

Make the explanation clear, engaging, and highlight the technical achievements. Use emojis and formatting to make it visually appealing.
"""
        return prompt
    
    def _fallback_social_media_explanation(self, data: Dict[str, Any]) -> str:
        """Provide fallback explanation for social media content when AI is not available"""
        platform = data.get('platform', 'social media')
        title = data.get('title', 'Unknown')
        duration = data.get('duration', 0)
        download_type = data.get('download_type', 'audio')
        processing_mode = data.get('processing_mode', 'balanced')
        denoising_strength = data.get('denoising_strength', 0.8)
        
        return f"""🎯 Social Media Content Enhancement Complete!

✅ PROCESSING SUMMARY:
Successfully processed {platform} content and enhanced the audio quality using our advanced DCCRN AI model.

📱 SOURCE INFORMATION:
• Platform: {platform}
• Title: "{title}"
• Duration: {duration} seconds
• Content Type: {download_type}
• Processing Mode: {processing_mode}

🔧 ENHANCEMENT PROCESS:
• Downloaded content using yt-dlp for optimal quality
• Extracted high-quality audio from the {platform} content
• Applied DCCRN AI enhancement with {processing_mode} settings
• Noise reduction level: {round(denoising_strength * 10)}/10
• Preserved voice characteristics while removing background noise

🎵 AUDIO IMPROVEMENTS:
• Removed compression artifacts from social media encoding
• Enhanced speech clarity and intelligibility
• Reduced background noise and interference
• Improved overall audio quality for better listening
• Maintained natural sound dynamics

Your {platform} content is now ready with significantly improved audio quality!"""

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate AI explanations for AudioClarity processing')
    parser.add_argument('--content-type', default='audio', choices=['audio', 'social_media'],
                       help='Type of content being processed')
    parser.add_argument('--platform', default='unknown', help='Social media platform')
    parser.add_argument('--title', default='Unknown', help='Content title')
    parser.add_argument('--duration', type=float, default=0, help='Content duration in seconds')
    parser.add_argument('--download-type', default='audio', choices=['audio', 'video'],
                       help='Type of download')
    parser.add_argument('--processing-mode', default='balanced', help='Processing mode used')
    parser.add_argument('--denoising-strength', type=float, default=0.8, help='Denoising strength applied')
    
    # Fallback for old JSON input method
    if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
        try:
            # Parse enhancement data from JSON
            enhancement_data = json.loads(sys.argv[1])
            explainer = GroqExplainer()
            explanation = explainer.generate_enhancement_explanation(enhancement_data)
            print(explanation)
            return
        except json.JSONDecodeError:
            print("Error: Invalid JSON data provided")
            sys.exit(1)
    
    args = parser.parse_args()
    
    try:
        # Initialize Groq explainer
        explainer = GroqExplainer()
        
        if args.content_type == 'social_media':
            # Prepare metadata for social media explanation
            metadata = {
                'platform': args.platform,
                'title': args.title,
                'duration': args.duration,
                'download_type': args.download_type,
                'processing_mode': args.processing_mode,
                'denoising_strength': args.denoising_strength
            }
            
            # Generate social media explanation
            explanation = explainer.generate_social_media_explanation(metadata)
        else:
            # Prepare data for regular audio enhancement explanation
            enhancement_data = {
                'source_type': 'uploaded audio file',
                'processing_mode': args.processing_mode,
                'denoising_strength': args.denoising_strength
            }
            
            # Generate regular explanation
            explanation = explainer.generate_enhancement_explanation(enhancement_data)
        
        # Output the explanation
        print(explanation)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

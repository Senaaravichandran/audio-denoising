import streamlit as st
import os
import subprocess
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from pathlib import Path
import time
import sys

# Set page configuration
st.set_page_config(
    page_title="Audio Denoising App",
    page_icon="🔊",
    layout="wide"
)

# Constants
MAX_AUDIO_LENGTH = 120  # seconds

# Determine the correct Python executable to use
PYTHON_EXECUTABLE = sys.executable  # Use the current Python interpreter

# Use the correct Python executable in the command
COMMAND_INFERENCE = f'"{PYTHON_EXECUTABLE}" main.py --mode "prediction" --audio_dir_prediction "input/" --dir_save_prediction "output/" --audio_output_prediction "input.wav"'

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem !important;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 2px dashed #ccc;
    }
    .results-section {
        background-color: #f1f8e9;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .audio-player {
        width: 100%;
        margin-bottom: 1rem;
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 1rem;
    }
    .download-btn {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 1rem 0;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

def clean_temp():
    """Remove all files in specific paths"""
    paths_to_remove = ['input/', 'output/']
    
    try:
        for path in paths_to_remove:
            Path(path).mkdir(exist_ok=True)  # Create directory if it doesn't exist
            for f in os.listdir(path):
                file_path = os.path.join(path, f)
                if not 'README' in f and os.path.isfile(file_path):
                    os.remove(file_path)
    except Exception as e:
        st.error(f"Error cleaning temporary files: {e}")

def plot_waveform(y, sr, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_spectrogram(y, sr, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.tight_layout()
    return fig

def process_audio(input_path, output_path):
    """
    Process the audio file directly without using subprocess.
    This is a fallback method if the inference command fails.
    
    For demonstration purposes, this just applies a simple filter.
    In a real app, you would implement the actual denoising logic here.
    """
    try:
        # Load audio
        y, sr = librosa.load(input_path)
        
        # Simple processing: high-pass filter (just as an example)
        from scipy import signal
        sos = signal.butter(10, 500, 'hp', fs=sr, output='sos')
        y_filtered = signal.sosfilt(sos, y)
        
        # Save processed audio
        sf.write(output_path, y_filtered, sr)
        return True
    except Exception as e:
        st.error(f"Error in fallback processing: {e}")
        return False

def main():
    # Clean temporary files
    clean_temp()
    
    # Page header
    st.markdown("<h1 class='main-header'>🔊 Audio Denoising - Speech Enhancement</h1>", unsafe_allow_html=True)
    
    # About section in expander
    with st.expander("ℹ️ About this app"):
        st.markdown("""
        This application helps you remove noise from your audio files, enhancing speech quality.
        Simply upload an audio file, and the app will process it to reduce background noise.
        
        **Features:**
        - Supports WAV, MP3, and OGG files
        - Visualizes audio waveforms and spectrograms
        - Analyzes audio characteristics
        
        **Credits:** [GitHub Repository](https://github.com/Senaaravichandran/audio-denoising)
        """)
    
    # File upload section
    st.markdown("<h2 class='sub-header'>Upload Your Audio</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='info-text'>", unsafe_allow_html=True)
        st.markdown("""
        **Supported formats:**
        - WAV
        - MP3
        - OGG
        
        **Max length:** 120 seconds
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Show file details
        file_details = {
            "Filename": uploaded_file.name,
            "File Type": uploaded_file.type,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        st.json(file_details)
        
        # Process uploaded file
        if uploaded_file.type in ['audio/wav', 'audio/mp3', 'audio/ogg']:
            # Create input directory if it doesn't exist
            input_dir = Path('input')
            input_dir.mkdir(exist_ok=True)
            
            input_filename = 'input/noisy_voice_long_t2'
            input_wav_path = f'{input_filename}.wav'
            
            # Save file according to its type
            if uploaded_file.type == 'audio/mp3':
                mp3_path = f'{input_filename}.mp3'
                with open(mp3_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                data, samplerate = librosa.load(mp3_path)
                sf.write(input_wav_path, data, samplerate)
            
            elif uploaded_file.type == 'audio/ogg':
                ogg_path = f'{input_filename}.ogg'
                with open(ogg_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                data, samplerate = librosa.load(ogg_path)
                sf.write(input_wav_path, data, samplerate)
            
            elif uploaded_file.type == 'audio/wav':
                with open(input_wav_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
            
            # Display input audio
            st.markdown("<h2 class='sub-header'>Original Audio</h2>", unsafe_allow_html=True)
            
            # Load audio for analysis
            y, sr = librosa.load(input_wav_path)
            
            audio_cols = st.columns(2)
            with audio_cols[0]:
                st.markdown("<div class='audio-player'>", unsafe_allow_html=True)
                st.audio(input_wav_path, format='audio/wav')
                st.markdown("</div>", unsafe_allow_html=True)
            
            with audio_cols[1]:
                # Audio information
                duration = librosa.get_duration(y=y, sr=sr)
                st.info(f"Duration: {duration:.2f} seconds | Sample Rate: {sr} Hz")
                
                # Tempo and beat - Fixed the numpy array handling
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                # Convert tempo to a scalar if it's a numpy array
                tempo_value = tempo.item() if hasattr(tempo, 'item') else float(tempo)
                st.info(f"Estimated tempo: {tempo_value:.2f} beats per minute")
            
            # Original waveform and spectrogram
            input_tabs = st.tabs(["Waveform", "Spectrogram"])
            with input_tabs[0]:
                st.pyplot(plot_waveform(y, sr, "Input Audio Waveform"))
            
            with input_tabs[1]:
                st.pyplot(plot_spectrogram(y, sr, "Input Audio Spectrogram"))
            
            # Process audio
            st.markdown("<h2 class='sub-header'>Processing Audio</h2>", unsafe_allow_html=True)
            
            # Create output directory if it doesn't exist
            output_dir = Path('output')
            output_dir.mkdir(exist_ok=True)
            
            output_path = 'output/input.wav'
            
            # Show progress bar during processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Processing method selection
            processing_method = st.radio(
                "Choose processing method:",
                ["External Command (Original Model)", "Built-in Filtering (Fallback)"]
            )
            
            # Add debug mode toggle
            debug_mode = st.checkbox("Enable Debug Mode", value=False, 
                                    help="Show additional debug information for troubleshooting")
            
            if debug_mode:
                st.info("Debug Information:")
                st.write(f"Python Executable: {PYTHON_EXECUTABLE}")
                st.write(f"Command: {COMMAND_INFERENCE}")
                st.write(f"Current Working Directory: {os.getcwd()}")
                st.write(f"Files in input directory: {os.listdir('input') if os.path.exists('input') else 'Directory not found'}")
            
            if st.button("Start Processing"):
                status_text.text("Running audio processing...")
                
                success = False
                
                if processing_method == "External Command (Original Model)":
                    try:
                        # Show the command being used (helpful for debugging)
                        if debug_mode:
                            st.code(COMMAND_INFERENCE, language="bash")
                        
                        # Start progress bar
                        for i in range(50):
                            progress_bar.progress(i)
                            time.sleep(0.01)
                        
                        # Execute the inference command and capture output
                        process = subprocess.Popen(
                            COMMAND_INFERENCE,
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        
                        stdout, stderr = process.communicate()
                        
                        # Complete progress bar
                        for i in range(50, 101):
                            progress_bar.progress(i)
                            time.sleep(0.01)
                        
                        if process.returncode == 0:
                            if os.path.exists(output_path):
                                status_text.success("Processing completed successfully!")
                                if debug_mode:
                                    st.code(stdout)
                                success = True
                            else:
                                status_text.error("Process completed but output file not found")
                                if debug_mode:
                                    st.code(stdout)
                                    st.code(stderr)
                                
                                # Try to manually create output if model command succeeded but file wasn't created
                                status_text.info("Attempting fallback processing...")
                                success = process_audio(input_wav_path, output_path)
                        else:
                            status_text.error("Error during processing")
                            if debug_mode:
                                st.code(stderr)
                            
                            # If there's an error about librosa.output, suggest updating prediction_denoise.py
                            if "No librosa attribute output" in stderr:
                                st.error("""
                                The error is due to using a deprecated librosa function. 
                                
                                Please update your prediction_denoise.py file by replacing:
                                ```python
                                librosa.output.write_wav(dir_save_prediction + audio_output_prediction, denoise_long[0, :], sample_rate)
                                ```
                                
                                with:
                                ```python
                                import soundfile as sf
                                sf.write(dir_save_prediction + audio_output_prediction, denoise_long[0, :], sample_rate)
                                ```
                                """)
                            
                            # Offer to use fallback processing
                            if st.button("Use Fallback Processing"):
                                success = process_audio(input_wav_path, output_path)
                    except Exception as e:
                        status_text.error(f"Error executing inference command: {e}")
                        if debug_mode:
                            st.exception(e)
                        success = False
                else:
                    # Use the built-in fallback method
                    status_text.text("Using built-in processing...")
                    
                    # Simulate progress
                    for i in range(101):
                        progress_bar.progress(i)
                        time.sleep(0.01)
                    
                    success = process_audio(input_wav_path, output_path)
                    
                    if success:
                        status_text.success("Processing completed successfully!")
                    else:
                        status_text.error("Error during built-in processing")
                
                # Display results if output file exists
                if success and os.path.exists(output_path):
                    st.markdown("<div class='results-section'>", unsafe_allow_html=True)
                    st.markdown("<h2 class='sub-header'>Enhanced Audio Result</h2>", unsafe_allow_html=True)
                    
                    # Load the processed audio
                    y_output, sr_output = librosa.load(output_path)
                    
                    output_cols = st.columns(2)
                    with output_cols[0]:
                        st.markdown("<div class='audio-player'>", unsafe_allow_html=True)
                        st.audio(output_path, format='audio/wav')
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with output_cols[1]:
                        # Audio information
                        duration = librosa.get_duration(y=y_output, sr=sr_output)
                        st.info(f"Duration: {duration:.2f} seconds | Sample Rate: {sr_output} Hz")
                        
                        # Tempo and beat - Fixed the numpy array handling
                        tempo, beat_frames = librosa.beat.beat_track(y=y_output, sr=sr_output)
                        # Convert tempo to a scalar if it's a numpy array
                        tempo_value = tempo.item() if hasattr(tempo, 'item') else float(tempo)
                        st.info(f"Estimated tempo: {tempo_value:.2f} beats per minute")
                    
                    # Output waveform and spectrogram
                    output_tabs = st.tabs(["Waveform", "Spectrogram"])
                    with output_tabs[0]:
                        st.pyplot(plot_waveform(y_output, sr_output, "Enhanced Audio Waveform"))
                    
                    with output_tabs[1]:
                        st.pyplot(plot_spectrogram(y_output, sr_output, "Enhanced Audio Spectrogram"))
                    
                    # Compare before and after
                    st.subheader("Before vs After Comparison")
                    comparison_tabs = st.tabs(["Waveform Comparison", "Spectrogram Comparison"])
                    
                    with comparison_tabs[0]:
                        # Create comparison figure
                        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                        librosa.display.waveshow(y, sr=sr, ax=axs[0], alpha=0.7)
                        axs[0].set_title("Original Audio", fontsize=14, fontweight='bold')
                        axs[0].grid(True, alpha=0.3)
                        
                        librosa.display.waveshow(y_output, sr=sr_output, ax=axs[1], alpha=0.7, color='g')
                        axs[1].set_title("Enhanced Audio", fontsize=14, fontweight='bold')
                        axs[1].set_xlabel("Time (s)", fontsize=12)
                        axs[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with comparison_tabs[1]:
                        # Create spectrogram comparison
                        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                        D_input = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                        img1 = librosa.display.specshow(D_input, x_axis='time', y_axis='log', sr=sr, ax=axs[0])
                        axs[0].set_title("Original Spectrogram", fontsize=14, fontweight='bold')
                        fig.colorbar(img1, ax=axs[0], format="%+2.f dB")
                        
                        D_output = librosa.amplitude_to_db(np.abs(librosa.stft(y_output)), ref=np.max)
                        img2 = librosa.display.specshow(D_output, x_axis='time', y_axis='log', sr=sr_output, ax=axs[1])
                        axs[1].set_title("Enhanced Spectrogram", fontsize=14, fontweight='bold')
                        fig.colorbar(img2, ax=axs[1], format="%+2.f dB")
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Add audio metrics
                    st.subheader("Audio Quality Metrics")
                    metrics_cols = st.columns(3)
                    
                    with metrics_cols[0]:
                        # Signal-to-Noise Ratio (estimated)
                        # Fix: Make sure arrays have the same length before calculating difference
                        min_length = min(len(y), len(y_output))
                        y_trim = y[:min_length]
                        y_output_trim = y_output[:min_length]
                        
                        signal_power = np.mean(y_output_trim**2)
                        noise_power = np.mean((y_trim - y_output_trim)**2)
                        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                        st.metric("Estimated SNR", f"{snr:.2f} dB")
                    
                    with metrics_cols[1]:
                        # RMS levels
                        rms_input = np.sqrt(np.mean(y_trim**2))  # Use trimmed version for consistency
                        rms_output = np.sqrt(np.mean(y_output_trim**2)) 
                        st.metric("RMS Level Reduction", f"{(1 - rms_output/rms_input)*100:.1f}%")
                    
                    with metrics_cols[2]:
                        # Peak amplitude
                        peak_input = np.max(np.abs(y_trim))  # Use trimmed version for consistency
                        peak_output = np.max(np.abs(y_output_trim))
                        st.metric("Peak Level", f"{peak_output:.2f}", f"{(peak_output-peak_input):.2f}")
                    
                    # Download button for the processed file
                    with open(output_path, 'rb') as file:
                        st.download_button(
                            label="Download Enhanced Audio",
                            data=file,
                            file_name="enhanced_audio.wav",
                            mime="audio/wav",
                            key="download-button",
                            help="Click to download the processed audio file"
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
                elif not success:
                    st.error("Output file not found. The audio processing may have failed.")
                    
                    # Give more helpful information
                    st.warning("""
                    The processing didn't generate the expected output file. 
                    Possible reasons:
                    - The model is not properly set up
                    - The command path is incorrect
                    - There was an error during processing
                    
                    Try using the "Built-in Filtering" option which doesn't require the external model.
                    """)
    
    # Footer
    st.markdown("<div class='footer'>", unsafe_allow_html=True)
    st.markdown("Audio Denoising App | Made with Streamlit | [GitHub](https://github.com/Senaaravichandran/audio-denoising)")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
import librosa
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moviepy.editor import VideoFileClip
from config import AUDIO_CUTOFF_FREQ


def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    try:
        print(f"Extracting audio from {os.path.basename(video_path)}...")
        clip = VideoFileClip(video_path)
        temp_audio_path = "temp_uploads/temp_extracted_audio.wav"
        clip.audio.write_audiofile(temp_audio_path, logger=None)
        clip.close()
        return temp_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None


def analyze_high_frequency_cutoff(file_path):
    """
    Analyze audio for high-frequency cutoff (AI voice indicator)
    AI voices often have sharp cutoffs above 16kHz
    """
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Compute spectrogram
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Analyze high frequencies
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_indices = np.where(freqs > AUDIO_CUTOFF_FREQ)[0]
        
        verdict = "INCONCLUSIVE"
        is_fake = False
        avg_high_freq_energy = -80
        
        if len(high_freq_indices) > 0:
            high_freq_energy = S_db[high_freq_indices, :]
            avg_high_freq_energy = np.mean(high_freq_energy)
            
            # Threshold logic
            if avg_high_freq_energy < -70:
                verdict = "FAKE AUDIO: High Frequency Cutoff Detected"
                is_fake = True
            else:
                verdict = "REAL AUDIO: Natural Spectrum"
                is_fake = False
        else:
            verdict = "INCONCLUSIVE: Sample rate too low"
        
        return {
            "avg_high_freq_energy": round(float(avg_high_freq_energy), 2),
            "sample_rate": sr,
            "verdict": verdict,
            "is_fake": is_fake
        }
        
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None


def analyze_silence_patterns(file_path):
    """
    Analyze silence patterns for breathing sounds
    Real humans have natural breathing in silence gaps
    AI voices often have perfectly clean silence
    """
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Detect non-silent intervals
        intervals = librosa.effects.split(y, top_db=30)
        
        if len(intervals) < 2:
            return {
                "silence_gaps_count": 0,
                "has_breathing_sounds": False,
                "verdict": "INCONCLUSIVE: Not enough audio segments",
                "is_fake": False
            }
        
        # Calculate silence gaps
        silence_gaps = []
        for i in range(len(intervals) - 1):
            gap_start = intervals[i][1]
            gap_end = intervals[i + 1][0]
            gap_duration = (gap_end - gap_start) / sr
            
            if gap_duration > 0.1:  # At least 100ms gap
                silence_gaps.append((gap_start, gap_end))
        
        # Analyze silence gaps for low-frequency noise (breathing)
        breathing_detected = 0
        for gap_start, gap_end in silence_gaps[:5]:  # Check first 5 gaps
            silence_segment = y[gap_start:gap_end]
            
            # Check for low-frequency energy (breathing is typically 100-500 Hz)
            S = np.abs(librosa.stft(silence_segment))
            freqs = librosa.fft_frequencies(sr=sr)
            low_freq_indices = np.where((freqs > 100) & (freqs < 500))[0]
            
            if len(low_freq_indices) > 0:
                low_freq_energy = np.mean(S[low_freq_indices, :])
                if low_freq_energy > 0.01:  # Threshold for breathing detection
                    breathing_detected += 1
        
        has_breathing = breathing_detected > len(silence_gaps) * 0.3  # 30% of gaps
        
        verdict = "REAL AUDIO: Natural breathing detected" if has_breathing else "SUSPICIOUS: No breathing in silence"
        
        return {
            "silence_gaps_count": len(silence_gaps),
            "breathing_gaps_detected": breathing_detected,
            "has_breathing_sounds": has_breathing,
            "verdict": verdict,
            "is_fake": not has_breathing
        }
        
    except Exception as e:
        print(f"Error analyzing silence: {e}")
        return None


def analyze_audio_full(file_path, is_video=False):
    """
    Full audio analysis pipeline
    """
    actual_audio_path = file_path
    temp_file_created = False
    
    # Extract audio if video
    if is_video:
        actual_audio_path = extract_audio_from_video(file_path)
        temp_file_created = True
        if not actual_audio_path:
            return {"error": "Failed to extract audio from video"}
    
    # Run all audio checks
    high_freq_result = analyze_high_frequency_cutoff(actual_audio_path)
    silence_result = analyze_silence_patterns(actual_audio_path)
    
    # Cleanup temp audio file
    if temp_file_created and os.path.exists(actual_audio_path):
        os.remove(actual_audio_path)
    
    # Combine results
    fake_indicators = 0
    if high_freq_result and high_freq_result.get("is_fake"):
        fake_indicators += 1
    if silence_result and silence_result.get("is_fake"):
        fake_indicators += 1
    
    overall_verdict = "FAKE AUDIO" if fake_indicators >= 1 else "REAL AUDIO"
    
    return {
        "high_frequency_analysis": high_freq_result,
        "silence_analysis": silence_result,
        "overall_verdict": overall_verdict,
        "is_fake": fake_indicators >= 1,
        "fake_indicators_count": fake_indicators
    }


# === STANDALONE TESTING ===
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
    print("Select a VIDEO or AUDIO file to analyze...")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Audio/Video File",
        filetypes=[
            ("Media Files", "*.mp4 *.avi *.mov *.wav *.mp3 *.flac"),
            ("Video Files", "*.mp4 *.avi *.mov"),
            ("Audio Files", "*.wav *.mp3 *.flac")
        ]
    )
    
    if file_path:
        is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        print(f"\n🔍 Analyzing: {os.path.basename(file_path)}")
        print(f"Type: {'Video (extracting audio)' if is_video else 'Audio'}\n")
        
        result = analyze_audio_full(file_path, is_video=is_video)
        
        print("\n" + "="*60)
        print("AUDIO ANALYSIS RESULTS")
        print("="*60)
        import json
        print(json.dumps(result, indent=2))
        print("="*60)
    else:
        print("No file selected.")

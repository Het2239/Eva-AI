#!/usr/bin/env python3
"""
EVA Speech Engine - STT & TTS
=============================
Speech-to-text using Whisper, text-to-speech using edge-tts.
"""

import os
import sys
import asyncio
import tempfile
import wave
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

# Audio recording
try:
    import sounddevice as sd
    import numpy as np
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# Speech-to-text
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# Text-to-speech
try:
    import edge_tts
    HAS_TTS = True
except ImportError:
    HAS_TTS = False


@dataclass
class SpeechConfig:
    """Speech engine configuration."""
    wake_word: str = "eva"
    whisper_model: str = "base"  # tiny, base, small, medium, large
    tts_voice: str = "en-US-AriaNeural"  # Microsoft Edge voice
    sample_rate: int = 16000
    silence_threshold: float = 0.01
    silence_duration: float = 1.5  # seconds of silence to stop recording


class SpeechToText:
    """Speech-to-text using faster-whisper."""
    
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None
    
    def _get_model(self):
        """Lazy load Whisper model."""
        if self._model is None:
            if not HAS_WHISPER:
                raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")
            print(f"Loading Whisper {self.model_size} model...")
            self._model = WhisperModel(self.model_size, compute_type="int8")
        return self._model
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        model = self._get_model()
        segments, _ = model.transcribe(audio_path, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()
    
    def transcribe_array(self, audio: "np.ndarray", sample_rate: int = 16000) -> str:
        """Transcribe numpy audio array."""
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            # Write wav
            import scipy.io.wavfile as wav
            wav.write(temp_path, sample_rate, audio)
            return self.transcribe(temp_path)
        finally:
            os.unlink(temp_path)


class TextToSpeech:
    """Text-to-speech using edge-tts."""
    
    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice
    
    async def _speak_async(self, text: str, output_path: Optional[str] = None) -> str:
        """Generate speech asynchronously."""
        if not HAS_TTS:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")
        
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)
        return output_path
    
    def speak(self, text: str, play: bool = True) -> str:
        """Generate and optionally play speech."""
        # Run async
        output_path = asyncio.run(self._speak_async(text))
        
        if play:
            self._play_audio(output_path)
        
        return output_path
    
    def _play_audio(self, path: str) -> None:
        """Play audio file."""
        import subprocess
        import platform
        
        system = platform.system()
        try:
            if system == "Linux":
                # Try mpv, then ffplay, then paplay
                for player in ["mpv --no-video", "ffplay -nodisp -autoexit", "paplay"]:
                    try:
                        subprocess.run(
                            f"{player} {path}",
                            shell=True,
                            capture_output=True,
                            check=True,
                        )
                        break
                    except:
                        continue
            elif system == "Darwin":
                subprocess.run(["afplay", path], check=True)
            elif system == "Windows":
                os.startfile(path)
        except Exception as e:
            print(f"Could not play audio: {e}")
        finally:
            # Cleanup
            try:
                os.unlink(path)
            except:
                pass


class AudioRecorder:
    """Record audio from microphone."""
    
    def __init__(self, sample_rate: int = 16000, silence_threshold: float = 0.01, silence_duration: float = 1.5):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
    
    def record_until_silence(self, max_duration: float = 30.0) -> "np.ndarray":
        """Record audio until silence is detected."""
        if not HAS_AUDIO:
            raise RuntimeError("sounddevice not installed. Run: pip install sounddevice")
        
        print("🎤 Listening...")
        
        chunks = []
        silence_samples = 0
        silence_limit = int(self.silence_duration * self.sample_rate)
        max_samples = int(max_duration * self.sample_rate)
        total_samples = 0
        
        def callback(indata, frames, time, status):
            nonlocal silence_samples, total_samples
            
            chunks.append(indata.copy())
            total_samples += frames
            
            # Check for silence
            volume = np.abs(indata).mean()
            if volume < self.silence_threshold:
                silence_samples += frames
            else:
                silence_samples = 0
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while silence_samples < silence_limit and total_samples < max_samples:
                sd.sleep(100)
        
        print("✓ Recording complete")
        
        if chunks:
            return np.concatenate(chunks, axis=0).flatten()
        return np.array([], dtype="float32")
    
    def record_duration(self, duration: float = 5.0) -> "np.ndarray":
        """Record for a fixed duration."""
        if not HAS_AUDIO:
            raise RuntimeError("sounddevice not installed")
        
        print(f"🎤 Recording for {duration}s...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("✓ Recording complete")
        return audio.flatten()


class SpeechEngine:
    """
    Complete speech engine with STT, TTS, and wake word detection.
    """
    
    def __init__(self, config: Optional[SpeechConfig] = None):
        self.config = config or SpeechConfig()
        
        self.stt = SpeechToText(self.config.whisper_model)
        self.tts = TextToSpeech(self.config.tts_voice)
        self.recorder = AudioRecorder(
            sample_rate=self.config.sample_rate,
            silence_threshold=self.config.silence_threshold,
            silence_duration=self.config.silence_duration,
        )
    
    def listen(self) -> str:
        """Record and transcribe speech."""
        audio = self.recorder.record_until_silence()
        if len(audio) == 0:
            return ""
        return self.stt.transcribe_array(audio, self.config.sample_rate)
    
    def speak(self, text: str) -> None:
        """Speak text aloud."""
        self.tts.speak(text)
    
    def detect_wake_word(self, text: str) -> tuple:
        """
        Check if wake word is present.
        
        Returns: (detected: bool, command: str)
        """
        text_lower = text.lower().strip()
        wake = self.config.wake_word.lower()
        
        # Check variations
        for prefix in [f"{wake} ", f"{wake}, ", f"hey {wake} ", f"okay {wake} "]:
            if text_lower.startswith(prefix):
                return True, text[len(prefix):].strip()
        
        # Just the wake word
        if text_lower == wake:
            return True, ""
        
        return False, text
    
    def listen_for_command(self) -> Optional[str]:
        """
        Listen for a command with wake word.
        
        Returns command if wake word detected, else None.
        """
        text = self.listen()
        if not text:
            return None
        
        detected, command = self.detect_wake_word(text)
        if detected:
            return command if command else self.listen()  # Wait for follow-up
        
        return None


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EVA Speech Engine")
    subparsers = parser.add_subparsers(dest="command")
    
    # speak
    speak_p = subparsers.add_parser("speak")
    speak_p.add_argument("text")
    
    # listen
    listen_p = subparsers.add_parser("listen")
    
    # test
    test_p = subparsers.add_parser("test")
    
    args = parser.parse_args()
    
    if args.command == "speak":
        engine = SpeechEngine()
        engine.speak(args.text)
    elif args.command == "listen":
        engine = SpeechEngine()
        text = engine.listen()
        print(f"Heard: {text}")
    elif args.command == "test":
        print("Testing EVA Speech Engine")
        print("=" * 50)
        
        # Test TTS
        print("\n1. Testing TTS...")
        tts = TextToSpeech()
        tts.speak("Hello! I am EVA, your AI assistant.")
        
        # Test STT
        if HAS_AUDIO and HAS_WHISPER:
            print("\n2. Testing STT (say something)...")
            engine = SpeechEngine()
            text = engine.listen()
            print(f"You said: {text}")
        else:
            print("\n2. Skipping STT (missing dependencies)")
        
        print("\n✓ Test complete")
    else:
        parser.print_help()

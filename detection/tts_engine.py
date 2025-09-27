"""
Text-to-Speech Engine for ASL Translation System
Provides natural speech output for translated ASL text using gTTS and other TTS engines
"""

import os
import io
import tempfile
import threading
import queue
import warnings
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import logging

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# TTS imports
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not available. Install with: pip install gtts")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("Warning: pyttsx3 not available. Install with: pip install pyttsx3")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Install with: pip install pygame")

# Audio processing
try:
    import pydub
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Install with: pip install pydub")

class TTSEngine:
    """Main Text-to-Speech engine with multiple backend support"""
    
    def __init__(self, 
                 engine_type: str = "gtts",
                 language: str = "en",
                 slow: bool = False,
                 cache_dir: str = "data/tts_cache"):
        """
        Initialize the TTS engine
        
        Args:
            engine_type: Type of TTS engine ('gtts', 'pyttsx3', 'system')
            language: Language code for TTS
            slow: Whether to speak slowly
            cache_dir: Directory to cache audio files
        """
        self.engine_type = engine_type
        self.language = language
        self.slow = slow
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio queue for async playback
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread = None
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_speech_error: Optional[Callable] = None
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the selected engine
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the selected TTS engine"""
        if self.engine_type == "gtts" and GTTS_AVAILABLE:
            self._setup_gtts()
        elif self.engine_type == "pyttsx3" and PYTTSX3_AVAILABLE:
            self._setup_pyttsx3()
        elif self.engine_type == "system":
            self._setup_system_tts()
        else:
            self._setup_fallback()
    
    def _setup_gtts(self):
        """Setup Google Text-to-Speech"""
        try:
            self.engine = "gtts"
            self.logger.info("Google TTS engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize gTTS: {e}")
            self._setup_fallback()
    
    def _setup_pyttsx3(self):
        """Setup pyttsx3 engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice properties
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Set speech rate and volume
            self.engine.setProperty('rate', 150 if not self.slow else 100)
            self.engine.setProperty('volume', 0.9)
            
            self.logger.info("pyttsx3 engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize pyttsx3: {e}")
            self._setup_fallback()
    
    def _setup_system_tts(self):
        """Setup system TTS (Windows SAPI)"""
        try:
            if os.name == 'nt':  # Windows
                import win32com.client
                self.engine = win32com.client.Dispatch("SAPI.SpVoice")
                self.logger.info("Windows SAPI engine initialized")
            else:
                # Linux/Mac system TTS
                self.engine = "system"
                self.logger.info("System TTS engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize system TTS: {e}")
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup fallback TTS (print to console)"""
        self.engine = "fallback"
        self.logger.info("Fallback TTS (console output) initialized")
    
    def speak(self, text: str, async_play: bool = True) -> bool:
        """
        Convert text to speech and play it
        
        Args:
            text: Text to convert to speech
            async_play: Whether to play asynchronously
            
        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            return False
        
        # Clean text
        text = text.strip()
        
        if async_play:
            return self._speak_async(text)
        else:
            return self._speak_sync(text)
    
    def _speak_async(self, text: str) -> bool:
        """Speak text asynchronously"""
        try:
            # Add to queue
            self.audio_queue.put(text)
            
            # Start playback thread if not running
            if not self.is_playing:
                self.playback_thread = threading.Thread(target=self._playback_worker)
                self.playback_thread.daemon = True
                self.playback_thread.start()
            
            return True
        except Exception as e:
            self.logger.error(f"Async speech failed: {e}")
            if self.on_speech_error:
                self.on_speech_error(e)
            return False
    
    def _speak_sync(self, text: str) -> bool:
        """Speak text synchronously"""
        try:
            if self.on_speech_start:
                self.on_speech_start(text)
            
            success = self._generate_and_play(text)
            
            if self.on_speech_end:
                self.on_speech_end(text, success)
            
            return success
        except Exception as e:
            self.logger.error(f"Sync speech failed: {e}")
            if self.on_speech_error:
                self.on_speech_error(e)
            return False
    
    def _playback_worker(self):
        """Worker thread for async playback"""
        self.is_playing = True
        
        while True:
            try:
                # Get next text from queue
                text = self.audio_queue.get(timeout=1.0)
                
                if self.on_speech_start:
                    self.on_speech_start(text)
                
                # Generate and play audio
                success = self._generate_and_play(text)
                
                if self.on_speech_end:
                    self.on_speech_end(text, success)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                # No more items in queue
                break
            except Exception as e:
                self.logger.error(f"Playback worker error: {e}")
                if self.on_speech_error:
                    self.on_speech_error(e)
        
        self.is_playing = False
    
    def _generate_and_play(self, text: str) -> bool:
        """Generate audio and play it"""
        if self.engine == "gtts":
            return self._play_gtts(text)
        elif self.engine == "pyttsx3":
            return self._play_pyttsx3(text)
        elif self.engine == "system":
            return self._play_system(text)
        else:
            return self._play_fallback(text)
    
    def _play_gtts(self, text: str) -> bool:
        """Play using Google TTS"""
        try:
            # Check cache first
            cache_file = self.cache_dir / f"{hash(text)}.mp3"
            
            if not cache_file.exists():
                # Generate audio
                tts = gTTS(text=text, lang=self.language, slow=self.slow)
                tts.save(str(cache_file))
            
            # Play audio
            if PYGAME_AVAILABLE:
                try:
                    pygame.mixer.init()
                    pygame.mixer.music.load(str(cache_file))
                    pygame.mixer.music.play()
                    
                    # Wait for playback to complete
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(100)
                    
                    pygame.mixer.quit()
                except Exception as pygame_error:
                    self.logger.warning(f"Pygame playback failed: {pygame_error}")
                    # Fallback to system player
                    os.system(f"start {cache_file}" if os.name == 'nt' else f"play {cache_file}")
            elif PYDUB_AVAILABLE:
                try:
                    audio = AudioSegment.from_mp3(str(cache_file))
                    play(audio)
                except Exception as pydub_error:
                    self.logger.warning(f"Pydub playback failed: {pydub_error}")
                    # Fallback to system player
                    os.system(f"start {cache_file}" if os.name == 'nt' else f"play {cache_file}")
            else:
                # Fallback to system player
                os.system(f"start {cache_file}" if os.name == 'nt' else f"play {cache_file}")
            
            return True
        except Exception as e:
            self.logger.error(f"gTTS playback failed: {e}")
            return False
    
    def _play_pyttsx3(self, text: str) -> bool:
        """Play using pyttsx3"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            self.logger.error(f"pyttsx3 playback failed: {e}")
            return False
    
    def _play_system(self, text: str) -> bool:
        """Play using system TTS"""
        try:
            if os.name == 'nt':  # Windows
                self.engine.Speak(text)
            else:  # Linux/Mac
                os.system(f"espeak '{text}'" if os.system("which espeak") == 0 else f"say '{text}'")
            return True
        except Exception as e:
            self.logger.error(f"System TTS playback failed: {e}")
            return False
    
    def _play_fallback(self, text: str) -> bool:
        """Fallback: print to console"""
        print(f"🔊 TTS: {text}")
        return True
    
    def stop(self):
        """Stop current speech and clear queue"""
        try:
            # Clear queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Stop pygame if playing
            if PYGAME_AVAILABLE and pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            
            # Stop pyttsx3 if running
            if self.engine == "pyttsx3":
                self.engine.stop()
            
            self.is_playing = False
            self.logger.info("TTS stopped")
        except Exception as e:
            self.logger.error(f"Error stopping TTS: {e}")
    
    def set_voice_properties(self, rate: Optional[int] = None, volume: Optional[float] = None):
        """Set voice properties (for pyttsx3)"""
        if self.engine == "pyttsx3":
            try:
                if rate is not None:
                    self.engine.setProperty('rate', rate)
                if volume is not None:
                    self.engine.setProperty('volume', volume)
            except Exception as e:
                self.logger.error(f"Failed to set voice properties: {e}")
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        if self.engine == "pyttsx3":
            try:
                voices = self.engine.getProperty('voices')
                return [{'id': voice.id, 'name': voice.name} for voice in voices]
            except Exception as e:
                self.logger.error(f"Failed to get voices: {e}")
        return []
    
    def set_voice(self, voice_id: str):
        """Set specific voice (for pyttsx3)"""
        if self.engine == "pyttsx3":
            try:
                self.engine.setProperty('voice', voice_id)
            except Exception as e:
                self.logger.error(f"Failed to set voice: {e}")
    
    def clear_cache(self):
        """Clear TTS cache"""
        try:
            for file in self.cache_dir.glob("*.mp3"):
                file.unlink()
            self.logger.info("TTS cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")

def main():
    """Test the TTS engine"""
    tts = TTSEngine()
    
    # Test cases
    test_texts = [
        "Hello, welcome to ASL translation system!",
        "Thank you for using our service.",
        "How are you today?",
        "I love you very much.",
        "Good morning, have a great day!"
    ]
    
    print("Testing TTS Engine:")
    print("=" * 40)
    
    for text in test_texts:
        print(f"Speaking: {text}")
        success = tts.speak(text, async_play=False)
        print(f"Success: {success}")
        print("-" * 30)
    
    # Test async playback
    print("\nTesting async playback:")
    tts.speak("This is async speech test.")
    tts.speak("This is the second async message.")
    
    # Wait for async playback to complete
    import time
    time.sleep(5)
    
    tts.stop()

if __name__ == "__main__":
    main()

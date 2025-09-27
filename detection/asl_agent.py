"""
ASL Agent Orchestrator
Main agent that coordinates sign language detection, sentence building, LLM translation, and TTS
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import logging

# Import our custom modules
from .realtime_predictor import RealtimeASLPredictor
from .sentence_builder import SentenceBuilder
from .llm_translator import ASLTranslator
from .tts_engine import TTSEngine

class ASLAgent:
    """Main ASL Agent that orchestrates the complete sign-to-speech pipeline"""
    
    def __init__(self, 
                 model_path: str = "data/models",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ASL Agent
        
        Args:
            model_path: Path to the trained model files
            config: Configuration dictionary for the agent
        """
        self.model_path = Path(model_path)
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.predictor = None
        self.sentence_builder = None
        self.translator = None
        self.tts_engine = None
        
        # State management
        self.is_running = False
        self.is_recording = False
        self.current_session = None
        
        # Callbacks
        self.on_letter_detected: Optional[Callable] = None
        self.on_word_completed: Optional[Callable] = None
        self.on_sentence_completed: Optional[Callable] = None
        self.on_translation_ready: Optional[Callable] = None
        self.on_speech_started: Optional[Callable] = None
        self.on_speech_ended: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'letters_detected': 0,
            'words_completed': 0,
            'sentences_completed': 0,
            'translations_made': 0,
            'speech_events': 0,
            'session_start_time': None,
            'total_session_time': 0
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the agent
        self._initialize_components()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'detection': {
                'confidence_threshold': 0.7,
                'smoothing_frames': 5,
                'camera_index': 0,
                'frame_width': 640,
                'frame_height': 480
            },
            'sentence_building': {
                'word_timeout': 2.0,
                'sentence_timeout': 5.0,
                'min_word_length': 2,
                'confidence_threshold': 0.7
            },
            'translation': {
                'model_type': 'ollama',
                'model_name': 'llama3',
                'fallback_to_local': True
            },
            'tts': {
                'engine_type': 'gtts',
                'language': 'en',
                'slow': False,
                'async_play': True
            },
            'ui': {
                'show_confidence': True,
                'show_translation': True,
                'show_speech_status': True,
                'display_fps': True
            }
        }
    
    def _initialize_components(self):
        """Initialize all agent components"""
        try:
            # Initialize predictor
            self.predictor = RealtimeASLPredictor(str(self.model_path))
            
            # Initialize sentence builder
            sb_config = self.config['sentence_building']
            self.sentence_builder = SentenceBuilder(
                word_timeout=sb_config['word_timeout'],
                sentence_timeout=sb_config['sentence_timeout'],
                min_word_length=sb_config['min_word_length'],
                confidence_threshold=sb_config['confidence_threshold']
            )
            
            # Initialize translator
            trans_config = self.config['translation']
            self.translator = ASLTranslator(
                model_type=trans_config['model_type'],
                model_name=trans_config['model_name'],
                fallback_to_local=trans_config['fallback_to_local']
            )
            
            # Initialize TTS engine
            tts_config = self.config['tts']
            self.tts_engine = TTSEngine(
                engine_type=tts_config['engine_type'],
                language=tts_config['language'],
                slow=tts_config['slow']
            )
            
            # Setup TTS callbacks
            self.tts_engine.on_speech_start = self._on_speech_started
            self.tts_engine.on_speech_end = self._on_speech_ended
            self.tts_engine.on_speech_error = self._on_speech_error
            
            self.logger.info("ASL Agent components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            if self.on_error:
                self.on_error(e)
            raise
    
    def start_session(self) -> bool:
        """Start a new ASL translation session"""
        try:
            if self.is_running:
                self.logger.warning("Session already running")
                return False
            
            # Reset state
            self.is_running = True
            self.is_recording = False
            self.current_session = {
                'start_time': time.time(),
                'letters': [],
                'words': [],
                'sentences': [],
                'translations': []
            }
            
            # Reset statistics
            self.stats['session_start_time'] = time.time()
            self.stats['letters_detected'] = 0
            self.stats['words_completed'] = 0
            self.stats['sentences_completed'] = 0
            self.stats['translations_made'] = 0
            self.stats['speech_events'] = 0
            
            # Reset sentence builder
            self.sentence_builder.reset()
            
            self.logger.info("ASL Agent session started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start session: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    def stop_session(self) -> bool:
        """Stop the current ASL translation session"""
        try:
            if not self.is_running:
                self.logger.warning("No session running")
                return False
            
            # Stop recording
            self.is_recording = False
            
            # Finalize any pending sentence
            if self.sentence_builder:
                final_sentence = self.sentence_builder.force_finalize_sentence()
                if final_sentence:
                    self._process_sentence(final_sentence)
            
            # Stop TTS
            if self.tts_engine:
                self.tts_engine.stop()
            
            # Calculate session statistics
            if self.current_session:
                self.stats['total_session_time'] = time.time() - self.current_session['start_time']
            
            self.is_running = False
            self.current_session = None
            
            self.logger.info("ASL Agent session stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop session: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    def start_recording(self):
        """Start recording ASL gestures"""
        if self.is_running and not self.is_recording:
            self.is_recording = True
            self.logger.info("Recording started")
    
    def stop_recording(self):
        """Stop recording ASL gestures"""
        if self.is_recording:
            self.is_recording = False
            self.logger.info("Recording stopped")
    
    def run_realtime(self):
        """Run real-time ASL detection and translation"""
        if not self.start_session():
            return
        
        try:
            # Initialize camera
            cap = cv2.VideoCapture(self.config['detection']['camera_index'])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['detection']['frame_width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['detection']['frame_height'])
            
            if not cap.isOpened():
                raise Exception("Cannot open camera")
            
            # Initialize MediaPipe
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            mp_drawing = mp.solutions.drawing_utils
            
            self.logger.info("🎥 Real-time ASL Agent Started!")
            print("Press 'SPACE' to start/stop recording, 'ESC' to quit")
            
            # Start recording by default
            self.start_recording()
            
            ptime = 0
            prediction_history = []
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                # Draw hand landmarks and make predictions
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                        )
                        
                        # Extract landmarks
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                            landmarks.append([0, x, y])
                        
                        # Get bounding box
                        x_coords = [lm[1] for lm in landmarks]
                        y_coords = [lm[2] for lm in landmarks]
                        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                        
                        # Extract features and predict
                        features = self.predictor.extract_features(landmarks, bbox)
                        
                        if features and self.is_recording:
                            prediction, confidence = self.predictor.predict_gesture(features)
                            
                            if prediction is not None and confidence >= self.config['detection']['confidence_threshold']:
                                # Add to history for smoothing
                                prediction_history.append((prediction, confidence))
                                if len(prediction_history) > self.config['detection']['smoothing_frames']:
                                    prediction_history.pop(0)
                                
                                # Get most common prediction
                                if prediction_history:
                                    predictions = [p[0] for p in prediction_history]
                                    avg_confidence = np.mean([p[1] for p in prediction_history])
                                    final_prediction = max(set(predictions), key=predictions.count)
                                    
                                    # Process the detected letter
                                    self._process_letter(final_prediction, avg_confidence)
                
                # Update UI
                self._update_ui(frame, ptime)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == 32:  # SPACE
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                
                # Update FPS
                ctime = time.time()
                ptime = ctime
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.logger.error(f"Real-time processing error: {e}")
            if self.on_error:
                self.on_error(e)
        finally:
            self.stop_session()
    
    def _process_letter(self, letter: str, confidence: float):
        """Process a detected letter"""
        try:
            # Add to sentence builder
            word, sentence = self.sentence_builder.add_letter(letter, confidence)
            
            # Update statistics
            self.stats['letters_detected'] += 1
            
            # Store in session
            if self.current_session:
                self.current_session['letters'].append({
                    'letter': letter,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
            
            # Callback for letter detection
            if self.on_letter_detected:
                self.on_letter_detected(letter, confidence)
            
            # Process completed word
            if word:
                self._process_word(word)
            
            # Process completed sentence
            if sentence:
                self._process_sentence(sentence)
                
        except Exception as e:
            self.logger.error(f"Error processing letter: {e}")
            if self.on_error:
                self.on_error(e)
    
    def _process_word(self, word: str):
        """Process a completed word"""
        try:
            # Update statistics
            self.stats['words_completed'] += 1
            
            # Store in session
            if self.current_session:
                self.current_session['words'].append({
                    'word': word,
                    'timestamp': time.time()
                })
            
            # Callback for word completion
            if self.on_word_completed:
                self.on_word_completed(word)
            
            self.logger.info(f"Word completed: {word}")
            
        except Exception as e:
            self.logger.error(f"Error processing word: {e}")
            if self.on_error:
                self.on_error(e)
    
    def _process_sentence(self, sentence: List[str]):
        """Process a completed sentence"""
        try:
            # Update statistics
            self.stats['sentences_completed'] += 1
            
            # Store in session
            if self.current_session:
                self.current_session['sentences'].append({
                    'sentence': sentence,
                    'timestamp': time.time()
                })
            
            # Callback for sentence completion
            if self.on_sentence_completed:
                self.on_sentence_completed(sentence)
            
            # Translate the sentence
            self._translate_and_speak(sentence)
            
            self.logger.info(f"Sentence completed: {' '.join(sentence)}")
            
        except Exception as e:
            self.logger.error(f"Error processing sentence: {e}")
            if self.on_error:
                self.on_error(e)
    
    def _translate_and_speak(self, sentence: List[str]):
        """Translate sentence and convert to speech"""
        try:
            # Convert sentence to ASL text
            asl_text = ' '.join(sentence)
            
            # Translate using LLM
            translation_result = self.translator.translate(asl_text)
            
            if translation_result and translation_result['translation']:
                # Update statistics
                self.stats['translations_made'] += 1
                
                # Store in session
                if self.current_session:
                    self.current_session['translations'].append({
                        'asl_text': asl_text,
                        'translation': translation_result['translation'],
                        'confidence': translation_result['confidence'],
                        'timestamp': time.time()
                    })
                
                # Callback for translation ready
                if self.on_translation_ready:
                    self.on_translation_ready(translation_result)
                
                # Convert to speech
                success = self.tts_engine.speak(
                    translation_result['translation'],
                    async_play=self.config['tts']['async_play']
                )
                
                if success:
                    self.stats['speech_events'] += 1
                
                self.logger.info(f"Translation: {translation_result['translation']}")
            
        except Exception as e:
            self.logger.error(f"Error in translation and speech: {e}")
            if self.on_error:
                self.on_error(e)
    
    def _update_ui(self, frame, ptime):
        """Update the user interface"""
        try:
            ui_config = self.config['ui']
            
            # Get current state
            state = self.sentence_builder.get_current_state()
            
            # Display current word
            if state['current_word']:
                cv2.putText(frame, f"Current: {state['current_word']}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display current sentence
            if state['current_sentence']:
                sentence_text = ' '.join(state['current_sentence'])
                cv2.putText(frame, f"Sentence: {sentence_text}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display recording status
            status_color = (0, 255, 0) if self.is_recording else (0, 0, 255)
            status_text = "RECORDING" if self.is_recording else "PAUSED"
            cv2.putText(frame, status_text, (10, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Display statistics
            if ui_config['show_confidence']:
                cv2.putText(frame, f"Letters: {self.stats['letters_detected']}", (10, 120),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Words: {self.stats['words_completed']}", (10, 140),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Sentences: {self.stats['sentences_completed']}", (10, 160),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display FPS
            if ui_config['display_fps']:
                ctime = time.time()
                fps = 1.0 / (ctime - ptime) if ptime else 0.0
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 180),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Display frame
            cv2.imshow("ASL Agent - Sign to Speech", frame)
            
        except Exception as e:
            self.logger.error(f"UI update error: {e}")
    
    def _on_speech_started(self, text: str):
        """Callback for speech started"""
        if self.on_speech_started:
            self.on_speech_started(text)
    
    def _on_speech_ended(self, text: str, success: bool):
        """Callback for speech ended"""
        if self.on_speech_ended:
            self.on_speech_ended(text, success)
    
    def _on_speech_error(self, error: Exception):
        """Callback for speech error"""
        if self.on_error:
            self.on_error(error)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return self.stats.copy()
    
    def get_session_data(self) -> Optional[Dict[str, Any]]:
        """Get current session data"""
        return self.current_session.copy() if self.current_session else None
    
    def save_session(self, filepath: str):
        """Save current session data to file"""
        try:
            session_data = {
                'session': self.current_session,
                'stats': self.stats,
                'config': self.config
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.logger.info(f"Session saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
            if self.on_error:
                self.on_error(e)

def main():
    """Test the ASL Agent"""
    print("🤖 ASL Agent - Sign to Speech Translation")
    print("=" * 50)
    
    # Create agent
    agent = ASLAgent()
    
    # Setup callbacks for testing
    def on_letter_detected(letter, confidence):
        print(f"Letter detected: {letter} (confidence: {confidence:.2f})")
    
    def on_word_completed(word):
        print(f"Word completed: {word}")
    
    def on_sentence_completed(sentence):
        print(f"Sentence completed: {' '.join(sentence)}")
    
    def on_translation_ready(translation):
        print(f"Translation: {translation['translation']}")
    
    def on_speech_started(text):
        print(f"🔊 Speaking: {text}")
    
    def on_speech_ended(text, success):
        print(f"✅ Speech completed: {success}")
    
    def on_error(error):
        print(f"❌ Error: {error}")
    
    # Set callbacks
    agent.on_letter_detected = on_letter_detected
    agent.on_word_completed = on_word_completed
    agent.on_sentence_completed = on_sentence_completed
    agent.on_translation_ready = on_translation_ready
    agent.on_speech_started = on_speech_started
    agent.on_speech_ended = on_speech_ended
    agent.on_error = on_error
    
    # Run real-time detection
    try:
        agent.run_realtime()
    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user")
    finally:
        agent.stop_session()
        print("Session ended")

if __name__ == "__main__":
    main()

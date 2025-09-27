import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import time
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our enhanced components
from llm_translator import ASLTranslator
from tts_engine import TTSEngine

class RealtimeASLPredictor:
    def __init__(self, model_path="data/models"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.hands = None
        self.mp_drawing = None
        
        # Initialize components
        self.translator = ASLTranslator()
        self.tts_engine = TTSEngine()
        
        # Simple letter collection
        self.letter_sequence = []
        
        # Statistics
        self.stats = {
            'letters_detected': 0,
            'translations_made': 0,
            'speech_events': 0
        }
        
        # Initialize hand tracking
        self.init_hand_tracking()
        
        # Load model
        self.load_model()
    
    def init_hand_tracking(self):
        """Initialize MediaPipe hand tracking"""
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load(self.model_path / 'asl_model.pkl')
            self.scaler = joblib.load(self.model_path / 'scaler.pkl')
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print("❌ Model not found. Train the model first!")
            print("Run: python detection/simple_model.py")
            return False
        return True
    
    def extract_features(self, landmarks, bbox):
        """Extract features from hand landmarks"""
        if not landmarks or not bbox:
            return None
        
        xmin, ymin, xmax, ymax = bbox
        bw = max(1, xmax - xmin)
        bh = max(1, ymax - ymin)
        
        features = []
        for _, x, y in landmarks:
            x_norm = (x - xmin) / bw
            y_norm = (y - ymin) / bh
            x_norm = max(0.0, min(1.0, float(x_norm)))
            y_norm = max(0.0, min(1.0, float(y_norm)))
            features.extend([x_norm, y_norm])
        
        return features
    
    def predict_gesture(self, features):
        """Predict gesture from features"""
        if self.model is None or self.scaler is None:
            return None, 0.0
        
        try:
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            confidence = self.model.predict_proba(features_scaled)[0].max()
            
            return prediction, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def run(self):
        """Run real-time prediction"""
        if not self.load_model():
            return
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print("🎥 Real-time ASL Agent Started!")
        print("Controls:")
        print("  - SPACE: Start/stop recording")
        print("  - ENTER: Send letters to LLM for translation")
        print("  - ESC: Quit")
        print("")
        print("How it works:")
        print("  1. Sign letters one by one (hold each letter for 3+ seconds)")
        print("  2. System will add letters to sequence when confident")
        print("  3. Press ENTER when done to send to LLM")
        print("  4. LLM figures out words and makes natural English")
        print("  5. System speaks the translation aloud")
        print("")
        print("Example: Sign H-I-D-E, then press ENTER → LLM outputs 'Hide'")
        print("")
        print("Debug info will show why letters are/aren't added")
        
        ptime = 0
        prediction_history = []
        is_recording = True
        
        # Letter detection debouncing
        last_letter = None
        last_letter_time = 0
        letter_debounce_time = 3.0  # Wait 3 seconds before detecting same letter again
        min_confidence = 0.85  # Slightly lower confidence threshold
        prediction_history = []  # Track recent predictions for stability

        # Gesture transition detection
        no_hand_frames = 0
        min_no_hand_frames = 15  # Need 15 frames without hand to detect transition
        
        # Track when we last saw a hand to detect transitions
        last_hand_time = 0
        hand_was_present = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                no_hand_frames = 0  # Reset no-hand counter
                
                # Check if this is a new hand appearance (transition)
                is_new_hand = not hand_was_present  # Hand just appeared
                if is_new_hand:
                    print(f"   Debug: Hand appeared - new gesture starting")
                
                hand_was_present = True
                last_hand_time = time.time()  # Update last hand time
                
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )
                    
                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        landmarks.append([0, x, y])  # [id, x, y] format
                    
                    # Get bounding box
                    x_coords = [lm[1] for lm in landmarks]
                    y_coords = [lm[2] for lm in landmarks]
                    bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                    
                    # Extract features
                    features = self.extract_features(landmarks, bbox)
                    
                    if features and is_recording:
                        # Make prediction
                        prediction, confidence = self.predict_gesture(features)
                        
                        if prediction is not None and confidence > min_confidence:
                            # If this is a new hand appearance, clear prediction history
                            if is_new_hand:
                                prediction_history = []
                                print(f"   Debug: New hand detected, cleared prediction history")
                            
                            # Add to history for smoothing
                            prediction_history.append((prediction, confidence))
                            if len(prediction_history) > 10:  # Keep more history for stability
                                prediction_history.pop(0)
                            
                            # Get most common prediction from recent history
                            if len(prediction_history) >= 5:  # Need at least 5 predictions
                                recent_predictions = prediction_history[-5:]  # Last 5 predictions
                                predictions = [p[0] for p in recent_predictions]
                                avg_confidence = np.mean([p[1] for p in recent_predictions])
                                
                                # Only accept if we have consistent predictions
                                most_common = max(set(predictions), key=predictions.count)
                                if predictions.count(most_common) >= 3:  # At least 3 out of 5 agree
                                    final_prediction = most_common
                                    
                                    # Debug: Show what predictions we're seeing
                                    print(f"   Debug: Recent predictions: {predictions}")
                                    print(f"   Debug: Most common: {most_common} (count: {predictions.count(most_common)})")
                                    print(f"   Debug: Current prediction: {prediction}")
                                    
                                    # Check if this is a new letter (debouncing)
                                    current_time = time.time()
                                    time_since_last = current_time - last_letter_time
                                    
                                    # Only add letter if:
                                    # 1. It's different from last letter AND enough time has passed, OR
                                    # 2. It's the same letter but enough time has passed (for repeats)
                                    is_new_letter = (
                                        (final_prediction != last_letter and time_since_last > letter_debounce_time) or
                                        (final_prediction == last_letter and time_since_last > letter_debounce_time * 2)
                                    )
                                    
                                    if is_new_letter and avg_confidence > min_confidence:
                                        # Add letter to sequence
                                        self.letter_sequence.append(final_prediction)
                                        self.stats['letters_detected'] += 1
                                        print(f"📝 Letter added: {final_prediction} (Sequence: {''.join(self.letter_sequence)})")
                                        print(f"   Debug: last_letter={last_letter}, time_since_last={time_since_last:.2f}s")
                                        
                                        # Update tracking
                                        last_letter = final_prediction
                                        last_letter_time = current_time
                                        
                                        # Clear prediction history after adding letter
                                        prediction_history = []
                                        
                                        # Reset hand transition detection
                                        no_hand_frames = 0
                                        
                                        # Skip the rest of the frame processing to avoid duplicate processing
                                        break
                                    else:
                                        # Debug why letter wasn't added
                                        if not is_new_letter:
                                            print(f"   Debug: Letter {final_prediction} not added - debouncing (last: {last_letter}, time: {time_since_last:.2f}s)")
                                        if avg_confidence <= min_confidence:
                                            print(f"   Debug: Letter {final_prediction} not added - low confidence ({avg_confidence:.2f} <= {min_confidence})")
                                
                                # Display prediction
                                color = (0, 255, 0) if avg_confidence > min_confidence else (0, 255, 255)
                                cv2.putText(frame, f"Letter: {final_prediction}", (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                                cv2.putText(frame, f"Confidence: {avg_confidence:.2f}", (10, 70),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                                
                                # Show debounce status
                                if final_prediction == last_letter:
                                    time_left = letter_debounce_time - (current_time - last_letter_time)
                                    cv2.putText(frame, f"Debounce: {time_left:.1f}s", (10, 100),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                # No hand detected
                no_hand_frames += 1
                hand_was_present = False  # Mark hand as not present
                
                if no_hand_frames >= min_no_hand_frames:
                    # Reset letter tracking after no hand for a while
                    if last_letter is not None:
                        print(f"🔄 Hand transition detected - ready for next letter")
                        last_letter = None
                        last_letter_time = 0
            
            # Display current letter sequence
            y_offset = 130  # Start below the debounce timer
            
            # Display recording status
            status_color = (0, 255, 0) if is_recording else (0, 0, 255)
            status_text = "RECORDING" if is_recording else "PAUSED"
            cv2.putText(frame, status_text, (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            y_offset += 30
            
            # Display current letter sequence
            current_sequence = ''.join(self.letter_sequence)
            cv2.putText(frame, f"Letters: {current_sequence}", (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
            
            # Display instructions
            cv2.putText(frame, "Press ENTER to translate", (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
            
            # Display last letter info
            if last_letter:
                time_since_last = time.time() - last_letter_time
                cv2.putText(frame, f"Last: {last_letter} ({time_since_last:.1f}s ago)", (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                y_offset += 15
                
                # Show debounce status
                if time_since_last < letter_debounce_time:
                    cv2.putText(frame, f"Debouncing... ({letter_debounce_time - time_since_last:.1f}s)", (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    y_offset += 15
            
            # Display statistics
            cv2.putText(frame, f"Letters: {self.stats['letters_detected']}", (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Translations: {self.stats['translations_made']}", (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Speech Events: {self.stats['speech_events']}", (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            # FPS
            ctime = time.time()
            fps = 1.0 / (ctime - ptime) if ptime else 0.0
            ptime = ctime
            cv2.putText(frame, f"FPS: {int(fps)}", (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Display frame
            cv2.imshow("ASL Agent - Sign to Speech", frame)
            
            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                is_recording = not is_recording
                print(f"Recording {'started' if is_recording else 'paused'}")
            elif key == 13:  # ENTER - Send letters to LLM
                if self.letter_sequence:
                    self._send_to_llm()
                else:
                    print("No letters to translate")
        
        # Send any remaining letters to LLM
        if self.letter_sequence:
            self._send_to_llm()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n📊 Session Statistics:")
        print(f"  Letters detected: {self.stats['letters_detected']}")
        print(f"  Translations made: {self.stats['translations_made']}")
        print(f"  Speech events: {self.stats['speech_events']}")
        
        print("👋 ASL Agent session ended!")
        print("🌟 Mission accomplished: Communication barriers bridged!")
    
    def _send_to_llm(self):
        """Send current letter sequence to LLM for translation"""
        try:
            if not self.letter_sequence:
                print("No letters to translate")
                return
            
            # Convert letter sequence to string
            letter_string = ''.join(self.letter_sequence)
            print(f"🤖 Sending to LLM for translation...")
            print(f"   Input letters: {letter_string}")

            # Send to LLM for translation
            translation_result = self.translator.translate(letter_string)

            if translation_result and translation_result['translation']:
                self.stats['translations_made'] += 1
                corrected_english = translation_result['translation']
                print(f"✅ LLM translated:")
                print(f"   Raw letters: {letter_string}")
                print(f"   Natural English: {corrected_english}")
                print(f"   Confidence: {translation_result['confidence']:.2f}")

                # Speak the translation
                success = self.tts_engine.speak(
                    corrected_english,
                    async_play=True
                )

                if success:
                    self.stats['speech_events'] += 1
                    print(f"🔊 Speaking: {corrected_english}")
                
                # Clear the sequence for next input
                self.letter_sequence = []
                print("📝 Letter sequence cleared - ready for next input")
            else:
                print(f"❌ LLM translation failed")

        except Exception as e:
            print(f"Error in translation and speech: {e}")

def main():
    predictor = RealtimeASLPredictor()
    predictor.run()

if __name__ == "__main__":
    main()

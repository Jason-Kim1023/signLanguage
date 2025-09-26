import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import time

class RealtimeASLPredictor:
    def __init__(self, model_path="data/models"):
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = None
        self.hands = None
        self.mp_drawing = None
        
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
        
        print("🎥 Real-time ASL Prediction Started!")
        print("Press 'ESC' to quit")
        
        ptime = 0
        prediction_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
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
                    
                    if features:
                        # Make prediction
                        prediction, confidence = self.predict_gesture(features)
                        
                        if prediction is not None:
                            # Add to history for smoothing
                            prediction_history.append((prediction, confidence))
                            if len(prediction_history) > 5:
                                prediction_history.pop(0)
                            
                            # Get most common prediction
                            if prediction_history:
                                predictions = [p[0] for p in prediction_history]
                                avg_confidence = np.mean([p[1] for p in prediction_history])
                                final_prediction = max(set(predictions), key=predictions.count)
                                
                                # Display prediction
                                color = (0, 255, 0) if avg_confidence > 0.7 else (0, 255, 255)
                                cv2.putText(frame, f"Prediction: {final_prediction}", (10, 30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                                cv2.putText(frame, f"Confidence: {avg_confidence:.2f}", (10, 70),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # FPS
            ctime = time.time()
            fps = 1.0 / (ctime - ptime) if ptime else 0.0
            ptime = ctime
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 110),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            # Display frame
            cv2.imshow("ASL Real-time Prediction", frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Prediction session ended!")

def main():
    predictor = RealtimeASLPredictor()
    predictor.run()

if __name__ == "__main__":
    main()

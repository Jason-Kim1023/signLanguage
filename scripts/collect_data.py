"""
Enhanced ASL Data Collection Script
Collects hand landmark data for ASL letter training
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class ASLDataCollector:
    def __init__(self, output_file="data/raw/asl_letters.csv"):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Data collection
        self.current_label = 'A'
        self.samples_collected = 0
        self.auto_recording = False
        self.last_save_time = 0
        self.save_interval = 0.5  # Save every 0.5 seconds
        
        # Load existing data
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing data if file exists"""
        if self.output_file.exists():
            self.df = pd.read_csv(self.output_file)
            self.samples_collected = len(self.df)
            print(f"📊 Loaded {self.samples_collected} existing samples")
        else:
            self.df = pd.DataFrame()
            print("📊 Starting fresh data collection")
    
    def extract_landmarks(self, landmarks):
        """Extract normalized landmark coordinates"""
        if not landmarks:
            return None
        
        # Get bounding box
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Normalize coordinates relative to bounding box
        normalized_landmarks = []
        for lm in landmarks.landmark:
            norm_x = (lm.x - x_min) / (x_max - x_min) if x_max != x_min else 0
            norm_y = (lm.y - y_min) / (y_max - y_min) if y_max != y_min else 0
            normalized_landmarks.extend([norm_x, norm_y])
        
        return normalized_landmarks
    
    def save_sample(self, landmarks):
        """Save a single sample"""
        if not landmarks:
            return
        
        # Extract features
        features = self.extract_landmarks(landmarks)
        if not features:
            return
        
        # Create sample data
        sample_data = {
            'label': self.current_label,
            'timestamp': time.time()
        }
        
        # Add landmark features
        for i, feature in enumerate(features):
            sample_data[f'landmark_{i}'] = feature
        
        # Add to DataFrame
        new_row = pd.DataFrame([sample_data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        self.samples_collected += 1
        print(f"💾 Saved sample {self.samples_collected} for letter '{self.current_label}'")
    
    def save_to_file(self):
        """Save all data to CSV file"""
        self.df.to_csv(self.output_file, index=False)
        print(f"💾 Data saved to: {self.output_file}")
    
    def run(self):
        """Main data collection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        print("🎥 ASL Data Collection Started!")
        print("Controls:")
        print("  - Press letter keys (A-Z) to set current label")
        print("  - Press 'S' to save a single sample")
        print("  - Press 'R' to toggle auto-recording")
        print("  - Press 'ESC' to quit and save")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Display information
            cv2.putText(frame, f"Current Label: {self.current_label}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {self.samples_collected}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Auto-recording: {'ON' if self.auto_recording else 'OFF'}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if self.auto_recording else (0, 0, 255), 2)
            
            # Auto-recording
            if self.auto_recording and results.multi_hand_landmarks:
                current_time = time.time()
                if current_time - self.last_save_time > self.save_interval:
                    self.save_sample(results.multi_hand_landmarks[0])
                    self.last_save_time = current_time
            
            # Display frame
            cv2.imshow("ASL Data Collection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('s'):  # Save single sample
                if results.multi_hand_landmarks:
                    self.save_sample(results.multi_hand_landmarks[0])
            elif key == ord('r'):  # Toggle auto-recording
                self.auto_recording = not self.auto_recording
                print(f"🔄 Auto-recording: {'ON' if self.auto_recording else 'OFF'}")
            elif 97 <= key <= 122:  # Letter keys (a-z)
                self.current_label = chr(key).upper()
                print(f"📝 Current label set to: {self.current_label}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save final data
        self.save_to_file()
        print(f"\n🎉 Data collection complete!")
        print(f"📊 Total samples collected: {self.samples_collected}")
        print(f"💾 Data saved to: {self.output_file}")

def main():
    """Main function"""
    collector = ASLDataCollector()
    collector.run()

if __name__ == "__main__":
    main()

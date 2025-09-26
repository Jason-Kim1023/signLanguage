import cv2
import mediapipe as mp
import time
import math
import csv
import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# -----------------------------
# Enhanced Hand Tracking
# -----------------------------
class EnhancedHandTracking:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.lmsList = []

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList, yList = [], []
        bbox, self.lmsList = [], []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = frame.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx); yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = (xmin, ymin, xmax, ymax)
            if draw:
                cv2.rectangle(frame, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)
        return self.lmsList, bbox

# -----------------------------
# Enhanced Feature Extraction
# -----------------------------
def to_feature_vector(lmsList, bbox, frame_shape):
    if not lmsList or not bbox:
        return None

    xmin, ymin, xmax, ymax = bbox
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)

    feats = []
    for _, x, y in lmsList:
        x_norm = (x - xmin) / bw
        y_norm = (y - ymin) / bh
        x_norm = max(0.0, min(1.0, float(x_norm)))
        y_norm = max(0.0, min(1.0, float(y_norm)))
        feats.extend([x_norm, y_norm])

    return feats  # 42 values

# -----------------------------
# Dataset Management
# -----------------------------
class DatasetManager:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.metadata_dir = self.data_dir / "metadata"
        
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.csv_path = self.raw_data_dir / "asl_letters.csv"
        self.metadata_path = self.metadata_dir / "dataset_metadata.json"
        
        # Initialize metadata
        self.metadata = self.load_metadata()
        
        # Header for CSV
        self.header = [f"k{i:02d}" for i in range(42)] + ["label", "timestamp", "session_id"]
    
    def load_metadata(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "total_samples": 0,
                "samples_per_class": {},
                "sessions": [],
                "last_updated": None,
                "target_samples_per_class": 100
            }
    
    def save_metadata(self):
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def ensure_csv_header(self):
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)
    
    def add_sample(self, features, label, session_id):
        timestamp = datetime.now().isoformat()
        
        # Add to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(features + [label, timestamp, session_id])
        
        # Update metadata
        self.metadata["total_samples"] += 1
        if label not in self.metadata["samples_per_class"]:
            self.metadata["samples_per_class"][label] = 0
        self.metadata["samples_per_class"][label] += 1
        
        self.save_metadata()
    
    def get_stats(self):
        return {
            "total_samples": self.metadata["total_samples"],
            "samples_per_class": self.metadata["samples_per_class"],
            "target_samples": self.metadata["target_samples_per_class"],
            "completion_percentage": self.get_completion_percentage()
        }
    
    def get_completion_percentage(self):
        if not self.metadata["samples_per_class"]:
            return 0
        
        total_target = len(self.metadata["samples_per_class"]) * self.metadata["target_samples_per_class"]
        total_collected = sum(self.metadata["samples_per_class"].values())
        return (total_collected / total_target) * 100 if total_target > 0 else 0

# -----------------------------
# Enhanced Data Collector
# -----------------------------
class EnhancedDataCollector:
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        # Initialize hand detector
        self.detector = EnhancedHandTracking(detectionCon=0.6, trackCon=0.6)
        
        # State variables
        self.current_label = "A"
        self.record = False
        self.frame_skip = 0
        self.ptime = 0
        
        # UI settings
        self.show_help = True
        self.colors = {
            'fps': (255, 0, 255),
            'label': (0, 255, 0),
            'rec_on': (0, 0, 255),
            'rec_off': (100, 100, 100),
            'samples': (255, 255, 0),
            'help': (255, 255, 255),
            'warning': (0, 255, 255)
        }
        
        # Ensure CSV header
        self.dataset_manager.ensure_csv_header()
    
    def draw_ui(self, frame, stats):
        h, w = frame.shape[:2]
        
        # FPS
        ctime = time.time()
        fps = 1.0 / (ctime - self.ptime) if self.ptime else 0.0
        self.ptime = ctime
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, self.colors['fps'], 2)
        
        # Current label
        cv2.putText(frame, f"Label: {self.current_label}", (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, self.colors['label'], 2)
        
        # Recording status
        rec_color = self.colors['rec_on'] if self.record else self.colors['rec_off']
        cv2.putText(frame, f"REC: {'ON' if self.record else 'OFF'}", (10, 90),
                    cv2.FONT_HERSHEY_PLAIN, 2, rec_color, 2)
        
        # Sample count for current label
        current_count = stats['samples_per_class'].get(self.current_label, 0)
        cv2.putText(frame, f"{self.current_label} Samples: {current_count}", (10, 120),
                    cv2.FONT_HERSHEY_PLAIN, 2, self.colors['samples'], 2)
        
        # Progress bar
        target = stats['target_samples']
        progress = min(current_count / target, 1.0) if target > 0 else 0
        bar_width = 200
        bar_height = 20
        bar_x, bar_y = 10, 150
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Progress text
        cv2.putText(frame, f"{current_count}/{target}", (bar_x + bar_width + 10, bar_y + 15),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, self.colors['samples'], 2)
        
        # Overall stats
        cv2.putText(frame, f"Total: {stats['total_samples']}", (10, 200),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, self.colors['help'], 2)
        cv2.putText(frame, f"Completion: {stats['completion_percentage']:.1f}%", (10, 220),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, self.colors['help'], 2)
        
        # Help text
        if self.show_help:
            help_text = [
                "Controls:",
                "A-Z: Set label",
                "1: Save sample",
                "2: Toggle auto-record",
                "3: Toggle help",
                "ESC: Quit"
            ]
            
            for i, text in enumerate(help_text):
                cv2.putText(frame, text, (w - 200, 30 + i * 25),
                            cv2.FONT_HERSHEY_PLAIN, 1, self.colors['help'], 1)
    
    def run(self):
        print("Enhanced ASL Data Collector")
        print("=" * 50)
        print("Starting data collection session...")
        print(f"Session ID: {self.session_id}")
        print("=" * 50)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            frame = self.detector.findFingers(frame, draw=True)
            lmsList, bbox = self.detector.findPosition(frame, handNo=0, draw=True)
            
            # Extract features
            feats = None
            if lmsList and bbox:
                feats = to_feature_vector(lmsList, bbox, frame.shape)
            
            # Get current stats
            stats = self.dataset_manager.get_stats()
            
            # Draw UI
            self.draw_ui(frame, stats)
            
            # Auto-record logic
            if self.record and feats is not None:
                self.frame_skip = (self.frame_skip + 1) % 3
                if self.frame_skip == 0:
                    self.dataset_manager.add_sample(feats, self.current_label, self.session_id)
                    print(f"Auto-saved {self.current_label} sample")
            
            # Display frame
            cv2.imshow("Enhanced ASL Data Collector", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord("1") and feats is not None:
                self.dataset_manager.add_sample(feats, self.current_label, self.session_id)
                print(f"Saved {self.current_label} sample")
            elif key == ord("2"):
                self.record = not self.record
                print(f"Auto-record: {'ON' if self.record else 'OFF'}")
            elif key == ord("3"):
                self.show_help = not self.show_help
            elif 97 <= key <= 122:  # a-z
                self.current_label = chr(key).upper()
                print(f"Changed label to: {self.current_label}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        final_stats = self.dataset_manager.get_stats()
        print("\n" + "=" * 50)
        print("Session Complete!")
        print(f"Total samples collected: {final_stats['total_samples']}")
        print(f"Completion: {final_stats['completion_percentage']:.1f}%")
        print("=" * 50)

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    try:
        collector = EnhancedDataCollector()
        collector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is connected and not being used by another application.")

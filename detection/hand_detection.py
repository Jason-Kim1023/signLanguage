import cv2
import mediapipe as mp
import time
import math
import csv
import os
from pathlib import Path

# -----------------------------
# Hand Tracking
# -----------------------------
class HandTrackingDynamic:
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
# Feature extraction
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
# Recording config
# -----------------------------
SAVE_DIR = Path("data")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = SAVE_DIR / "asl_letters.csv"

header = [f"k{i:02d}" for i in range(42)] + ["label"]

def ensure_csv_header(path):
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

# -----------------------------
# Main loop
# -----------------------------
def main():
    ensure_csv_header(CSV_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    detector = HandTrackingDynamic(detectionCon=0.6, trackCon=0.6)
    ptime = 0
    frame_skip = 0

    current_label = "A"  # default
    record = False

    # Track per-letter counts (A–Z)
    letter_counts = {chr(c): 0 for c in range(65, 91)}

    print("[Controls] Press any letter key (a–z) to set label, S = save sample, R = toggle auto-record, ESC = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detector.findFingers(frame, draw=True)
        lmsList, bbox = detector.findPosition(frame, handNo=0, draw=True)

        ctime = time.time()
        fps = 1.0 / (ctime - ptime) if ptime else 0.0
        ptime = ctime
        cv2.putText(frame, f"{int(fps)} FPS", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.putText(frame, f"Label: {current_label}", (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(frame, f"REC: {'ON' if record else 'OFF'}", (10, 90),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255) if record else (100, 100, 100), 2)
        cv2.putText(frame, f"{current_label} Samples: {letter_counts[current_label]}", (10, 120),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        feats = None
        if lmsList and bbox:
            feats = to_feature_vector(lmsList, bbox, frame.shape)

        # Auto-record mode
        if record and feats is not None:
            frame_skip = (frame_skip + 1) % 3
            if frame_skip == 0:
                with open(CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(feats + [current_label])
                letter_counts[current_label] += 1

        cv2.imshow("ASL Letter Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord("s") and feats is not None:
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(feats + [current_label])
            letter_counts[current_label] += 1
            print(f"Saved {current_label}, total for {current_label}: {letter_counts[current_label]}")
        elif key == ord("r"):
            record = not record
        elif 97 <= key <= 122:  # a–z
            current_label = chr(key).upper()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

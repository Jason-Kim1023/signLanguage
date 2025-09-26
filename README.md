# Sign Language Detection Project

A real-time sign language recognition tool using hand tracking and machine learning. This project combines OpenCV for real-time hand detection, MediaPipe for hand landmark extraction, and TensorFlow/Keras for neural network training and prediction.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe to detect and track hand landmarks
- **Data Collection**: Interactive tool for collecting ASL letter samples
- **Machine Learning**: Neural network training with TensorFlow/Keras
- **Jupyter Notebook**: Interactive development and model training

## Quick Setup

### Option 1: Automated Setup (Recommended)

**For Windows (PowerShell):**
```powershell
.\setup.ps1
```

**For Windows (Command Prompt):**
```cmd
setup.bat
```

### Option 2: Manual Setup

1. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate Virtual Environment:**
   
   **Windows:**
   ```cmd
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Hand Detection and Data Collection

Run the hand detection script to collect training data:

```bash
python detection/hand_detection.py
```

**Controls:**
- Press any letter key (a-z) to set the current label
- Press 'S' to save a single sample
- Press 'R' to toggle auto-recording mode
- Press 'ESC' to quit

### Jupyter Notebook

Start Jupyter notebook for model development:

```bash
jupyter notebook
```

Open `signLanguage.ipynb` to work with the neural network model.

## Project Structure

```
signLanguage/
├── detection/
│   └── hand_detection.py      # Real-time hand tracking and data collection
├── data/
│   └── asl_letters.csv       # Collected training data (created automatically)
├── signLanguage.ipynb         # Jupyter notebook for model training
├── requirements.txt           # Python dependencies
├── setup.bat                 # Windows batch setup script
├── setup.ps1                 # PowerShell setup script
└── README.md                 # This file
```

## Dependencies

- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Hand detection and landmark extraction
- **TensorFlow**: Machine learning framework
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **Jupyter**: Interactive development environment

## Technical Notes

### Hand Tracking Approach
- Uses MediaPipe for robust hand detection
- Extracts 21 hand landmarks per hand
- Normalizes coordinates relative to hand bounding box
- Creates 42-feature vectors (x,y coordinates for 21 landmarks)

### Neural Network Architecture
- Convolutional Neural Network (CNN) for image classification
- Multiple Conv2D layers with ReLU activation
- MaxPooling for dimensionality reduction
- Dense layers for final classification
- Output: 25 classes (A-Z, excluding J and Z which require motion)

### Performance Considerations
- CNN requires significant computational resources
- GPU acceleration recommended for training
- Image size reduced to 50x50 pixels for efficiency
- Consider using transfer learning for better performance

## Troubleshooting

### Common Issues

1. **Camera not detected**: Ensure your webcam is connected and not used by other applications
2. **Import errors**: Make sure virtual environment is activated and all dependencies are installed
3. **Performance issues**: Reduce image resolution or use GPU acceleration for training

### Environment Recreation

If you need to recreate the environment:

```bash
# Remove old environment
rmdir /s venv  # Windows
rm -rf venv    # macOS/Linux

# Run setup script again
.\setup.ps1    # Windows PowerShell
.\setup.bat    # Windows CMD
```

## Contributing

Feel free to contribute improvements, bug fixes, or additional features to this project.

# ASL Recognition System

🤖 **Mission**: Advancing accessibility by bridging communication barriers for the Deaf and hard-of-hearing through inclusive, real-time AI translation tools.

A real-time American Sign Language (ASL) recognition system that translates sign language to speech using computer vision, machine learning, and large language models. This agentic AI prototype integrates gesture recognition with LLMs (LangChain + LLaMA 3) and gTTS to enable autonomous sign-to-text and sign-to-speech translation.

## ✨ Features

- **Real-time Hand Tracking**: Uses MediaPipe for robust hand detection and landmark extraction
- **ASL Letter Recognition**: Trained neural network for accurate letter classification
- **LLM Translation**: Uses LLaMA 3 via Ollama for natural English translation
- **Text-to-Speech**: Converts translations to spoken audio
- **Data Collection**: Tools for collecting and training on custom ASL data
- **Agentic AI**: Autonomous translation pipeline with intelligent sentence construction

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd signLanguage

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Ollama (for LLM translation)
# Download from: https://ollama.com/download
ollama serve
ollama pull llama3
```

### 2. Run the System

```bash
python main.py
```

**Controls:**
- **SPACE**: Start/stop recording letters
- **ENTER**: Send letters to LLM for translation
- **ESC**: Quit

**Usage Tips:**
- Hold each letter for 2-3 seconds for best detection
- Press ENTER when you've finished signing to get translation
- Ensure good lighting and clear hand visibility

## 🎯 Training Your Own Model

### 1. Collect Training Data

```bash
python scripts/run_data_collection.py
```

**Data Collection Controls:**
- **Letter keys (A-Z)**: Set current label
- **'S'**: Save a single sample
- **'R'**: Toggle auto-recording
- **'ESC'**: Quit and save

**Data Collection Tips:**
- Collect 50-100 samples per letter for best results
- Vary hand positions, lighting, and angles
- Use consistent signing style throughout

### 2. Train the Model

```bash
python scripts/train_model.py
```

This will train a neural network on your collected data and save the model to `data/models/`.

### 3. Alternative Entry Points

```bash
# Alternative main entry point
python scripts/run_asl_agent.py

# Windows setup script
scripts/setup.bat

# PowerShell setup script
scripts/setup.ps1
```

## Project Structure

```
signLanguage/
├── src/                          # Core system modules
│   ├── realtime_predictor.py     # Main ASL recognition system
│   ├── llm_translator.py         # LLM translation with Ollama
│   └── tts_engine.py            # Text-to-speech engine
├── scripts/                      # Utility scripts
│   ├── run_data_collection.py    # Data collection entry point
│   ├── collect_data.py          # Data collection implementation
│   ├── train_model.py           # Model training script
│   ├── run_asl_agent.py         # Alternative entry point
│   ├── setup.bat               # Windows setup script
│   └── setup.ps1               # PowerShell setup script
├── data/                        # Data storage
│   ├── models/                  # Trained models
│   ├── raw/                     # Training data
│   └── tts_cache/              # TTS audio cache
├── docs/                        # Documentation
├── main.py                      # Main entry point
└── requirements.txt             # Dependencies
```

## 🔧 Technical Details

### Hand Tracking
- **MediaPipe**: 21-point hand landmark detection
- **Normalization**: Coordinates relative to hand bounding box
- **Features**: 42 features (x,y coordinates for 21 landmarks)
- **Debouncing**: Prevents duplicate letter detection during transitions

### Machine Learning
- **Algorithm**: MLPClassifier (Multi-layer Perceptron) with scikit-learn
- **Architecture**: Hidden layers (100, 50) neurons
- **Activation**: ReLU activation, Adam optimizer
- **Training**: Normalized hand landmark features
- **Confidence**: 95% threshold for letter detection

### LLM Integration
- **Model**: Ollama with LLaMA 3 for translation
- **Function**: Converts letter sequences to natural English
- **Features**: Handles spelling errors and word spacing
- **Fallback**: Rule-based translation available
- **Examples**: "HIMYNAMEISBEN" → "Hi my name is Ben"

### Text-to-Speech
- **Primary**: Google Text-to-Speech (gTTS)
- **Fallback**: pyttsx3 (offline TTS)
- **Performance**: Audio caching for speed
- **Playback**: Async playback support

## 📋 Requirements

- **Python**: 3.8+
- **Webcam**: USB or built-in camera
- **Ollama**: For LLM features (download from https://ollama.com)
- **OS**: Windows/macOS/Linux
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space

## 📦 Dependencies

See `requirements.txt` for full list. Key dependencies:

### Core Dependencies
- **opencv-python**: Computer vision and camera handling
- **mediapipe**: Hand tracking and landmark detection
- **scikit-learn**: Machine learning model training
- **numpy**: Numerical computations
- **pandas**: Data handling

### LLM & TTS Dependencies
- **langchain**: LLM framework
- **langchain-ollama**: Ollama integration
- **gtts**: Google Text-to-Speech
- **pyttsx3**: Offline text-to-speech
- **pygame**: Audio playback

## 🛠️ Troubleshooting

### Common Issues

1. **Camera not detected**: 
   - Ensure webcam is connected and not used by other apps
   - Check camera permissions in system settings

2. **Model not found**: 
   - Run data collection and training first
   - Check that `data/models/asl_model.pkl` exists

3. **Ollama connection error**: 
   - Make sure `ollama serve` is running
   - Verify Ollama is installed and LLaMA 3 model is pulled

4. **Import errors**: 
   - Activate virtual environment: `venv\Scripts\activate`
   - Install dependencies: `pip install -r requirements.txt`

5. **TensorFlow DLL errors**: 
   - This is a known Windows issue with MediaPipe
   - The system uses scikit-learn instead of TensorFlow

### Performance Tips

- **Lighting**: Ensure good, even lighting for hand detection
- **Positioning**: Hold letters steady for 2-3 seconds
- **Consistency**: Use consistent hand positioning
- **Training**: Collect diverse hand positions and lighting conditions
- **Debouncing**: System prevents duplicate letters during transitions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe team** for hand tracking technology
- **Ollama team** for local LLM deployment
- **scikit-learn community** for machine learning tools
- **ASL community** for inspiration and guidance
- **LangChain team** for LLM integration framework

## 🌟 Mission Impact

This project aims to:
- **Bridge communication barriers** for the Deaf and hard-of-hearing community
- **Advance accessibility** through inclusive AI technology
- **Enable real-time translation** from sign language to speech
- **Promote understanding** and inclusion in digital spaces

---

**Made with ❤️ for accessibility and inclusion**
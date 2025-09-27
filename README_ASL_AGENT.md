# ASL Agent - Sign-to-Speech Translation System

## Mission Statement
**Advancing accessibility by bridging communication barriers for the Deaf and hard-of-hearing through inclusive, real-time AI translation tools that leverage LLMs to convert sign language into fluent, natural English for broader understanding.**

## Overview
The ASL Agent is an agentic AI prototype that integrates gesture recognition with Large Language Models (LangChain + LLaMA 3/Phi-3) and Google Text-to-Speech (gTTS), enabling autonomous sign-to-text and sign-to-speech translation.

## Features
- **Real-time ASL Detection**: Uses MediaPipe and trained ML models to detect ASL letters
- **Intelligent Sentence Building**: Combines detected letters into words and sentences
- **LLM-Powered Translation**: Converts ASL sequences to natural English using LangChain
- **Text-to-Speech Output**: Provides natural speech output using gTTS
- **Agentic Architecture**: Autonomous operation with intelligent decision-making
- **Accessibility Focus**: Designed specifically for Deaf and hard-of-hearing communication

## System Architecture

```
ASL Agent Pipeline:
Camera → Hand Detection → Letter Recognition → Sentence Building → LLM Translation → TTS → Speech Output
```

### Components
1. **RealtimeASLPredictor**: Hand tracking and letter detection
2. **SentenceBuilder**: Intelligent word and sentence construction
3. **ASLTranslator**: LLM-powered translation to natural English
4. **TTSEngine**: Text-to-speech conversion
5. **ASLAgent**: Main orchestrator coordinating all components

## Installation

### Prerequisites
- Python 3.8+
- Camera/webcam
- Trained ASL model (run training first)

### Setup
1. **Clone and navigate to the project**:
   ```bash
   cd signLanguage
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the ASL model** (if not already done):
   ```bash
   python detection/simple_model.py
   ```

4. **Test the system**:
   ```bash
   python test_asl_agent.py
   ```

## Usage

### Quick Start
```bash
python run_asl_agent.py
```

### Controls
- **SPACE**: Start/Stop recording ASL gestures
- **ESC**: Quit the application

### Real-time Operation
1. Start the agent
2. Position your hand in front of the camera
3. Sign ASL letters clearly
4. The system will:
   - Detect individual letters
   - Build words and sentences
   - Translate to natural English
   - Speak the translation aloud

## Configuration

The system can be configured through the `ASLAgent` constructor:

```python
config = {
    'detection': {
        'confidence_threshold': 0.7,
        'smoothing_frames': 5,
        'camera_index': 0
    },
    'sentence_building': {
        'word_timeout': 2.0,
        'sentence_timeout': 5.0,
        'min_word_length': 2
    },
    'translation': {
        'model_type': 'ollama',
        'model_name': 'llama3'
    },
    'tts': {
        'engine_type': 'gtts',
        'language': 'en',
        'slow': False
    }
}

agent = ASLAgent(config=config)
```

## LLM Integration

### Supported Models
- **Ollama**: Local LLaMA 3, Phi-3, and other models
- **OpenAI**: GPT models (requires API key)
- **Local Transformers**: Hugging Face models
- **Fallback**: Rule-based translation

### Setup Ollama (Recommended)
1. Install Ollama: https://ollama.ai/
2. Pull a model:
   ```bash
   ollama pull llama3
   # or
   ollama pull phi3
   ```
3. The system will automatically use Ollama if available

## Text-to-Speech Options

### Supported Engines
- **gTTS**: Google Text-to-Speech (default)
- **pyttsx3**: Cross-platform TTS
- **System TTS**: Windows SAPI, Linux espeak
- **Fallback**: Console output

### Audio Quality
- High-quality speech synthesis
- Configurable speech rate and volume
- Async playback for smooth operation
- Audio caching for performance

## Testing

### Component Tests
```bash
python test_asl_agent.py
```

### Individual Component Testing
```python
# Test sentence builder
python detection/sentence_builder.py

# Test LLM translator
python detection/llm_translator.py

# Test TTS engine
python detection/tts_engine.py
```

## Performance Optimization

### Real-time Performance
- Optimized hand tracking with MediaPipe
- Efficient feature extraction
- Smoothing and confidence filtering
- Async TTS playback

### Accuracy Improvements
- Confidence-based letter filtering
- Intelligent word completion
- Context-aware translation
- Grammar rule application

## Troubleshooting

### Common Issues
1. **Camera not detected**: Check camera connection and permissions
2. **Model not found**: Train the model first with `python detection/simple_model.py`
3. **TTS not working**: Install audio dependencies (`pip install pygame pydub`)
4. **LLM errors**: Check Ollama installation or use fallback mode

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Accessibility Features

### Designed for Accessibility
- **Real-time feedback**: Visual and audio confirmation
- **Configurable thresholds**: Adjustable sensitivity
- **Error handling**: Graceful degradation
- **Multiple TTS options**: Choose preferred voice
- **Session statistics**: Track usage and performance

### Inclusive Design
- Works with various lighting conditions
- Handles different signing styles
- Provides multiple translation options
- Supports different communication preferences

## Future Enhancements

### Planned Features
- **Multi-language support**: Spanish, French, etc.
- **Advanced gestures**: Numbers, punctuation, emotions
- **Context memory**: Conversation history
- **Custom vocabulary**: User-specific terms
- **Mobile support**: iOS/Android apps
- **Cloud integration**: Remote processing

### Research Directions
- **Improved accuracy**: Better models and training
- **Faster processing**: Edge optimization
- **Natural language**: More fluent translations
- **Emotion recognition**: Sentiment analysis
- **Multi-modal input**: Voice + sign combination

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Code Style
- Follow PEP 8
- Add docstrings
- Include type hints
- Write tests for new features

## License
This project is designed for accessibility and educational purposes. Please ensure compliance with local regulations and respect for the Deaf community.

## Acknowledgments
- **MediaPipe**: Hand tracking and pose estimation
- **LangChain**: LLM integration framework
- **Google TTS**: Speech synthesis
- **Deaf Community**: Inspiration and feedback
- **Accessibility Advocates**: Guidance and support

## Support
For issues, questions, or contributions:
1. Check the troubleshooting section
2. Run the test suite
3. Review the documentation
4. Submit an issue with details

---

**Mission Accomplished**: Bridging communication barriers through inclusive AI technology! 🌟

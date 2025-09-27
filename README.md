Sign Language Recognition System with AI Translation

Technologies: Python, OpenCV, MediaPipe, scikit-learn, LangChain, Ollama, LLaMA 3, gTTS, real-time computer vision

Description:
Built a real-time ASL recognition system that translates sign language to speech. Uses MediaPipe for hand tracking, scikit-learn for letter classification, and LLaMA 3 via Ollama for translation. Includes a TTS pipeline for spoken output.

Key Features:
Real-time hand tracking and ASL letter detection
ML model trained on hand landmark features
LLM-based translation with LLaMA 3
Text-to-speech output
Debounced detection to reduce false positives
Modular design with error handling

Technical Achievements:
Implemented 3-second debouncing to prevent duplicate letters
Integrated LangChain with Ollama for translation
Added fallback translation for offline use
Optimized performance with warning suppression and dependency cleanup
Achieved 95% confidence threshold for reliable detection

Impact:
Enables real-time communication for deaf and hard-of-hearing users by converting ASL to natural English speech.

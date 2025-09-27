#!/usr/bin/env python3
"""
ASL Agent Runner
Main entry point for the ASL Sign-to-Speech Translation System
Uses camera-based ASL detection with trained model
"""

import sys
import os
from pathlib import Path

# Add the detection module to the path
sys.path.append(str(Path(__file__).parent / "detection"))

def main():
    """Main entry point for the ASL Agent"""
    print("🤖 ASL Agent - Sign to Speech Translation System")
    print("=" * 60)
    print("Mission: Advancing accessibility by bridging communication barriers")
    print("for the Deaf and hard-of-hearing through inclusive, real-time AI translation")
    print("=" * 60)
    
    try:
        # Import the enhanced realtime predictor
        from detection.realtime_predictor import RealtimeASLPredictor
        
        # Create the enhanced predictor
        predictor = RealtimeASLPredictor()
        
        print("\n🎥 Starting real-time ASL detection...")
        print("Controls:")
        print("  - SPACE: Start/Stop recording")
        print("  - ESC: Quit")
        print("  - Make sure your camera is connected and working")
        print("\nPress any key to continue...")
        input()
        
        # Run the enhanced predictor
        predictor.run()
        
    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that your model is trained and camera is working")
    finally:
        print("\n👋 Thank you for using ASL Agent!")
        print("Mission accomplished: Communication barriers bridged! 🌟")

if __name__ == "__main__":
    main()
"""
ASL Recognition System - Main Entry Point
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    print("🤖 ASL Recognition System")
    print("=" * 40)
    print("Mission: Advancing accessibility by bridging communication barriers")
    print("for the Deaf and hard-of-hearing through inclusive, real-time AI translation")
    print("=" * 40)

    try:
        from realtime_predictor import RealtimeASLPredictor
        predictor = RealtimeASLPredictor()

        print("\n🎥 Starting real-time ASL detection...")
        print("Controls:")
        print("  - SPACE: Start/Stop recording")
        print("  - ENTER: Send letters to LLM for translation")
        print("  - ESC: Quit")
        print("  - Make sure your camera is connected and working")
        print("\nPress any key to continue...")
        input()

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
        print("\n👋 Thank you for using ASL Recognition System!")
        print("Mission accomplished: Communication barriers bridged! 🌟")

if __name__ == "__main__":
    main()

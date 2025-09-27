"""
ASL Data Collection Entry Point
Run this script to collect training data for ASL letter recognition
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    print("🎯 ASL Data Collection Tool")
    print("=" * 40)
    print("This tool will help you collect training data for ASL letter recognition.")
    print("Make sure your camera is connected and working.")
    print()
    
    try:
        from collect_data import ASLDataCollector
        
        collector = ASLDataCollector()
        collector.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that your camera is working and try again.")

if __name__ == "__main__":
    main()

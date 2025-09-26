#!/usr/bin/env python3
"""
ASL Sign Language Dataset Collection System
Part 2: Enhanced Gesture Dataset Collection

This script provides a comprehensive interface for collecting, managing,
and preparing ASL gesture datasets for machine learning.
"""

import sys
import os
from pathlib import Path

# Add detection directory to path
sys.path.append(str(Path(__file__).parent / "detection"))

def main():
    print("🤟 ASL Sign Language Dataset Collection System")
    print("=" * 60)
    print("Part 2: Enhanced Gesture Dataset Collection")
    print("=" * 60)
    
    while True:
        print("\n📋 MAIN MENU:")
        print("1. 🎥 Enhanced Data Collection (Recommended)")
        print("2. 📊 Dataset Analysis & Statistics")
        print("3. 🔍 Data Quality Validation")
        print("4. 🤖 ML Dataset Preparation")
        print("5. 📈 View Dataset Statistics")
        print("6. 🧹 Clean Dataset")
        print("7. 📤 Export Dataset")
        print("8. ❓ Help & Documentation")
        print("9. 🚪 Exit")
        print("\n💡 Tip: In data collection, use A-Z keys to set labels!")
        
        choice = input("\nSelect an option (1-9): ").strip()
        
        if choice == "1":
            run_enhanced_collection()
        elif choice == "2":
            run_dataset_analysis()
        elif choice == "3":
            run_data_validation()
        elif choice == "4":
            run_ml_preparation()
        elif choice == "5":
            run_quick_stats()
        elif choice == "6":
            run_data_cleaning()
        elif choice == "7":
            run_data_export()
        elif choice == "8":
            show_help()
        elif choice == "9":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please try again.")

def run_enhanced_collection():
    """Run the enhanced data collection system"""
    print("\n🎥 Starting Enhanced Data Collection...")
    print("=" * 50)
    
    try:
        from enhanced_data_collector import EnhancedDataCollector
        collector = EnhancedDataCollector()
        collector.run()
    except ImportError as e:
        print(f"❌ Error importing enhanced data collector: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"❌ Error running data collection: {e}")

def run_dataset_analysis():
    """Run comprehensive dataset analysis"""
    print("\n📊 Running Dataset Analysis...")
    print("=" * 50)
    
    try:
        from dataset_manager import DatasetAnalyzer
        analyzer = DatasetAnalyzer()
        analyzer.analyze_dataset()
    except ImportError as e:
        print(f"❌ Error importing dataset analyzer: {e}")
    except Exception as e:
        print(f"❌ Error running analysis: {e}")

def run_data_validation():
    """Run data quality validation"""
    print("\n🔍 Running Data Quality Validation...")
    print("=" * 50)
    
    try:
        from data_validator import DataValidator
        validator = DataValidator()
        df = validator.load_data()
        if df is not None:
            validator.generate_quality_report(df)
    except ImportError as e:
        print(f"❌ Error importing data validator: {e}")
    except Exception as e:
        print(f"❌ Error running validation: {e}")

def run_ml_preparation():
    """Run ML dataset preparation"""
    print("\n🤖 Preparing Dataset for Machine Learning...")
    print("=" * 50)
    
    # Get user preferences
    print("\nConfiguration options:")
    test_size = float(input("Test set size (0.1-0.4, default 0.2): ") or "0.2")
    feature_selection = int(input("Number of features to select (default 20): ") or "20")
    balance = input("Create balanced dataset? (y/n, default n): ").lower() == 'y'
    
    try:
        from ml_preparer import MLPreparer
        preparer = MLPreparer()
        result = preparer.prepare_full_pipeline(
            test_size=test_size,
            feature_selection_k=feature_selection,
            balance_dataset=balance
        )
        
        if result:
            print("\n✅ ML preparation completed successfully!")
            print(f"   Data saved to: data/processed/")
            print(f"   TensorFlow format: data/processed/tensorflow/")
            print(f"   PyTorch format: data/processed/pytorch/")
    except ImportError as e:
        print(f"❌ Error importing ML preparer: {e}")
    except Exception as e:
        print(f"❌ Error running ML preparation: {e}")

def run_quick_stats():
    """Show quick dataset statistics"""
    print("\n📈 Dataset Statistics...")
    print("=" * 50)
    
    try:
        from dataset_manager import DatasetAnalyzer
        analyzer = DatasetAnalyzer()
        df = analyzer.load_data()
        
        if df is not None:
            print(f"Total samples: {len(df)}")
            print(f"Classes: {df['label'].nunique()}")
            print(f"Features: {len([col for col in df.columns if col.startswith('k')])}")
            
            class_counts = df['label'].value_counts()
            print(f"\nClass distribution:")
            for label, count in class_counts.head(10).items():
                print(f"  {label}: {count} samples")
            
            if len(class_counts) > 10:
                print(f"  ... and {len(class_counts) - 10} more classes")
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")

def run_data_cleaning():
    """Run data cleaning"""
    print("\n🧹 Cleaning Dataset...")
    print("=" * 50)
    
    try:
        from data_validator import DataValidator
        validator = DataValidator()
        df = validator.load_data()
        
        if df is not None:
            cleaned_df = validator.clean_dataset(df)
            print(f"✅ Dataset cleaned and saved!")
    except Exception as e:
        print(f"❌ Error cleaning dataset: {e}")

def run_data_export():
    """Export dataset in various formats"""
    print("\n📤 Exporting Dataset...")
    print("=" * 50)
    
    print("Export formats available:")
    print("1. CSV format")
    print("2. JSON format")
    print("3. NumPy arrays")
    print("4. All formats")
    
    choice = input("Select format (1-4): ").strip()
    
    try:
        from dataset_manager import DatasetAnalyzer
        analyzer = DatasetAnalyzer()
        
        if choice == "1":
            analyzer.export_for_training('csv')
        elif choice == "2":
            analyzer.export_for_training('json')
        elif choice == "3":
            analyzer.export_for_training('numpy')
        elif choice == "4":
            analyzer.export_for_training('csv')
            analyzer.export_for_training('json')
            analyzer.export_for_training('numpy')
        else:
            print("❌ Invalid choice")
    except Exception as e:
        print(f"❌ Error exporting dataset: {e}")

def show_help():
    """Show help and documentation"""
    print("\n❓ HELP & DOCUMENTATION")
    print("=" * 60)
    
    print("""
🎯 SYSTEM OVERVIEW:
This is Part 2 of the ASL Sign Language Detection System, focusing on
enhanced gesture dataset collection and management.

📋 MAIN COMPONENTS:

1. 🎥 Enhanced Data Collection:
   - Real-time hand tracking with MediaPipe
   - Interactive UI with progress tracking
   - Session management and metadata
   - Quality validation during collection

2. 📊 Dataset Analysis:
   - Comprehensive statistical analysis
   - Class distribution visualization
   - Feature quality assessment
   - Temporal analysis of data collection

3. 🔍 Data Validation:
   - Data integrity checks
   - Outlier detection
   - Class balance analysis
   - Quality scoring system

4. 🤖 ML Preparation:
   - Train/test splitting
   - Feature preprocessing
   - Feature selection
   - Export for TensorFlow/PyTorch

📁 DATA STRUCTURE:
data/
├── raw/                    # Raw collected data
│   ├── asl_letters.csv     # Main dataset
│   └── session_*.json       # Session metadata
├── processed/              # Processed ML data
│   ├── tensorflow/         # TensorFlow format
│   ├── pytorch/           # PyTorch format
│   └── *.npy              # NumPy arrays
└── metadata/              # Analysis reports
    ├── dataset_metadata.json
    └── quality_report.json

🎮 CONTROLS (Data Collection):
- A-Z: Set current label
- S: Save single sample
- R: Toggle auto-recording
- H: Toggle help display
- ESC: Quit

💡 TIPS:
- Collect at least 100 samples per class for good ML performance
- Use consistent lighting and hand positioning
- Record samples from different angles and distances
- Validate data quality before ML training
- Use balanced datasets for better model performance
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your installation and try again.")

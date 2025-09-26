import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Ensure directories exist
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.raw_data_dir / "asl_letters.csv"
        self.metadata_path = self.metadata_dir / "dataset_metadata.json"
        
    def load_data(self):
        """Load the raw dataset from CSV"""
        if not self.csv_path.exists():
            print("No dataset found. Run the data collector first.")
            return None
        
        df = pd.read_csv(self.csv_path)
        print(f"Loaded dataset with {len(df)} samples")
        return df
    
    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        df = self.load_data()
        if df is None:
            return
        
        print("\n" + "="*60)
        print("DATASET ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS:")
        print(f"Total samples: {len(df)}")
        print(f"Features per sample: {len([col for col in df.columns if col.startswith('k')])}")
        print(f"Unique labels: {df['label'].nunique()}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Class distribution
        print(f"\n📈 CLASS DISTRIBUTION:")
        class_counts = df['label'].value_counts().sort_index()
        print(class_counts)
        
        # Visualizations
        self.plot_class_distribution(class_counts)
        self.plot_temporal_distribution(df)
        self.plot_feature_statistics(df)
        
        # Data quality checks
        self.check_data_quality(df)
        
        # Recommendations
        self.generate_recommendations(df, class_counts)
    
    def plot_class_distribution(self, class_counts):
        """Plot class distribution"""
        plt.figure(figsize=(15, 6))
        
        # Bar plot
        plt.subplot(1, 2, 1)
        class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Samples per Class')
        plt.xlabel('Letter')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Pie chart for top classes
        plt.subplot(1, 2, 2)
        top_classes = class_counts.head(10)
        plt.pie(top_classes.values, labels=top_classes.index, autopct='%1.1f%%')
        plt.title('Top 10 Classes Distribution')
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_distribution(self, df):
        """Plot temporal distribution of data collection"""
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_counts = df.groupby('date').size()
        
        plt.figure(figsize=(12, 6))
        daily_counts.plot(kind='line', marker='o', color='green')
        plt.title('Data Collection Over Time')
        plt.xlabel('Date')
        plt.ylabel('Samples Collected')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.data_dir / 'temporal_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_statistics(self, df):
        """Plot feature statistics"""
        feature_cols = [col for col in df.columns if col.startswith('k')]
        
        plt.figure(figsize=(15, 10))
        
        # Feature means
        plt.subplot(2, 2, 1)
        feature_means = df[feature_cols].mean()
        plt.plot(feature_means.values)
        plt.title('Feature Means')
        plt.xlabel('Feature Index')
        plt.ylabel('Mean Value')
        
        # Feature standard deviations
        plt.subplot(2, 2, 2)
        feature_stds = df[feature_cols].std()
        plt.plot(feature_stds.values)
        plt.title('Feature Standard Deviations')
        plt.xlabel('Feature Index')
        plt.ylabel('Std Dev')
        
        # Correlation heatmap (sample)
        plt.subplot(2, 2, 3)
        sample_features = df[feature_cols[:10]]  # First 10 features
        corr_matrix = sample_features.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation (First 10 Features)')
        
        # Feature distribution
        plt.subplot(2, 2, 4)
        df[feature_cols].boxplot()
        plt.title('Feature Value Distributions')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'feature_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def check_data_quality(self, df):
        """Check data quality issues"""
        print(f"\n🔍 DATA QUALITY CHECKS:")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"⚠️  Missing values found:")
            print(missing[missing > 0])
        else:
            print("✅ No missing values")
        
        # Duplicate samples
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️  {duplicates} duplicate samples found")
        else:
            print("✅ No duplicate samples")
        
        # Feature range checks
        feature_cols = [col for col in df.columns if col.startswith('k')]
        out_of_range = 0
        for col in feature_cols:
            if df[col].min() < 0 or df[col].max() > 1:
                out_of_range += 1
        
        if out_of_range > 0:
            print(f"⚠️  {out_of_range} features have values outside [0,1] range")
        else:
            print("✅ All features are properly normalized")
        
        # Class imbalance
        class_counts = df['label'].value_counts()
        min_count = class_counts.min()
        max_count = class_counts.max()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 3:
            print(f"⚠️  Significant class imbalance detected (ratio: {imbalance_ratio:.2f})")
        else:
            print("✅ Classes are reasonably balanced")
    
    def generate_recommendations(self, df, class_counts):
        """Generate recommendations for dataset improvement"""
        print(f"\n💡 RECOMMENDATIONS:")
        
        # Target samples per class
        target_samples = 100
        min_samples = class_counts.min()
        max_samples = class_counts.max()
        
        if min_samples < target_samples:
            under_collected = class_counts[class_counts < target_samples]
            print(f"📝 Collect more samples for: {list(under_collected.index)}")
            print(f"   Target: {target_samples} samples per class")
        
        # Class balance
        if max_samples / min_samples > 2:
            print("⚖️  Consider balancing classes by collecting more samples for underrepresented classes")
        
        # Data augmentation
        if len(df) < 1000:
            print("🔄 Consider data augmentation techniques for small datasets")
        
        # Feature engineering
        print("🔧 Consider feature engineering:")
        print("   - Add relative distances between landmarks")
        print("   - Include hand orientation features")
        print("   - Add temporal features for gesture sequences")
    
    def prepare_ml_dataset(self, test_size=0.2, random_state=42):
        """Prepare dataset for machine learning"""
        df = self.load_data()
        if df is None:
            return None
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col.startswith('k')]
        X = df[feature_cols].values
        y = df['label'].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save processed data
        processed_data = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'scaler': scaler
        }
        
        # Save to files
        np.save(self.processed_data_dir / 'X_train.npy', X_train_scaled)
        np.save(self.processed_data_dir / 'X_test.npy', X_test_scaled)
        np.save(self.processed_data_dir / 'y_train.npy', y_train)
        np.save(self.processed_data_dir / 'y_test.npy', y_test)
        
        # Save scaler
        import joblib
        joblib.dump(scaler, self.processed_data_dir / 'scaler.pkl')
        
        # Save metadata
        ml_metadata = {
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'n_classes': len(np.unique(y)),
            'test_size': test_size,
            'random_state': random_state,
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.processed_data_dir / 'ml_metadata.json', 'w') as f:
            json.dump(ml_metadata, f, indent=2)
        
        print(f"\n✅ ML dataset prepared:")
        print(f"   Training samples: {len(X_train_scaled)}")
        print(f"   Test samples: {len(X_test_scaled)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Classes: {len(np.unique(y))}")
        print(f"   Saved to: {self.processed_data_dir}")
        
        return processed_data
    
    def export_for_training(self, format='csv'):
        """Export dataset in various formats for training"""
        df = self.load_data()
        if df is None:
            return
        
        if format == 'csv':
            # Export as CSV
            export_path = self.processed_data_dir / 'training_dataset.csv'
            df.to_csv(export_path, index=False)
            print(f"✅ Dataset exported to: {export_path}")
        
        elif format == 'json':
            # Export as JSON
            export_path = self.processed_data_dir / 'training_dataset.json'
            df.to_json(export_path, orient='records', indent=2)
            print(f"✅ Dataset exported to: {export_path}")
        
        elif format == 'numpy':
            # Export as NumPy arrays
            feature_cols = [col for col in df.columns if col.startswith('k')]
            X = df[feature_cols].values
            y = df['label'].values
            
            np.save(self.processed_data_dir / 'features.npy', X)
            np.save(self.processed_data_dir / 'labels.npy', y)
            print(f"✅ NumPy arrays exported to: {self.processed_data_dir}")

# -----------------------------
# Command Line Interface
# -----------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Dataset Manager')
    parser.add_argument('--analyze', action='store_true', help='Analyze the dataset')
    parser.add_argument('--prepare-ml', action='store_true', help='Prepare dataset for ML training')
    parser.add_argument('--export', choices=['csv', 'json', 'numpy'], help='Export dataset in specified format')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer(args.data_dir)
    
    if args.analyze:
        analyzer.analyze_dataset()
    elif args.prepare_ml:
        analyzer.prepare_ml_dataset()
    elif args.export:
        analyzer.export_for_training(args.export)
    else:
        print("Use --help to see available options")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class DataValidator:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.csv_path = self.raw_data_dir / "asl_letters.csv"
        
    def load_data(self):
        """Load the dataset"""
        if not self.csv_path.exists():
            print("No dataset found. Run the data collector first.")
            return None
        
        df = pd.read_csv(self.csv_path)
        print(f"Loaded dataset with {len(df)} samples")
        return df
    
    def validate_data_integrity(self, df):
        """Validate data integrity and quality"""
        print("\n🔍 DATA INTEGRITY VALIDATION")
        print("=" * 50)
        
        issues = []
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            issues.append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
        
        # Check for duplicate samples
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate samples found: {duplicates}")
        
        # Check feature ranges
        feature_cols = [col for col in df.columns if col.startswith('k')]
        out_of_range_features = []
        for col in feature_cols:
            if df[col].min() < 0 or df[col].max() > 1:
                out_of_range_features.append(col)
        
        if out_of_range_features:
            issues.append(f"Features out of [0,1] range: {out_of_range_features}")
        
        # Check for infinite values
        inf_values = np.isinf(df[feature_cols]).sum().sum()
        if inf_values > 0:
            issues.append(f"Infinite values found: {inf_values}")
        
        # Check for NaN values in features
        nan_features = df[feature_cols].isna().sum()
        if nan_features.sum() > 0:
            issues.append(f"NaN values in features: {nan_features[nan_features > 0].to_dict()}")
        
        # Report results
        if issues:
            print("❌ Data integrity issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Data integrity validation passed")
        
        return len(issues) == 0
    
    def validate_class_balance(self, df):
        """Validate class balance and distribution"""
        print("\n⚖️  CLASS BALANCE VALIDATION")
        print("=" * 50)
        
        class_counts = df['label'].value_counts()
        min_count = class_counts.min()
        max_count = class_counts.max()
        mean_count = class_counts.mean()
        std_count = class_counts.std()
        
        print(f"Class distribution statistics:")
        print(f"   Min samples: {min_count}")
        print(f"   Max samples: {max_count}")
        print(f"   Mean samples: {mean_count:.1f}")
        print(f"   Std deviation: {std_count:.1f}")
        print(f"   Imbalance ratio: {max_count/min_count:.2f}")
        
        # Identify problematic classes
        under_represented = class_counts[class_counts < mean_count - std_count]
        over_represented = class_counts[class_counts > mean_count + std_count]
        
        if len(under_represented) > 0:
            print(f"\n⚠️  Under-represented classes: {list(under_represented.index)}")
        
        if len(over_represented) > 0:
            print(f"⚠️  Over-represented classes: {list(over_represented.index)}")
        
        # Balance recommendations
        if max_count / min_count > 3:
            print("\n💡 Recommendations:")
            print("   - Collect more samples for under-represented classes")
            print("   - Consider data augmentation")
            print("   - Use stratified sampling for train/test split")
        
        return {
            'min_count': min_count,
            'max_count': max_count,
            'imbalance_ratio': max_count / min_count,
            'under_represented': list(under_represented.index),
            'over_represented': list(over_represented.index)
        }
    
    def validate_feature_quality(self, df):
        """Validate feature quality and distributions"""
        print("\n📊 FEATURE QUALITY VALIDATION")
        print("=" * 50)
        
        feature_cols = [col for col in df.columns if col.startswith('k')]
        X = df[feature_cols].values
        
        # Check feature variance
        feature_vars = np.var(X, axis=0)
        low_variance_features = np.where(feature_vars < 0.001)[0]
        
        if len(low_variance_features) > 0:
            print(f"⚠️  Low variance features detected: {len(low_variance_features)}")
            print(f"   Feature indices: {low_variance_features}")
        
        # Check for constant features
        constant_features = np.where(feature_vars == 0)[0]
        if len(constant_features) > 0:
            print(f"❌ Constant features found: {len(constant_features)}")
            print(f"   Feature indices: {constant_features}")
        
        # Check feature correlations
        corr_matrix = np.corrcoef(X.T)
        high_corr_pairs = []
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                if abs(corr_matrix[i, j]) > 0.95:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))
        
        if high_corr_pairs:
            print(f"⚠️  High correlation pairs found: {len(high_corr_pairs)}")
            print("   Consider removing redundant features")
        
        # Feature distribution analysis
        print(f"\nFeature distribution analysis:")
        print(f"   Mean: {np.mean(X):.4f}")
        print(f"   Std: {np.std(X):.4f}")
        print(f"   Min: {np.min(X):.4f}")
        print(f"   Max: {np.max(X):.4f}")
        
        return {
            'low_variance_features': low_variance_features,
            'constant_features': constant_features,
            'high_corr_pairs': high_corr_pairs,
            'feature_stats': {
                'mean': np.mean(X),
                'std': np.std(X),
                'min': np.min(X),
                'max': np.max(X)
            }
        }
    
    def detect_outliers(self, df, method='isolation_forest'):
        """Detect outliers in the dataset"""
        print(f"\n🎯 OUTLIER DETECTION ({method.upper()})")
        print("=" * 50)
        
        feature_cols = [col for col in df.columns if col.startswith('k')]
        X = df[feature_cols].values
        
        if method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            outliers = np.where(outlier_labels == -1)[0]
        
        elif method == 'z_score':
            # Z-score method
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            outliers = np.where(np.any(z_scores > 3, axis=1))[0]
        
        elif method == 'iqr':
            # IQR method
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = np.where(np.any((X < lower_bound) | (X > upper_bound), axis=1))[0]
        
        print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(X)*100:.1f}%)")
        
        if len(outliers) > 0:
            print("Outlier samples:")
            for idx in outliers[:10]:  # Show first 10
                print(f"   Sample {idx}: Label {df.iloc[idx]['label']}")
        
        return outliers
    
    def generate_quality_report(self, df):
        """Generate comprehensive quality report"""
        print("\n📋 COMPREHENSIVE QUALITY REPORT")
        print("=" * 60)
        
        # Basic statistics
        print(f"Dataset Overview:")
        print(f"   Total samples: {len(df)}")
        print(f"   Features: {len([col for col in df.columns if col.startswith('k')])}")
        print(f"   Classes: {df['label'].nunique()}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Run all validations
        integrity_ok = self.validate_data_integrity(df)
        balance_info = self.validate_class_balance(df)
        feature_info = self.validate_feature_quality(df)
        outliers = self.detect_outliers(df)
        
        # Overall quality score
        quality_score = 0
        if integrity_ok:
            quality_score += 25
        if balance_info['imbalance_ratio'] < 3:
            quality_score += 25
        if len(feature_info['constant_features']) == 0:
            quality_score += 25
        if len(outliers) / len(df) < 0.1:
            quality_score += 25
        
        print(f"\n🎯 OVERALL QUALITY SCORE: {quality_score}/100")
        
        if quality_score >= 80:
            print("✅ Dataset quality is excellent!")
        elif quality_score >= 60:
            print("⚠️  Dataset quality is good with minor issues")
        else:
            print("❌ Dataset quality needs improvement")
        
        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_score,
            'integrity_ok': integrity_ok,
            'balance_info': balance_info,
            'feature_info': feature_info,
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(df) * 100
        }
        
        report_path = self.data_dir / 'quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Quality report saved to: {report_path}")
        
        return report
    
    def clean_dataset(self, df, remove_outliers=True, remove_duplicates=True):
        """Clean the dataset based on validation results"""
        print("\n🧹 CLEANING DATASET")
        print("=" * 50)
        
        original_size = len(df)
        cleaned_df = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            before_dup = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            after_dup = len(cleaned_df)
            print(f"Removed {before_dup - after_dup} duplicate samples")
        
        # Remove outliers
        if remove_outliers:
            feature_cols = [col for col in cleaned_df.columns if col.startswith('k')]
            X = cleaned_df[feature_cols].values
            
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            outlier_mask = outlier_labels == -1
            
            before_outlier = len(cleaned_df)
            cleaned_df = cleaned_df[~outlier_mask]
            after_outlier = len(cleaned_df)
            print(f"Removed {before_outlier - after_outlier} outlier samples")
        
        # Save cleaned dataset
        cleaned_path = self.raw_data_dir / "asl_letters_cleaned.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        
        print(f"\n✅ Dataset cleaned:")
        print(f"   Original size: {original_size}")
        print(f"   Cleaned size: {len(cleaned_df)}")
        print(f"   Removed: {original_size - len(cleaned_df)} samples")
        print(f"   Saved to: {cleaned_path}")
        
        return cleaned_df

# -----------------------------
# Command Line Interface
# -----------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Data Validator')
    parser.add_argument('--validate', action='store_true', help='Validate dataset quality')
    parser.add_argument('--clean', action='store_true', help='Clean the dataset')
    parser.add_argument('--report', action='store_true', help='Generate quality report')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    args = parser.parse_args()
    
    validator = DataValidator(args.data_dir)
    df = validator.load_data()
    
    if df is None:
        return
    
    if args.validate:
        validator.validate_data_integrity(df)
        validator.validate_class_balance(df)
        validator.validate_feature_quality(df)
    
    elif args.clean:
        validator.clean_dataset(df)
    
    elif args.report:
        validator.generate_quality_report(df)
    
    else:
        print("Use --help to see available options")

if __name__ == "__main__":
    main()

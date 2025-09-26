import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLPreparer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        
        # Ensure directories exist
        for dir_path in [self.processed_data_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.raw_data_dir / "asl_letters.csv"
        
    def load_data(self):
        """Load the raw dataset"""
        if not self.csv_path.exists():
            print("No dataset found. Run the data collector first.")
            return None
        
        df = pd.read_csv(self.csv_path)
        print(f"Loaded dataset with {len(df)} samples")
        return df
    
    def prepare_features_and_labels(self, df):
        """Prepare features and labels for ML"""
        # Separate features and labels
        feature_cols = [col for col in df.columns if col.startswith('k')]
        X = df[feature_cols].values
        y = df['label'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Unique labels: {len(np.unique(y))}")
        
        return X, y, feature_cols
    
    def create_train_test_split(self, X, y, test_size=0.2, random_state=42, stratify=True):
        """Create train/test split with optional stratification"""
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def apply_preprocessing(self, X_train, X_test, y_train, y_test, method='standard'):
        """Apply preprocessing to the data"""
        print(f"\n🔧 Applying {method} preprocessing...")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard' or 'minmax'")
        
        # Fit on training data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        print(f"Preprocessing applied: {method}")
        print(f"Feature range after scaling: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, label_encoder
    
    def feature_selection(self, X_train, X_test, y_train, k=20):
        """Apply feature selection"""
        print(f"\n🎯 Applying feature selection (top {k} features)...")
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature indices
        selected_features = selector.get_support(indices=True)
        
        print(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")
        print(f"Selected feature indices: {selected_features}")
        
        return X_train_selected, X_test_selected, selector, selected_features
    
    def create_cross_validation_splits(self, X, y, n_splits=5, random_state=42):
        """Create cross-validation splits"""
        print(f"\n🔄 Creating {n_splits}-fold cross-validation splits...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_splits = list(skf.split(X, y))
        
        print(f"Created {len(cv_splits)} CV splits")
        
        return cv_splits
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, 
                          scaler, label_encoder, feature_names, selected_features=None):
        """Save processed data and preprocessors"""
        print(f"\n💾 Saving processed data...")
        
        # Save arrays
        np.save(self.processed_data_dir / 'X_train.npy', X_train)
        np.save(self.processed_data_dir / 'X_test.npy', X_test)
        np.save(self.processed_data_dir / 'y_train.npy', y_train)
        np.save(self.processed_data_dir / 'y_test.npy', y_test)
        
        # Save preprocessors
        joblib.dump(scaler, self.processed_data_dir / 'scaler.pkl')
        joblib.dump(label_encoder, self.processed_data_dir / 'label_encoder.pkl')
        
        # Save metadata
        metadata = {
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train)),
            'feature_names': feature_names,
            'selected_features': selected_features.tolist() if selected_features is not None else None,
            'preprocessing_method': 'standard',
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.processed_data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Processed data saved to: {self.processed_data_dir}")
        
        return metadata
    
    def export_for_tensorflow(self, X_train, X_test, y_train, y_test, output_dir=None):
        """Export data in TensorFlow-friendly format"""
        if output_dir is None:
            output_dir = self.processed_data_dir / "tensorflow"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🤖 Exporting for TensorFlow to: {output_dir}")
        
        # Save as NumPy arrays
        np.save(output_dir / 'train_features.npy', X_train)
        np.save(output_dir / 'test_features.npy', X_test)
        np.save(output_dir / 'train_labels.npy', y_train)
        np.save(output_dir / 'test_labels.npy', y_test)
        
        # Create TensorFlow dataset info
        tf_info = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_shape': X_train.shape[1:],
            'num_classes': len(np.unique(y_train)),
            'class_names': [f"class_{i}" for i in range(len(np.unique(y_train)))],
            'data_type': 'float32',
            'label_type': 'int32'
        }
        
        with open(output_dir / 'dataset_info.json', 'w') as f:
            json.dump(tf_info, f, indent=2)
        
        print(f"✅ TensorFlow data exported to: {output_dir}")
        
        return tf_info
    
    def export_for_pytorch(self, X_train, X_test, y_train, y_test, output_dir=None):
        """Export data in PyTorch-friendly format"""
        if output_dir is None:
            output_dir = self.processed_data_dir / "pytorch"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔥 Exporting for PyTorch to: {output_dir}")
        
        # Convert to PyTorch tensors format
        import torch
        
        # Save as PyTorch tensors
        torch.save(torch.FloatTensor(X_train), output_dir / 'train_features.pt')
        torch.save(torch.FloatTensor(X_test), output_dir / 'test_features.pt')
        torch.save(torch.LongTensor(y_train), output_dir / 'train_labels.pt')
        torch.save(torch.LongTensor(y_test), output_dir / 'test_labels.pt')
        
        # Create PyTorch dataset info
        pytorch_info = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_dim': X_train.shape[1],
            'num_classes': len(np.unique(y_train)),
            'class_names': [f"class_{i}" for i in range(len(np.unique(y_train)))],
            'data_type': 'float32',
            'label_type': 'long'
        }
        
        with open(output_dir / 'dataset_info.json', 'w') as f:
            json.dump(pytorch_info, f, indent=2)
        
        print(f"✅ PyTorch data exported to: {output_dir}")
        
        return pytorch_info
    
    def create_balanced_dataset(self, X, y, target_samples_per_class=100):
        """Create a balanced dataset by sampling"""
        print(f"\n⚖️  Creating balanced dataset (target: {target_samples_per_class} per class)...")
        
        unique_labels = np.unique(y)
        balanced_indices = []
        
        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            if len(label_indices) >= target_samples_per_class:
                # Randomly sample target_samples_per_class
                selected_indices = np.random.choice(label_indices, target_samples_per_class, replace=False)
            else:
                # Use all available samples
                selected_indices = label_indices
                print(f"   Warning: Class {label} has only {len(label_indices)} samples")
            
            balanced_indices.extend(selected_indices)
        
        balanced_indices = np.array(balanced_indices)
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        print(f"Balanced dataset: {len(X_balanced)} samples")
        print(f"Classes: {len(unique_labels)}")
        
        return X_balanced, y_balanced
    
    def prepare_full_pipeline(self, test_size=0.2, feature_selection_k=20, 
                            preprocessing='standard', balance_dataset=False):
        """Run the complete ML preparation pipeline"""
        print("🚀 STARTING ML PREPARATION PIPELINE")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Prepare features and labels
        X, y, feature_names = self.prepare_features_and_labels(df)
        
        # Create balanced dataset if requested
        if balance_dataset:
            X, y = self.create_balanced_dataset(X, y)
        
        # Create train/test split
        X_train, X_test, y_train, y_test = self.create_train_test_split(X, y, test_size=test_size)
        
        # Apply preprocessing
        X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, label_encoder = self.apply_preprocessing(
            X_train, X_test, y_train, y_test, method=preprocessing
        )
        
        # Apply feature selection
        X_train_selected, X_test_selected, selector, selected_features = self.feature_selection(
            X_train_scaled, X_test_scaled, y_train_encoded, k=feature_selection_k
        )
        
        # Save processed data
        metadata = self.save_processed_data(
            X_train_selected, X_test_selected, y_train_encoded, y_test_encoded,
            scaler, label_encoder, feature_names, selected_features
        )
        
        # Export for different frameworks
        tf_info = self.export_for_tensorflow(X_train_selected, X_test_selected, y_train_encoded, y_test_encoded)
        pytorch_info = self.export_for_pytorch(X_train_selected, X_test_selected, y_train_encoded, y_test_encoded)
        
        # Create cross-validation splits
        cv_splits = self.create_cross_validation_splits(X_train_selected, y_train_encoded)
        
        # Save CV splits
        np.save(self.processed_data_dir / 'cv_splits.npy', cv_splits)
        
        print(f"\n✅ ML PREPARATION COMPLETE!")
        print(f"   Training samples: {len(X_train_selected)}")
        print(f"   Test samples: {len(X_test_selected)}")
        print(f"   Features: {X_train_selected.shape[1]}")
        print(f"   Classes: {len(np.unique(y_train_encoded))}")
        print(f"   Data saved to: {self.processed_data_dir}")
        
        return {
            'X_train': X_train_selected,
            'X_test': X_test_selected,
            'y_train': y_train_encoded,
            'y_test': y_test_encoded,
            'metadata': metadata,
            'tf_info': tf_info,
            'pytorch_info': pytorch_info,
            'cv_splits': cv_splits
        }

# -----------------------------
# Command Line Interface
# -----------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL ML Data Preparer')
    parser.add_argument('--prepare', action='store_true', help='Prepare data for ML')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--feature-selection', type=int, default=20, help='Number of features to select (default: 20)')
    parser.add_argument('--preprocessing', choices=['standard', 'minmax'], default='standard', help='Preprocessing method')
    parser.add_argument('--balance', action='store_true', help='Create balanced dataset')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    
    args = parser.parse_args()
    
    preparer = MLPreparer(args.data_dir)
    
    if args.prepare:
        preparer.prepare_full_pipeline(
            test_size=args.test_size,
            feature_selection_k=args.feature_selection,
            preprocessing=args.preprocessing,
            balance_dataset=args.balance
        )
    else:
        print("Use --help to see available options")

if __name__ == "__main__":
    main()

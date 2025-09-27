"""
Simple ASL Model Training Script
Trains a scikit-learn MLPClassifier for ASL letter recognition
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """Load ASL training data"""
    data_path = Path("data/raw/asl_letters.csv")
    
    if not data_path.exists():
        print("❌ Training data not found!")
        print("Run data collection first: python run_data_collection.py")
        return None, None
    
    print("📊 Loading training data...")
    df = pd.read_csv(data_path)
    
    # Extract features and labels
    feature_columns = [col for col in df.columns if col not in ['label', 'timestamp']]
    X = df[feature_columns].values
    y = df['label'].values
    
    print(f"✅ Loaded {len(X)} samples with {len(feature_columns)} features")
    print(f"📝 Labels: {sorted(set(y))}")
    
    return X, y

def train_model(X, y):
    """Train the ASL classification model"""
    print("\n🤖 Training ASL model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MLPClassifier
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
    
    print("🔄 Training in progress...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"✅ Training complete!")
    print(f"📊 Training accuracy: {train_score:.3f}")
    print(f"📊 Test accuracy: {test_score:.3f}")
    
    # Detailed evaluation
    y_pred = model.predict(X_test_scaled)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler

def save_model(model, scaler):
    """Save trained model and scaler"""
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / "asl_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = model_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved to: {scaler_path}")

def main():
    """Main training function"""
    print("🎯 ASL Model Training")
    print("=" * 40)
    
    # Load data
    X, y = load_data()
    if X is None:
        return
    
    # Train model
    model, scaler = train_model(X, y)
    
    # Save model
    save_model(model, scaler)
    
    print("\n🎉 Training complete!")
    print("🚀 You can now run: python detection/realtime_predictor.py")

if __name__ == "__main__":
    main()

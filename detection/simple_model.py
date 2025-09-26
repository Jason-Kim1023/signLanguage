import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path

class SimpleASLModel:
    def __init__(self, data_path="data/raw/asl_letters.csv"):
        self.data_path = Path(data_path)
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
    def load_data(self):
        """Load the dataset"""
        if not self.data_path.exists():
            print("No dataset found. Run the data collector first.")
            return None, None
        
        df = pd.read_csv(self.data_path)
        print(f"Loaded dataset with {len(df)} samples")
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col.startswith('k')]
        X = df[feature_cols].values
        y = df['label'].values
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2):
        """Prepare data for training"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the neural network"""
        print("Training neural network...")
        
        # Create MLP classifier
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = self.model.score(X_test, y_test)
        print(f"\nModel Accuracy: {accuracy:.3f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def save_model(self, model_path="data/models"):
        """Save the trained model"""
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_dir / 'asl_model.pkl')
        joblib.dump(self.scaler, model_dir / 'scaler.pkl')
        
        print(f"Model saved to: {model_dir}")
    
    def load_model(self, model_path="data/models"):
        """Load a trained model"""
        model_dir = Path(model_path)
        
        self.model = joblib.load(model_dir / 'asl_model.pkl')
        self.scaler = joblib.load(model_dir / 'scaler.pkl')
        
        print(f"Model loaded from: {model_dir}")
    
    def predict(self, features):
        """Make a prediction on new data"""
        if self.model is None or self.scaler is None:
            print("Model not loaded!")
            return None
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0].max()
        
        return prediction, confidence

def main():
    print("🤖 Simple ASL Model Training")
    print("=" * 40)
    
    # Create model
    model = SimpleASLModel()
    
    # Load data
    X, y = model.load_data()
    if X is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(X, y)
    
    # Train model
    model.train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = model.evaluate_model(X_test, y_test)
    
    # Save model
    model.save_model()
    
    print(f"\n✅ Model training complete!")
    print(f"Accuracy: {accuracy:.3f}")
    print("Model saved and ready for use!")

if __name__ == "__main__":
    main()

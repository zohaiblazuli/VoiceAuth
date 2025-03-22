import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from utils import get_output_path, get_model_path  # Import path utilities
import argparse
import time

def train_model(features_path, model_output_path, test_size=0.2, random_state=42, callback=None, use_feature_selection=False):
    """
    Train a simplified logistic regression model with lower memory usage
    
    Parameters:
    -----------
    features_path : str
        Path to CSV file with extracted features
    model_output_path : str
        Path to save the trained model
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    callback : function
        Optional callback function to report progress (takes a string message)
    use_feature_selection : bool
        Whether to use feature selection to improve model
    """
    def report_progress(message):
        """Report progress via callback if available"""
        print(message)
        if callback:
            callback(message)
            time.sleep(0.05)  # Small delay to allow UI update
    
    report_progress(f"Loading features from {features_path}")
    try:
        features_df = pd.read_csv(features_path)
    except Exception as e:
        report_progress(f"Error loading features: {str(e)}")
        raise
    
    # Split data into features and labels
    report_progress("Preparing dataset...")
    try:
        X = features_df.drop(['label', 'file_path'], axis=1, errors='ignore').values
        y = features_df['label'].values
    except Exception as e:
        report_progress(f"Error preparing dataset: {str(e)}")
        raise
    
    report_progress(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    report_progress(f"Class distribution: {np.bincount(y)[0]} human, {np.bincount(y)[1]} AI")
    
    # Split into training and testing sets
    report_progress("Splitting dataset into training and validation sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except Exception as e:
        report_progress(f"Error splitting dataset: {str(e)}")
        raise
    
    # Scale features
    report_progress("Scaling features...")
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        report_progress(f"Error scaling features: {str(e)}")
        raise
    
    # Train a simple logistic regression model
    report_progress("Training logistic regression model...")
    try:
        model = LogisticRegression(
            C=1.0,
            solver='liblinear',
            penalty='l2',
            random_state=random_state,
            class_weight='balanced',
            max_iter=1000
        )
        
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        report_progress(f"Error training model: {str(e)}")
        raise
    
    # Make predictions
    report_progress("Evaluating model performance...")
    try:
        y_pred = model.predict(X_test_scaled)
    except Exception as e:
        report_progress(f"Error making predictions: {str(e)}")
        raise
    
    # Calculate metrics
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
    except Exception as e:
        report_progress(f"Error calculating metrics: {str(e)}")
        raise
    
    report_progress("Model Performance:")
    report_progress(f"Accuracy: {accuracy:.4f}")
    report_progress(f"Precision: {precision:.4f}")
    report_progress(f"Recall: {recall:.4f}")
    report_progress(f"F1 Score: {f1:.4f}")
    
    # Save model and scaler
    report_progress("Saving model and scaler...")
    try:
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        scaler_path = f"{os.path.splitext(model_output_path)[0]}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
    except Exception as e:
        report_progress(f"Error saving model: {str(e)}")
        raise
    
    report_progress(f"Model saved to {model_output_path}")
    report_progress(f"Scaler saved to {scaler_path}")
    
    return model, scaler

def classify_audio(features, model_path, scaler_path):
    """
    Classify audio based on pre-extracted features
    
    Parameters:
    -----------
    features : numpy array
        Extracted features from an audio file
    model_path : str
        Path to trained model
    scaler_path : str
        Path to fitted scaler
        
    Returns:
    --------
    is_ai : bool
        True if the voice is AI-generated, False if human
    confidence : float
        Probability of the prediction
    """
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Reshape features if needed
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get confidence
    is_ai = bool(prediction)
    confidence = probabilities[1] if is_ai else probabilities[0]
    
    return is_ai, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a voice classifier model.")
    parser.add_argument("--features", type=str, default=get_output_path("features.csv"),
                        help="Path to features CSV file")
    parser.add_argument("--output", type=str, default=get_model_path("voice_classifier.pkl"),
                        help="Path to save the model")
    parser.add_argument("--use_feature_selection", action="store_true",
                        help="Use feature selection to improve model")
    
    args = parser.parse_args()
    
    print("Starting model training with the following settings:")
    print(f"Features file: {args.features}")
    print(f"Model output: {args.output}")
    print(f"Use feature selection: {args.use_feature_selection}")
    
    train_model(
        features_path=args.features,
        model_output_path=args.output,
        use_feature_selection=args.use_feature_selection
    ) 
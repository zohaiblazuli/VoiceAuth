import argparse
import os
import joblib
import numpy as np
from batch_process import extract_features
from simple_model import classify_audio

def classify_single_file(audio_path, model_path):
    """
    Classify a single audio file using the trained model
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file to classify
    model_path : str
        Path to the trained model
    """
    if not os.path.exists(audio_path):
        print(f"Error: Audio file {audio_path} not found")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    # Get scaler path
    scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file {scaler_path} not found")
        return
    
    # Extract features
    print(f"Extracting features from {audio_path}...")
    features = extract_features(audio_path)
    
    if features is None:
        print("Error: Failed to extract features from the audio file")
        return
    
    # Classify audio
    print("Classifying audio...")
    is_ai, confidence = classify_audio(features, model_path, scaler_path)
    
    # Print result
    print("\n===== Classification Result =====")
    print(f"Audio file: {audio_path}")
    prediction = "AI-generated" if is_ai else "Human"
    print(f"Prediction: {prediction} voice")
    print(f"Confidence: {confidence*100:.2f}%")
    print("================================")
    
    # Ask for feedback
    while True:
        feedback = input("\nWas this prediction correct? (y/n): ").lower().strip()
        if feedback in ['y', 'n']:
            break
        print("Please enter 'y' or 'n'")
    
    if feedback == 'y':
        print("Thank you for confirming the prediction was correct!")
    else:
        print(f"Thank you for the feedback. The voice was actually a {'Human' if is_ai else 'AI-generated'} voice.")
        
        # Here you could implement code to save this feedback for model improvement
        # This would typically involve:
        # 1. Saving the features and correct label
        # 2. Periodically retraining the model with this feedback
    
    return is_ai, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a voice sample as AI-generated or human")
    parser.add_argument("--audio", type=str, required=True, 
                      help="Path to the audio file to classify")
    parser.add_argument("--model", type=str, default="output/models/voice_classifier.pkl", 
                      help="Path to the trained model")
    
    args = parser.parse_args()
    
    classify_single_file(args.audio, args.model) 
#!/usr/bin/env python3
"""
Launcher script for the Voice Classifier application
"""

import os
import sys
from app import main

if __name__ == "__main__":
    # Make sure output directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/models", exist_ok=True)
    
    print("Starting Voice Classifier Application...")
    print("Dataset paths are pre-configured to:")
    print("  - Dataset path: C:\\Project Resources\\dataset\\raw")
    print("  - AI folder: ai")
    print("  - Human folder: human")
    print("\nPlease follow these steps:")
    print("1. In the 'Feature Extraction' tab, verify the paths are correct")
    print("2. Click 'Extract Features' to process the dataset")
    print("3. In the 'Model Training' tab, click 'Train Model'")
    print("4. In the 'Voice Classification' tab, browse for a test audio file and click 'Classify Audio'")
    print("5. Provide feedback on the classification results to improve the model")
    
    # Launch the application
    main() 
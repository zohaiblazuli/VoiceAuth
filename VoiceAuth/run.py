#!/usr/bin/env python3
"""
Launcher script for the Voice Classifier application
"""

import os
import sys
from utils import get_output_path, get_model_path  # Import path utilities
from app import main

if __name__ == "__main__":
    # Use utilities to ensure output directories exist (these functions
    # internally create the directories if they don't exist)
    output_dir = get_output_path()
    model_dir = get_model_path()
    
    # Launch the application
    main() 
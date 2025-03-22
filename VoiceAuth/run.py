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
    
    # Launch the application
    main() 
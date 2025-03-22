#!/usr/bin/env python3
"""
Launcher script for the VoiceAuth application
"""

import os
import sys
from voiceauth import main

if __name__ == "__main__":
    # Make sure output directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/models", exist_ok=True)
    
    print("Starting VoiceAuth - AI Voice Detection System...")
    print("Developed by Zohaib Khan & Umer Kashif for Regeneron ISEF 2025")
    
    # Launch the application
    main() 
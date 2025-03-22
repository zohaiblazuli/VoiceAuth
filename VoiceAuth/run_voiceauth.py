#!/usr/bin/env python3
"""
Launcher script for the VoiceAuth application
"""

import sys
import os
from utils import get_model_path  # Import the utility function

def main():
    """Entry point for the VoiceAuth application"""
    # No need to manually create directories now, utils.py handles it
    
    # Import here to avoid circular imports
    from voiceauth import main as voiceauth_main
    
    # Run the main function from voiceauth.py
    voiceauth_main()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
"""
VoiceAuth Setup Checker
This script verifies that your system has all the necessary components
to run the VoiceAuth application.
"""

import sys
import platform
import os
import subprocess
import pkg_resources

def print_status(message, status, details=None):
    """Print a status message with color coding"""
    status_color = {
        "OK": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "INFO": "\033[94m"  # Blue
    }
    
    reset_color = "\033[0m"
    
    # Windows command prompt doesn't support ANSI colors by default
    if platform.system() == "Windows":
        print(f"{message}: {status}")
    else:
        print(f"{message}: {status_color.get(status, '')}{status}{reset_color}")
    
    if details:
        print(f"  {details}")
    
    print()

def check_python_version():
    """Check Python version"""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version >= required_version:
        print_status("Python version", "OK", f"Using Python {current_version.major}.{current_version.minor}.{current_version.micro}")
        return True
    else:
        print_status("Python version", "ERROR", 
                    f"Using Python {current_version.major}.{current_version.minor}.{current_version.micro}, but {required_version[0]}.{required_version[1]}+ is required.")
        return False

def check_c_compiler():
    """Check for C compiler"""
    if platform.system() == "Windows":
        # Try to find Visual C++ Compiler
        try:
            # This will list installed Visual C++ Redistributables
            result = subprocess.run(["dir", "/B", "C:\\Program Files (x86)\\Microsoft Visual Studio"], 
                                   shell=True, capture_output=True, text=True)
            
            if "2019" in result.stdout or "2022" in result.stdout:
                print_status("C++ Build Tools", "OK", "Visual Studio Build Tools appear to be installed")
                return True
            else:
                # Check alternative paths for build tools
                result = subprocess.run(["dir", "/B", "C:\\Program Files (x86)\\Microsoft Visual C++ Build Tools"], 
                                      shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print_status("C++ Build Tools", "OK", "Visual C++ Build Tools appear to be installed")
                    return True
                else:
                    print_status("C++ Build Tools", "WARNING", 
                                "Microsoft Visual C++ Build Tools may not be installed. This is required for compiling certain packages.")
                    print("Please install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
                    return False
        except Exception as e:
            print_status("C++ Build Tools", "WARNING", 
                        "Could not determine if Microsoft Visual C++ Build Tools are installed")
            print("Please install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            return False
    elif platform.system() == "Linux":
        # Check for gcc
        try:
            result = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print_status("C Compiler", "OK", f"GCC is installed: {result.stdout.splitlines()[0]}")
                return True
            else:
                print_status("C Compiler", "WARNING", "GCC does not appear to be installed")
                return False
        except Exception:
            print_status("C Compiler", "WARNING", "GCC does not appear to be installed")
            return False
    elif platform.system() == "Darwin":  # macOS
        # Check for xcode command line tools
        try:
            result = subprocess.run(["xcode-select", "-p"], capture_output=True, text=True)
            if result.returncode == 0:
                print_status("C Compiler", "OK", "Xcode Command Line Tools appear to be installed")
                return True
            else:
                print_status("C Compiler", "WARNING", 
                            "Xcode Command Line Tools may not be installed. Run 'xcode-select --install' to install.")
                return False
        except Exception:
            print_status("C Compiler", "WARNING", 
                        "Xcode Command Line Tools may not be installed. Run 'xcode-select --install' to install.")
            return False
    
    return False

def check_required_packages():
    """Check if key packages can be imported"""
    core_packages = ['numpy', 'pandas', 'matplotlib', 'librosa', 'sounddevice', 'PyQt5']
    missing_packages = []
    
    for package in core_packages:
        try:
            __import__(package)
            print_status(f"Package: {package}", "OK")
        except ImportError:
            print_status(f"Package: {package}", "ERROR", f"{package} is not installed or cannot be imported")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_audio_device():
    """Check if audio devices are available"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if input_devices:
            print_status("Audio input devices", "OK", f"Found {len(input_devices)} input device(s)")
            return True
        else:
            print_status("Audio input devices", "WARNING", "No audio input devices found")
            return False
    except Exception as e:
        print_status("Audio input devices", "WARNING", f"Could not check audio devices: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" VoiceAuth Setup Checker ")
    print("=" * 60 + "\n")
    
    print("Checking system requirements...\n")
    
    python_ok = check_python_version()
    compiler_ok = check_c_compiler()
    
    print("Checking installed packages...\n")
    packages_ok = check_required_packages()
    
    print("Checking audio devices...\n")
    audio_ok = check_audio_device()
    
    print("\n" + "=" * 60)
    print(" Summary ")
    print("=" * 60)
    
    if all([python_ok, packages_ok, audio_ok]):
        print_status("Overall setup", "OK", "Your system appears to be correctly configured for VoiceAuth")
        if not compiler_ok and platform.system() == "Windows":
            print("NOTE: C++ Build Tools might not be installed, but required packages are already working.")
    else:
        print_status("Overall setup", "WARNING", "Some components may need to be installed or configured")
        
        if not python_ok:
            print("- Please install Python 3.8 or higher")
        
        if not compiler_ok:
            if platform.system() == "Windows":
                print("- Install Microsoft C++ Build Tools:")
                print("  https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            elif platform.system() == "Linux":
                print("- Install GCC with: sudo apt-get install build-essential")
            elif platform.system() == "Darwin":
                print("- Install Xcode Command Line Tools with: xcode-select --install")
        
        if not packages_ok:
            print("- Install required packages with: pip install -r requirements.txt")
        
        if not audio_ok:
            print("- Check your audio device configuration")
    
    print("\nFor detailed setup instructions, please refer to the README.md file.") 
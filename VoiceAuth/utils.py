import os
import sys
import platform

def get_base_dir():
    """
    Get the base directory of the application, works in both development and when packaged.
    This handles cases whether running as script, frozen app or from different relative directories.
    """
    if getattr(sys, 'frozen', False):
        # If the application is frozen (executable)
        base_dir = os.path.dirname(sys.executable)
    else:
        # If running as script
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return base_dir

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    """
    base_dir = get_base_dir()
    return os.path.join(base_dir, relative_path)

def get_media_path(filename):
    """
    Get path to a media file
    """
    return get_resource_path(os.path.join('media', filename))

def get_output_path(filename=None):
    """
    Get path to an output file or directory
    """
    output_path = get_resource_path('output')
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    if filename:
        return os.path.join(output_path, filename)
    return output_path

def get_model_path(filename=None):
    """
    Get path to a model file or the models directory
    """
    models_path = os.path.join(get_output_path(), 'models')
    # Create the models directory if it doesn't exist
    os.makedirs(models_path, exist_ok=True)
    
    if filename:
        return os.path.join(models_path, filename)
    return models_path

def normalize_path(path):
    """
    Normalize a path to use the correct path separators for the current OS
    """
    return os.path.normpath(path)

def is_windows():
    """
    Check if the current OS is Windows
    """
    return platform.system() == 'Windows'

def is_macos():
    """
    Check if the current OS is macOS
    """
    return platform.system() == 'Darwin'

def is_linux():
    """
    Check if the current OS is Linux
    """
    return platform.system() == 'Linux' 
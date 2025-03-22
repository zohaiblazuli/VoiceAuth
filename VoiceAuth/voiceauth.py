import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QFileDialog, QLabel, QProgressBar, 
                           QWidget, QTabWidget, QTextEdit, QComboBox, QSpinBox,
                           QRadioButton, QButtonGroup, QListWidget, QGroupBox, 
                           QGridLayout, QSplitter, QMessageBox, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QPropertyAnimation, QEasingCurve, QRect, QSize
from PyQt5.QtGui import QFont, QPixmap, QColor, QPainter, QLinearGradient, QBrush, QPainterPath, QFontMetrics

from ui_components import GradientAnimatedBackground, LogoWidget, StylesheetProvider
from audio_processor import AudioProcessor
from simple_model import classify_audio, train_model
from batch_process import extract_features
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from animations import TabAnimator
from simple_startup import run_simple_startup
from utils import get_media_path, get_output_path, get_model_path

import tabs  # Import tabs module

# Global constants
VERSION = "1.2.9"
LAST_UPDATED = "March 22, 2025"
DEFAULT_MODEL_PATH = get_model_path("voice_classifier.pkl")
DEFAULT_FEATURES_PATH = get_output_path("features.csv")
FEEDBACK_FILE = get_model_path("feedback_data.csv")
VERSION_FILE = get_output_path("version_info.json")

# Gradient Text Label for main UI
class StaticGradientLabel(QLabel):
    """A simplified label with gradient text that doesn't use animation or custom painting"""
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        
        # Set font size
        font = QFont("Segoe UI", 90)  # Significantly increased size
        font.setWeight(QFont.Normal)
        self.setFont(font)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumHeight(120)  # Increased minimum height
        
        # Apply simple gradient stylesheet
        self.setStyleSheet("""
            QLabel {
                background-color: transparent;
                background: transparent;
                color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, 
                                      stop:0 rgba(157, 108, 211, 255), 
                                      stop:0.33 rgba(140, 130, 222, 255), 
                                      stop:0.66 rgba(123, 142, 220, 255), 
                                      stop:1 rgba(106, 159, 220, 255));
            }
        """)
        
        # Set alignment
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

class VoiceAuthApp(QMainWindow):
    """Main VoiceAuth application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiceAuth - AI Voice Detection")
        self.setMinimumSize(960, 720)  # Set larger minimum size for better layout
        
        # Initialize variables
        self.model_path = DEFAULT_MODEL_PATH
        self.features_path = DEFAULT_FEATURES_PATH
        self.feedback_file = FEEDBACK_FILE
        self.current_version = VERSION
        self.last_updated = LAST_UPDATED
        
        # Load version info if exists
        self.load_version_info()
        
        # Setup UI (but don't show it yet)
        self.setup_ui()
        
        # Show window to ensure proper sizing
        self.show()
        
        # Run startup animation (simplified version)
        self.startup_widget = run_simple_startup(
            self, 
            self.content_layout
        )
        
    def load_version_info(self):
        """Load version info from file if exists"""
        if os.path.exists(VERSION_FILE):
            try:
                with open(VERSION_FILE, 'r') as f:
                    info = json.load(f)
                    self.current_version = info.get('version', VERSION)
                    self.last_updated = info.get('last_updated', LAST_UPDATED)
            except:
                pass
    
    def save_version_info(self, version=None, date=None):
        """Save version info to file"""
        if version:
            self.current_version = version
        if date:
            self.last_updated = date
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(VERSION_FILE), exist_ok=True)
        
        # Save info
        with open(VERSION_FILE, 'w') as f:
            json.dump({
                'version': self.current_version,
                'last_updated': self.last_updated
            }, f, indent=4)
    
    def update_version(self):
        """Update version after model retraining"""
        # Parse current version
        parts = self.current_version.split('.')
        if len(parts) == 3:
            major, minor, patch = map(int, parts)
            patch += 1
            new_version = f"{major}.{minor}.{patch}"
            
            # Update version and date
            self.save_version_info(
                version=new_version,
                date=datetime.now().strftime("%B %d, %Y")
            )
            
            # Update info tab
            if hasattr(self, 'tabs') and hasattr(self.tabs, 'info_tab'):
                self.tabs.info_tab.update_version_info(
                    self.current_version, self.last_updated)
                
            # Update version info display
            if hasattr(self, 'version_label'):
                self.version_label.setText(f"<i>VoiceAuth version: {self.current_version}</i>")
    
    def setup_ui(self):
        """Set up the main UI components"""
        # Set stylesheet
        self.setStyleSheet(StylesheetProvider.get_app_stylesheet())
        
        # Create main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create main layout - change to QGridLayout for better scaling behavior
        main_layout = QGridLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create a simple static gradient background instead of animated one
        background = QWidget()
        background.setStyleSheet("""
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, 
                          stop:0 #1E2E5E, 
                          stop:0.33 rgba(89, 95, 154, 255), 
                          stop:0.66 rgba(176, 138, 217, 255), 
                          stop:1 rgba(215, 133, 161, 255));
        """)
        background.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(background, 0, 0)
        
        # Create content widget for elements on top of the background
        content_widget = QWidget()
        content_widget.setObjectName("contentWidget")
        content_widget.setStyleSheet("#contentWidget { background-color: transparent; }")
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(15)
        
        # Add version info label in the top-left corner
        version_container = QWidget()
        version_container.setStyleSheet("background-color: transparent; padding: 5px;")
        version_layout = QVBoxLayout(version_container)
        version_layout.setContentsMargins(10, 10, 10, 10)
        version_layout.setSpacing(2)
        
        self.version_label = QLabel(f"<i>VoiceAuth version: {self.current_version}</i>")
        self.version_label.setStyleSheet("font-style: italic; color: rgba(255, 255, 255, 0.8); font-size: 12px;")
        self.version_label.setAlignment(Qt.AlignLeft)
        
        # Changelog in bullet points
        changelog_label = QLabel("Changelog:")
        changelog_label.setStyleSheet("font-style: italic; color: rgba(255, 255, 255, 0.8); font-size: 11px;")
        changelog_label.setAlignment(Qt.AlignLeft)
        
        # Each bullet point as a separate label for better formatting
        changes_list = QVBoxLayout()
        changes_list.setSpacing(0)
        changes_list.setContentsMargins(10, 0, 0, 0)
        
        bullet_points = [
            "fixed media loading errors",
            "fixed aspect ratio bug",
            "fed umer's cat"
        ]
        
        for point in bullet_points:
            point_label = QLabel(f"• {point}")
            point_label.setStyleSheet("font-style: italic; color: rgba(255, 255, 255, 0.7); font-size: 11px;")
            point_label.setAlignment(Qt.AlignLeft)
            point_label.setWordWrap(True)
            changes_list.addWidget(point_label)
        
        version_layout.addWidget(self.version_label)
        version_layout.addWidget(changelog_label)
        version_layout.addLayout(changes_list)
        
        self.content_layout.addWidget(version_container, 0, Qt.AlignLeft)
        
        # Create centered header with app name and logo
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(10, 5, 10, 5)
        header_layout.setAlignment(Qt.AlignCenter)  # Center the entire layout
        
        # Create logo widget with proper size
        self.logo_widget = LogoWidget(header_container, logo_path=get_media_path("logo-no-shadow.png"), scale=1.2)
        self.logo_widget.setMinimumHeight(180)
        self.logo_widget.setMaximumHeight(220)
        self.logo_widget.setMinimumWidth(180)
        self.logo_widget.setMaximumWidth(220)
        self.logo_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Create app name label to right of logo
        self.app_name = QLabel()
        pixmap = QPixmap(get_media_path("text-only-white.png"))
        scaled_pixmap = pixmap.scaled(int(pixmap.width()*0.4), int(pixmap.height()*0.4), 
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.app_name.setPixmap(scaled_pixmap)
        self.app_name.setStyleSheet("background-color: transparent;")
        self.app_name.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.app_name.setMinimumHeight(80)  # Reduced minimum height
        self.app_name.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Add to header layout
        header_layout.addStretch(1)
        header_layout.addWidget(self.logo_widget)
        header_layout.addWidget(self.app_name)
        header_layout.addStretch(1)
        
        # Add header to main layout
        self.content_layout.addWidget(header_container)
        
        # Create standard tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Create tabs
        self.tabs = tabs.TabManager(
            parent=self,
            model_path=self.model_path,
            features_path=self.features_path,
            feedback_file=self.feedback_file,
            version=self.current_version,
            last_updated=self.last_updated
        )
        
        # Add tabs to the tab widget
        self.tab_widget.addTab(self.tabs.import_tab, "Import Sample")
        self.tab_widget.addTab(self.tabs.record_tab, "Record Sample")
        self.tab_widget.addTab(self.tabs.feedback_tab, "Feedback Manager")
        self.tab_widget.addTab(self.tabs.info_tab, "Information")
        
        # Add tab animator
        self.tab_animator = TabAnimator(self.tab_widget, animation_type="slide_fade")
        
        # Add the tab widget to the content layout with stretch factor
        self.content_layout.addWidget(self.tab_widget, 1)  # 1 is the stretch factor
        
        # Add the content widget on top of the background
        main_layout.addWidget(content_widget, 0, 0)
    
    def change_animation(self, index):
        """Change the tab transition animation type"""
        animation_types = ["fade", "slide", "slide_fade", "none"]
        if 0 <= index < len(animation_types) and hasattr(self, 'tab_animator'):
            self.tab_animator.set_animation_type(animation_types[index])
    
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        # Update startup animation widget position if it exists and hasn't been deleted
        if hasattr(self, 'startup_widget') and self.startup_widget:
            try:
                self.startup_widget.setGeometry(self.rect())
            except RuntimeError:
                # Widget may have been deleted
                pass

def main():
    app = QApplication(sys.argv)
    window = VoiceAuthApp()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 
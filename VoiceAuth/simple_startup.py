"""
Simple and reliable startup animation for VoiceAuth.
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                         QLabel, QPushButton, QProgressBar, QSpacerItem, 
                         QSizePolicy, QDesktopWidget, QGraphicsOpacityEffect)
from PyQt5.QtCore import Qt, QTimer, QSize, QRect, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase, QLinearGradient, QColor, QBrush, QPainter
from utils import get_media_path

class SimpleStartupAnimation(QWidget):
    """A simple and reliable startup animation widget"""
    
    def __init__(self, parent=None, main_layout=None):
        super().__init__(parent)
        self.main_layout = main_layout
        
        # Set up widget properties
        self.setGeometry(parent.rect() if parent else QRect(0, 0, 800, 600))
        self.setStyleSheet("background-color: transparent;")
        
        # Hide main UI elements
        self.hide_main_widgets()
        
        # Set up the UI
        self.setup_ui()
        
        # Set up fallback timer (10 seconds maximum)
        QTimer.singleShot(10000, self.fallback_show_ui)
        
        # Start animation sequence
        QTimer.singleShot(300, self.start_animation)
    
    def setup_ui(self):
        """Set up UI elements"""
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Add spacer to center content vertically
        layout.addStretch(1)
        
        # Create a container for centered content
        center_container = QHBoxLayout()
        center_container.addStretch(1)  # Add stretch before content for centering
        
        # Create content container
        content_container = QVBoxLayout()
        content_container.setAlignment(Qt.AlignCenter)
        
        # Create app name label
        self.app_name = QLabel()
        pixmap = QPixmap(get_media_path("text-only-white.png"))
        scaled_pixmap = pixmap.scaled(int(pixmap.width()*0.4), int(pixmap.height()*0.4), 
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.app_name.setPixmap(scaled_pixmap)
        self.app_name.setAlignment(Qt.AlignCenter)
        self.app_name.setStyleSheet("background-color: transparent;")
        self.app_name.setMinimumHeight(200)  # Reduced height
        
        # Add app name to content container
        content_container.addWidget(self.app_name)
        
        # Add spacing between title and message
        content_container.addSpacing(20)
        
        # Create loading message label
        self.message = QLabel("loading UI...")
        self.message.setFont(QFont("Segoe UI", 36))
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setStyleSheet("color: rgba(255, 255, 255, 0.7); background-color: transparent;")
        content_container.addWidget(self.message)
        
        # Add content container to center container
        center_container.addLayout(content_container)
        center_container.addStretch(1)  # Add stretch after content for centering
        
        # Add center container to main layout
        layout.addLayout(center_container)
        
        # Add bottom spacer for vertical centering
        layout.addStretch(1)
    
    def hide_main_widgets(self):
        """Hide all widgets in the main layout"""
        if self.main_layout:
            for i in range(self.main_layout.count()):
                item = self.main_layout.itemAt(i)
                if item and item.widget():
                    item.widget().hide()
    
    def show_main_widgets(self):
        """Show all widgets in the main layout"""
        if self.main_layout:
            for i in range(self.main_layout.count()):
                item = self.main_layout.itemAt(i)
                if item and item.widget():
                    item.widget().show()
    
    def start_animation(self):
        """Start the animation sequence"""
        print("Starting animation sequence")
        
        try:
            # Create opacity effect for title
            effect = QGraphicsOpacityEffect(self.app_name)
            effect.setOpacity(0)
            self.app_name.setGraphicsEffect(effect)
            
            # Create fade-in animation
            self.anim = QPropertyAnimation(effect, b"opacity")
            self.anim.setDuration(1000)
            self.anim.setStartValue(0.0)
            self.anim.setEndValue(1.0)
            self.anim.setEasingCurve(QEasingCurve.InOutQuad)
            self.anim.finished.connect(self.show_loading_messages)
            
            # Start animation
            self.anim.start()
        except Exception as e:
            print(f"Error in start_animation: {e}")
            # Fall back to showing UI if there's an error
            self.show_loading_messages()
    
    def show_loading_messages(self):
        """Show loading messages in sequence"""
        self.message_index = 0
        self.messages = [
            "loading assets..",
            "initializing audio processor...",
            "yelling DO NOT REDEEM IT....",
            "petting umer's cat..",
            "trying not to procrastinate....",
            "successfully deleted indians from earth & imported model!"
        ]
        self.update_message()
    
    def update_message(self):
        """Update to the next loading message"""
        try:
            if self.message_index < len(self.messages):
                self.message.setText(self.messages[self.message_index])
                print(f"Message: {self.messages[self.message_index]}")
                self.message_index += 1
                
                # Schedule next message or finish
                if self.message_index < len(self.messages):
                    QTimer.singleShot(700, self.update_message)
                else:
                    # All messages shown, fade out
                    QTimer.singleShot(700, self.finish_animation)
        except Exception as e:
            print(f"Error in update_message: {e}")
            self.finish_animation()
    
    def finish_animation(self):
        """Finish animation and show main UI"""
        print("Finishing animation")
        
        try:
            # Create fade-out effect for the whole widget
            effect = QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(effect)
            
            # Create animation
            anim = QPropertyAnimation(effect, b"opacity")
            anim.setDuration(1000)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.setEasingCurve(QEasingCurve.InOutQuad)
            anim.finished.connect(self.show_ui)
            
            # Start animation
            anim.start()
        except Exception as e:
            print(f"Error in finish_animation: {e}")
            self.show_ui()
    
    def show_ui(self):
        """Show the main UI"""
        print("Animation complete, showing UI")
        self.show_main_widgets()
        try:
            # Schedule deletion to avoid resource conflicts
            QTimer.singleShot(100, self.deleteLater)
        except Exception as e:
            print(f"Error cleaning up animation: {e}")
    
    def fallback_show_ui(self):
        """Fallback function to ensure UI is shown even if animation fails"""
        print("Fallback timer triggered")
        self.show_main_widgets()
        try:
            self.deleteLater()
        except Exception as e:
            print(f"Error in fallback cleanup: {e}")

def run_simple_startup(main_window, main_layout):
    """Run the simplified startup animation"""
    try:
        anim = SimpleStartupAnimation(main_window, main_layout)
        anim.show()
        return anim
    except Exception as e:
        print(f"Error in startup animation: {str(e)}")
        # Show the main UI immediately
        if main_layout:
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item and item.widget():
                    item.widget().show()
        return None 
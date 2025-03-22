from PyQt5.QtWidgets import QWidget, QLabel, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QLinearGradient, QPalette, QFont, QPixmap, QBrush
from PyQt5.QtCore import Qt, QPropertyAnimation, QPoint, QEasingCurve, QTimer, pyqtProperty
import sys
import os
import math
import random
from utils import get_media_path

class GradientAnimatedBackground(QWidget):
    """
    A widget that displays an animated gradient background.
    The gradient shifts colors gradually, creating a subtle movement effect.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Get the color palette from the background.png
        # These are extrapolated from the image
        self.colors = [
            QColor(30, 46, 94),    # Deep blue
            QColor(46, 56, 114),   # Medium blue
            QColor(89, 95, 154),   # Periwinkle blue
            QColor(176, 138, 217), # Lavender purple
            QColor(215, 133, 161), # Pink
            QColor(232, 170, 131)  # Peach
        ]
        
        # Animation state variables
        self.animation_time = 0
        self.animation_speed = 0.0005  # Even slower animation speed for smoother transitions
        
        # Set background role and attributes for better performance
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        
        # Create intermediate colors for smoother animation
        self.expanded_colors = []
        for i in range(len(self.colors)):
            next_idx = (i + 1) % len(self.colors)
            # Add 5 intermediate colors between each pair (reduced from 10)
            for j in range(5):
                t = j / 5.0
                interp_color = self.blend_colors(self.colors[i], self.colors[next_idx], t)
                self.expanded_colors.append(interp_color)
        
        # Start the animation timer with a lower refresh rate for better performance
        self.startTimer()
    
    def startTimer(self):
        """Start the animation timer with proper error handling"""
        try:
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_animation)
            self.timer.start(50)  # 20fps is sufficient for a background animation
        except Exception as e:
            print(f"Error starting background animation timer: {e}")
    
    def update_animation(self):
        """Update animation state and trigger repaint"""
        try:
            self.animation_time += self.animation_speed
            if self.animation_time > 1.0:
                self.animation_time = 0.0
            self.update()  # Request a repaint
        except Exception as e:
            print(f"Error in update_animation: {e}")
            # Restart timer if there was an error
            self.timer.stop()
            QTimer.singleShot(1000, self.startTimer)
    
    def paintEvent(self, event):
        """Paint the gradient background with animated colors"""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            
            # Make sure expanded_colors isn't empty
            if not hasattr(self, 'expanded_colors') or len(self.expanded_colors) == 0:
                # Create a simple gradient as fallback
                gradient = QLinearGradient(0, 0, self.width(), self.height())
                gradient.setColorAt(0.0, QColor(30, 46, 94))
                gradient.setColorAt(1.0, QColor(176, 138, 217))
                painter.fillRect(self.rect(), gradient)
                return
            
            # Use expanded colors for smoother transitions
            num_colors = len(self.expanded_colors)
            color_index = int(self.animation_time * num_colors) % num_colors
            next_color_index = (color_index + 1) % num_colors
            next_next_index = (color_index + int(num_colors/4)) % num_colors
            next_next_next_index = (next_next_index + 1) % num_colors
            
            # Calculate blend factor for current pair of colors
            blend_factor = (self.animation_time * num_colors) % 1.0
            
            # Create gradient with better color distribution
            gradient = QLinearGradient(0, 0, self.width(), self.height())
            
            # Blend colors for smooth transitions
            start_color = self.blend_colors(
                self.expanded_colors[color_index], 
                self.expanded_colors[next_color_index], 
                blend_factor
            )
            
            end_color = self.blend_colors(
                self.expanded_colors[next_next_index], 
                self.expanded_colors[next_next_next_index], 
                blend_factor
            )
            
            # Simplified gradient with fewer color stops
            gradient.setColorAt(0.0, start_color)
            gradient.setColorAt(1.0, end_color)
            
            # Fill rectangle with the gradient
            painter.fillRect(self.rect(), gradient)
            
        except Exception as e:
            print(f"Error in paintEvent: {e}")
            # Fallback to solid color if there's an error
            painter.fillRect(self.rect(), QColor(30, 46, 94))
    
    def blend_colors(self, color1, color2, factor):
        """Blend two colors according to the given factor (0.0 - 1.0)"""
        try:
            # Simpler linear blending
            r = int(color1.red() * (1 - factor) + color2.red() * factor)
            g = int(color1.green() * (1 - factor) + color2.green() * factor)
            b = int(color1.blue() * (1 - factor) + color2.blue() * factor)
            return QColor(r, g, b)
        except Exception as e:
            print(f"Error in blend_colors: {e}")
            return QColor(30, 46, 94)  # Return default color on error

class LogoWidget(QLabel):
    """
    A widget that displays the logo with a drop shadow effect
    """
    
    def __init__(self, parent=None, logo_path=None, scale=0.85):
        super().__init__(parent)
        
        # Use default logo path if none provided
        if logo_path is None:
            logo_path = get_media_path("logo-no-shadow.png")
            
        self.pixmap = QPixmap(logo_path)
        self.scale_factor = scale
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: transparent;")
        self.setup_logo()
        
    def setup_logo(self):
        """Load and set up the logo"""
        # Load logo pixmap
        self.pixmap = QPixmap(self.pixmap)
        if self.pixmap.isNull():
            print(f"Error: Could not load logo from {self.pixmap}")
            return
        
        # Apply initial scaling
        self.update_logo_size()
        
    def update_logo_size(self):
        """Update logo size based on current widget size"""
        if self.pixmap is None or self.pixmap.isNull():
            return
            
        # Get current dimensions
        current_width = self.width()
        current_height = self.height()
        
        # Calculate maximum dimensions while maintaining aspect ratio
        max_height = min(current_height, 150) * self.scale_factor  # Limit maximum height
        max_width = current_width * self.scale_factor
        
        # Scale pixmap maintaining aspect ratio
        scaled_pixmap = self.pixmap.scaled(
            int(max_width), 
            int(max_height),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Update the logo label
        self.setPixmap(scaled_pixmap)
        self.setFixedSize(scaled_pixmap.width(), scaled_pixmap.height())
        
        # Center the logo label
        self.move(
            int((current_width - self.width()) / 2),
            int((current_height - self.height()) / 2)
        )
        
    def resizeEvent(self, event):
        """Handle resize events to update logo size"""
        super().resizeEvent(event)
        self.update_logo_size()

class StylesheetProvider:
    """Provides consistent stylesheets for the application"""
    
    @staticmethod
    def get_app_stylesheet():
        """Return the main application stylesheet"""
        return """
        QMainWindow {
            background-color: #1E2E5E;
        }
        
        QTabWidget::pane {
            border: none;
            background: transparent;
        }
        
        QTabBar::tab {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            padding: 12px 20px;
            margin-right: 4px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-family: 'Segoe UI', Arial;
            font-size: 13px;
            font-weight: 500;
            min-width: 120px;
        }
        
        QTabBar::tab:selected {
            background: rgba(255, 255, 255, 0.25);
            color: white;
        }
        
        QTabBar::tab:hover:!selected {
            background: rgba(255, 255, 255, 0.15);
        }
        
        QPushButton {
            background: rgba(255, 255, 255, 0.15);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-family: 'Segoe UI', Arial;
            font-size: 13px;
            font-weight: 500;
        }
        
        QPushButton:hover {
            background: rgba(255, 255, 255, 0.25);
        }
        
        QPushButton:pressed {
            background: rgba(255, 255, 255, 0.3);
        }
        
        QPushButton:disabled {
            background: rgba(255, 255, 255, 0.05);
            color: rgba(255, 255, 255, 0.4);
        }
        
        QLabel {
            color: white;
            font-family: 'Segoe UI', Arial;
            font-size: 13px;
        }
        
        QGroupBox {
            color: white;
            font-family: 'Segoe UI', Arial;
            font-size: 14px;
            font-weight: 500;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            padding-top: 15px;
            margin-top: 10px;
            background: rgba(0, 0, 0, 0.1);
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 10px;
        }
        
        QLineEdit, QTextEdit {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            padding: 6px;
            color: white;
            font-family: 'Segoe UI', Arial;
            selection-background-color: rgba(176, 138, 217, 0.5);
        }
        
        QProgressBar {
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            background: rgba(0, 0, 0, 0.2);
            text-align: center;
            color: white;
            font-family: 'Segoe UI', Arial;
            font-size: 12px;
        }
        
        QProgressBar::chunk {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                              stop:0 rgba(176, 138, 217, 0.8), 
                              stop:1 rgba(215, 133, 161, 0.8));
            border-radius: 3px;
        }
        
        QRadioButton {
            color: white;
            font-family: 'Segoe UI', Arial;
            spacing: 8px;
        }
        
        QRadioButton::indicator {
            width: 18px;
            height: 18px;
            border-radius: 9px;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        
        QRadioButton::indicator:checked {
            background: qradialgradient(cx:0.5, cy:0.5, radius:0.4, fx:0.5, fy:0.5, 
                          stop:0 rgba(215, 133, 161, 1), 
                          stop:0.6 rgba(215, 133, 161, 1), 
                          stop:0.7 rgba(0, 0, 0, 0), 
                          stop:1 rgba(0, 0, 0, 0));
        }
        
        QListWidget {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            color: white;
            font-family: 'Segoe UI', Arial;
            padding: 4px;
        }
        
        QListWidget::item {
            padding: 6px;
            border-radius: 4px;
        }
        
        QListWidget::item:selected {
            background: rgba(176, 138, 217, 0.3);
        }
        
        QListWidget::item:hover:!selected {
            background: rgba(255, 255, 255, 0.1);
        }
        
        QScrollBar:vertical {
            border: none;
            background: rgba(0, 0, 0, 0.1);
            width: 12px;
            margin: 0px 0px 0px 0px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background: rgba(255, 255, 255, 0.2);
            min-height: 20px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QScrollBar:horizontal {
            border: none;
            background: rgba(0, 0, 0, 0.1);
            height: 12px;
            margin: 0px 0px 0px 0px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:horizontal {
            background: rgba(255, 255, 255, 0.2);
            min-width: 20px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        """
        
    @staticmethod
    def get_title_label_style():
        """Style for title labels"""
        return """
        QLabel {
            color: white;
            font-family: 'Segoe UI', Arial;
            font-size: 20px;
            font-weight: 600;
        }
        """
    
    @staticmethod
    def get_subtitle_label_style():
        """Style for subtitle labels"""
        return """
        QLabel {
            color: rgba(255, 255, 255, 0.8);
            font-family: 'Segoe UI', Arial;
            font-size: 14px;
            font-weight: 400;
        }
        """
    
    @staticmethod
    def get_status_label_style():
        """Style for status labels"""
        return """
        QLabel {
            color: rgba(255, 255, 255, 0.7);
            font-family: 'Segoe UI', Arial;
            font-size: 12px;
            font-style: italic;
        }
        """
    
    @staticmethod
    def get_header_label_style():
        """Style for header labels"""
        return """
        QLabel {
            color: white;
            font-family: 'Segoe UI', Arial;
            font-size: 16px;
            font-weight: 600;
        }
        """
    
    @staticmethod
    def get_result_label_style(is_ai=True):
        """Style for result labels, with color based on AI or human result"""
        color = "#D7859D" if is_ai else "#85A3DA"  # Pink for AI, Blue for human
        return f"""
        QLabel {{
            color: {color};
            font-family: 'Segoe UI', Arial;
            font-size: 32px;
            font-weight: 700;
            qproperty-alignment: AlignCenter;
        }}
        """
    
    @staticmethod
    def get_primary_button_style():
        """Style for primary action buttons"""
        return """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                      stop:0 rgba(176, 138, 217, 0.8), 
                      stop:1 rgba(215, 133, 161, 0.8));
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-family: 'Segoe UI', Arial;
            font-size: 14px;
            font-weight: 600;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                      stop:0 rgba(176, 138, 217, 0.9), 
                      stop:1 rgba(215, 133, 161, 0.9));
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                      stop:0 rgba(176, 138, 217, 1), 
                      stop:1 rgba(215, 133, 161, 1));
        }
        
        QPushButton:disabled {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                      stop:0 rgba(176, 138, 217, 0.3), 
                      stop:1 rgba(215, 133, 161, 0.3));
            color: rgba(255, 255, 255, 0.4);
        }
        """ 
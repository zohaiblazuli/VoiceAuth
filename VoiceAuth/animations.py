"""
Animation components for the VoiceAuth application.
This module provides animations for UI transitions.
"""

from PyQt5.QtCore import (Qt, QPropertyAnimation, QEasingCurve, 
                         QParallelAnimationGroup, QSequentialAnimationGroup,
                         QPoint, QSize, QRect, QTimer, pyqtProperty)
from PyQt5.QtWidgets import QWidget, QTabWidget, QGraphicsOpacityEffect, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QFont, QColor, QPainter, QLinearGradient, QPalette, QPen

class TabAnimator:
    """
    A simple, reliable tab animator that adds transitions to a standard QTabWidget
    """
    def __init__(self, tab_widget, animation_type="fade"):
        """
        Initialize the tab animator
        
        Parameters:
        -----------
        tab_widget : QTabWidget
            The tab widget to animate
        animation_type : str
            Type of animation to use:
            - "fade": Fade in/out (default)
            - "slide": Slide horizontally
            - "slide_fade": Slide with fade
            - "none": No animation
        """
        self.tab_widget = tab_widget
        self.animation_type = animation_type
        self.animation_duration = 300  # milliseconds
        self.current_index = tab_widget.currentIndex()
        self.is_animating = False
        self.current_animation = None
        
        # Connect to tab widget's current changed signal
        self.tab_widget.currentChanged.connect(self.animate_tab_transition)

    def animate_tab_transition(self, new_index):
        """Animate the transition to a new tab"""
        # Don't animate if we're already animating
        if self.is_animating or self.animation_type == "none":
            return
            
        # Set animating flag
        self.is_animating = True
        
        # Get the current widget
        current_widget = self.tab_widget.widget(new_index)
        if not current_widget:
            self.is_animating = False
            return
            
        try:
            if self.animation_type == "fade":
                self._fade_animation(current_widget)
            elif self.animation_type == "slide":
                self._slide_animation(current_widget)
            elif self.animation_type == "slide_fade":
                self._slide_fade_animation(current_widget)
            else:
                # Default to no animation
                self.is_animating = False
        except Exception as e:
            print(f"Animation error: {str(e)}")
            self.is_animating = False
            
        # Update current index
        self.current_index = new_index
            
    def _fade_animation(self, widget):
        """Simple fade in animation"""
        # Create opacity effect
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        
        # Create animation
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(self.animation_duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        
        # Connect finished signal
        animation.finished.connect(self._animation_finished)
        
        # Start animation
        self.current_animation = animation
        animation.start()
        
    def _slide_animation(self, widget):
        """Simple slide animation"""
        # Store the original geometry
        original_geometry = widget.geometry()
        widget_width = original_geometry.width()
        
        # Set up slide direction (from right to left)
        start_x = original_geometry.x() + widget_width
        end_x = original_geometry.x()
        
        # Create animation
        animation = QPropertyAnimation(widget, b"pos")
        animation.setDuration(self.animation_duration)
        animation.setStartValue(QPoint(start_x, original_geometry.y()))
        animation.setEndValue(QPoint(end_x, original_geometry.y()))
        animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Connect finished signal
        animation.finished.connect(self._animation_finished)
        
        # Start animation
        self.current_animation = animation
        animation.start()
        
    def _slide_fade_animation(self, widget):
        """Combined slide and fade animation"""
        # Create opacity effect
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        
        # Store the original geometry
        original_geometry = widget.geometry()
        
        # Create parallel animation group
        animation_group = QParallelAnimationGroup()
        
        # Add fade animation
        fade_anim = QPropertyAnimation(effect, b"opacity")
        fade_anim.setDuration(self.animation_duration)
        fade_anim.setStartValue(0.0)
        fade_anim.setEndValue(1.0)
        fade_anim.setEasingCurve(QEasingCurve.InOutQuad)
        animation_group.addAnimation(fade_anim)
        
        # Add slide animation (more subtle - just a small shift)
        slide_anim = QPropertyAnimation(widget, b"pos")
        slide_anim.setDuration(self.animation_duration)
        start_x = original_geometry.x() + 40  # Just a small shift, not full width
        end_x = original_geometry.x()
        slide_anim.setStartValue(QPoint(start_x, original_geometry.y()))
        slide_anim.setEndValue(QPoint(end_x, original_geometry.y()))
        slide_anim.setEasingCurve(QEasingCurve.OutCubic)
        animation_group.addAnimation(slide_anim)
        
        # Connect finished signal
        animation_group.finished.connect(self._animation_finished)
        
        # Start animation
        self.current_animation = animation_group
        animation_group.start()
        
    def _animation_finished(self):
        """Called when animation finishes"""
        # Clear any graphics effects
        current_widget = self.tab_widget.widget(self.tab_widget.currentIndex())
        if current_widget and current_widget.graphicsEffect():
            current_widget.setGraphicsEffect(None)
            
        # Reset animation state
        self.is_animating = False
        self.current_animation = None
        
    def set_animation_type(self, animation_type):
        """Change the animation type"""
        valid_types = ["fade", "slide", "slide_fade", "none"]
        if animation_type in valid_types:
            self.animation_type = animation_type

    def set_animation_duration(self, duration):
        """Set the animation duration in milliseconds"""
        if 50 <= duration <= 1000:  # Reasonable limits
            self.animation_duration = duration

class StartupAnimation(QWidget):
    """
    Creates a startup animation for the application
    with gradient text animation and transition to main UI
    """
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.animation_finished = False
        
        # Set up widget properties
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Animation state - use underscore prefix for private properties
        self._gradient_position = 0.0
        self._text_position = QPoint(0, 0)
        self._text_size = 80  # Initial large size
        self._opacity = 1.0
        
        # Setup UI
        self.setup_ui()
        
        # Start animation sequence after a short delay
        QTimer.singleShot(300, self.start_animation_sequence)
    
    def setup_ui(self):
        """Set up the UI components"""
        # Make widget full-screen over parent
        if self.parent():
            self.setGeometry(self.parent().rect())
        else:
            self.resize(800, 600)
        
        # Create main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create label for app name
        self.app_name_label = QLabel("VoiceAuth")
        self.app_name_label.setFont(QFont("Segoe UI", self._text_size, QFont.Bold))
        self.app_name_label.setAlignment(Qt.AlignCenter)
        self.app_name_label.setStyleSheet("color: white;")
        
        # Add label to layout with stretch for vertical centering
        layout.addStretch(1)
        layout.addWidget(self.app_name_label)
        layout.addStretch(1)
    
    def start_animation_sequence(self):
        """Start the animation sequence"""
        # 1. Gradient animation
        self.gradient_anim = QPropertyAnimation(self, b"gradientPosition")
        self.gradient_anim.setStartValue(0.0)
        self.gradient_anim.setEndValue(1.0)
        self.gradient_anim.setDuration(2000)
        self.gradient_anim.setEasingCurve(QEasingCurve.InOutQuad)
        
        # 2. Wait briefly at the end of the gradient animation
        self.pause_anim = QPropertyAnimation(self, b"gradientPosition")
        self.pause_anim.setStartValue(1.0)
        self.pause_anim.setEndValue(1.0)
        self.pause_anim.setDuration(300)
        
        # 3. Final fade out animation for the splash screen
        self.fade_anim = QPropertyAnimation(self, b"opacity")
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.setDuration(500)
        self.fade_anim.setEasingCurve(QEasingCurve.InQuad)
        
        # Create sequential animation group
        self.anim_sequence = QSequentialAnimationGroup()
        self.anim_sequence.addAnimation(self.gradient_anim)
        self.anim_sequence.addAnimation(self.pause_anim)
        self.anim_sequence.addAnimation(self.fade_anim)
        
        # Connect to finished signal
        self.anim_sequence.finished.connect(self.animation_complete)
        
        # Start the animation sequence
        self.anim_sequence.start()
    
    def animation_complete(self):
        """Called when animation sequence is complete"""
        self.animation_finished = True
        
        # Show the main window and hide the animation
        if self.main_window:
            self.main_window.show()
        
        # Hide this widget
        self.hide()
    
    # Property for gradient position animation
    def _get_gradient_position(self):
        return self._gradient_position
    
    def _set_gradient_position(self, pos):
        self._gradient_position = pos
        self.update()  # Trigger repaint
    
    gradientPosition = pyqtProperty(float, _get_gradient_position, _set_gradient_position)
    
    # Property for text position animation
    def _get_text_position(self):
        return self._text_position
    
    def _set_text_position(self, pos):
        self._text_position = pos
        self.update()
    
    textPosition = pyqtProperty(QPoint, _get_text_position, _set_text_position)
    
    # Property for text size animation
    def _get_text_size(self):
        return self._text_size
    
    def _set_text_size(self, size):
        self._text_size = size
        self.app_name_label.setFont(QFont("Segoe UI", size, QFont.Bold))
        self.update()
    
    textSize = pyqtProperty(int, _get_text_size, _set_text_size)
    
    # Property for opacity animation
    def _get_opacity(self):
        return self._opacity
    
    def _set_opacity(self, opacity):
        self._opacity = opacity
        self.setWindowOpacity(opacity)
        self.update()
    
    opacity = pyqtProperty(float, _get_opacity, _set_opacity)
    
    def paintEvent(self, event):
        """Custom paint event to draw the gradient text"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background (dark blue-purple)
        painter.fillRect(self.rect(), QColor(30, 30, 50))
        
        # Don't draw label normally - we'll draw it manually with gradient
        self.app_name_label.hide()
        
        # Get the text and font from the label
        text = self.app_name_label.text()
        font = self.app_name_label.font()
        
        # Calculate text position - center in widget
        text_rect = self.rect()
        
        # Prepare gradient for text (purple to blue gradient)
        gradient = QLinearGradient(0, 0, text_rect.width(), 0)  # Horizontal gradient
        gradient.setColorAt(0.0, QColor(165, 126, 241))  # Light purple
        gradient.setColorAt(0.5, QColor(130, 145, 230))  # Medium blue-purple
        gradient.setColorAt(1.0, QColor(95, 164, 219))   # Light blue
        
        # Apply animation position to gradient
        anim_pos = self._gradient_position
        # Animate gradient from left to right
        gradient.setStart(text_rect.width() * -0.5 + (text_rect.width() * 1.5 * anim_pos), 0)
        gradient.setFinalStop(text_rect.width() * 0.5 + (text_rect.width() * 1.5 * anim_pos), 0)
        
        # Apply gradient as text color
        painter.setFont(font)
        painter.setPen(QPen(gradient, 1))
        
        # Draw text centered in widget
        painter.drawText(text_rect, Qt.AlignCenter, text)

class TransitionAnimation:
    """Helper class for creating widget transition animations"""
    
    @staticmethod
    def fade(widget, duration=300, start_value=0, end_value=1):
        """Create and return a fade animation"""
        from PyQt5.QtWidgets import QGraphicsOpacityEffect
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        
        fade_anim = QPropertyAnimation(effect, b"opacity")
        fade_anim.setDuration(duration)
        fade_anim.setStartValue(start_value)
        fade_anim.setEndValue(end_value)
        fade_anim.setEasingCurve(QEasingCurve.OutCubic)
        
        return fade_anim
    
    @staticmethod
    def slide(widget, direction="right", duration=300):
        """
        Create and return a slide animation
        
        Parameters:
        -----------
        widget : QWidget
            Widget to animate
        direction : str
            Direction to slide from: "left", "right", "up", "down"
        duration : int
            Animation duration in milliseconds
        """
        pos_anim = QPropertyAnimation(widget, b"pos")
        pos_anim.setDuration(duration)
        
        start_pos = widget.pos()
        end_pos = widget.pos()
        
        if direction == "right":
            start_pos = QPoint(widget.pos().x() + widget.width(), widget.pos().y())
        elif direction == "left":
            start_pos = QPoint(widget.pos().x() - widget.width(), widget.pos().y())
        elif direction == "down":
            start_pos = QPoint(widget.pos().x(), widget.pos().y() + widget.height())
        elif direction == "up":
            start_pos = QPoint(widget.pos().x(), widget.pos().y() - widget.height())
        
        pos_anim.setStartValue(start_pos)
        pos_anim.setEndValue(end_pos)
        pos_anim.setEasingCurve(QEasingCurve.OutCubic)
        
        return pos_anim
    
    @staticmethod
    def scale(widget, start_scale=0.8, end_scale=1.0, duration=300):
        """Create and return a scale animation (approximation)"""
        # Store original geometry
        original_geometry = widget.geometry()
        center_x = original_geometry.x() + original_geometry.width() / 2
        center_y = original_geometry.y() + original_geometry.height() / 2
        
        # Create a new animation for the widget's geometry
        geom_anim = QPropertyAnimation(widget, b"geometry")
        geom_anim.setDuration(duration)
        
        start_width = int(original_geometry.width() * start_scale)
        start_height = int(original_geometry.height() * start_scale)
        start_x = int(center_x - start_width / 2)
        start_y = int(center_y - start_height / 2)
        
        geom_anim.setStartValue(QRect(start_x, start_y, start_width, start_height))
        geom_anim.setEndValue(original_geometry)
        geom_anim.setEasingCurve(QEasingCurve.OutCubic)
        
        return geom_anim
    
    @staticmethod
    def bounce(widget, height=20, duration=500):
        """Create a bounce animation"""
        pos_anim = QPropertyAnimation(widget, b"pos")
        pos_anim.setDuration(duration)
        
        # Original position
        end_pos = widget.pos()
        
        # Bouncing requires multiple keyframes
        pos_anim.setKeyValueAt(0, QPoint(end_pos.x(), end_pos.y() + height))  # Start a bit lower
        pos_anim.setKeyValueAt(0.4, QPoint(end_pos.x(), end_pos.y() - height))  # Bounce up
        pos_anim.setKeyValueAt(0.6, QPoint(end_pos.x(), end_pos.y() + height/2))  # Bounce down smaller
        pos_anim.setKeyValueAt(0.8, QPoint(end_pos.x(), end_pos.y() - height/4))  # Bounce up smaller
        pos_anim.setKeyValueAt(1.0, end_pos)  # End at original position
        
        pos_anim.setEasingCurve(QEasingCurve.OutBounce)
        
        return pos_anim 
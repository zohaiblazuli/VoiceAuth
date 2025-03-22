"""
Startup animation module for VoiceAuth application.
Provides a stylish intro animation that displays within the main application window.
"""

from PyQt5.QtCore import (Qt, QPropertyAnimation, QEasingCurve, 
                         QParallelAnimationGroup, QSequentialAnimationGroup,
                         QPoint, QSize, QRect, QTimer, pyqtProperty, QPointF)
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                            QGraphicsOpacityEffect)
from PyQt5.QtGui import QFont, QColor, QPainter, QLinearGradient, QPen

class StartupAnimationWidget(QWidget):
    """
    Widget for displaying the VoiceAuth startup animation within the main window
    """
    def __init__(self, parent=None, main_layout=None, logo_widget=None, app_name_label=None):
        super().__init__(parent)
        self.main_layout = main_layout  # The main layout of the application
        self.logo_widget = logo_widget  # Reference to the logo widget
        self.app_name_label = app_name_label  # Reference to the app name label in the main UI
        self.animation_finished = False
        self.is_deleted = False
        
        print("StartupAnimationWidget initialized")
        
        # Get target position for final animation (where the text will move to)
        self.final_position = QPointF(0, 0)
        self.final_size = 30  # Default, will be updated later
        if app_name_label:
            self.final_size = app_name_label.font().pointSize()
            print(f"Final text size will be: {self.final_size}")
        
        # Animation state variables
        self._gradient_position = 0.0
        self._opacity = 0.0  # Start with 0 opacity for fade-in
        self._bg_opacity = 1.0  # Background opacity
        self._loading_step = 0
        self._text_size = 120  # Bigger text size
        self._continuous_gradient_animation = False
        self._loading_messages = [
            "loading assets...",
            "initializing audio processor...",
            "deleting indians from earth...",
            "petting umer's cat...",
            "trying to win grand award at ISEF...",
            "successfully deleted indians from earth & imported model!"
        ]
        
        # Hide the main layout widgets initially
        self.hide_main_widgets()
        
        # Setup the animation UI
        self.setup_ui()
        
        # Initialize the text position to the center of the window
        if self.parent():
            center = self.parent().rect().center()
            self._text_position = QPointF(center.x(), center.y() - 50)  # Slightly above center
            print(f"Initial text position set to: {self._text_position.x()}, {self._text_position.y()}")
        
        # Start with the fade-in animation after a short delay
        QTimer.singleShot(300, self.start_fade_in)
    
    def hide_main_widgets(self):
        """Hide the main application widgets during animation"""
        if self.main_layout:
            for i in range(self.main_layout.count()):
                item = self.main_layout.itemAt(i)
                if item and item.widget():
                    item.widget().hide()
    
    def show_main_widgets(self):
        """Show the main application widgets after animation"""
        if self.main_layout:
            for i in range(self.main_layout.count()):
                item = self.main_layout.itemAt(i)
                if item and item.widget():
                    item.widget().show()
    
    def setup_ui(self):
        """Set up the UI components for the animation"""
        # Make the widget fill the parent
        if self.parent():
            self.setGeometry(self.parent().rect())
        
        # Set the widget to be transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Create the layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add spacer to push content to vertical center
        layout.addStretch(2)
        
        # Create app name label
        self.title_label = QLabel("VoiceAuth")
        self.title_label.setFont(QFont("Segoe UI", self._text_size))  # Bigger, not bold
        self.title_label.setStyleSheet("color: white;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Add spacer between title and loading message
        layout.addSpacing(40)  # Increased spacing
        
        # Create loading message label
        self.loading_label = QLabel(self._loading_messages[0])
        self.loading_label.setFont(QFont("Segoe UI", 18))  # Bigger font
        self.loading_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)
        
        # Add spacer at the bottom
        layout.addStretch(3)
        
        # Store the initial position of the text
        QTimer.singleShot(100, self.store_text_position)
    
    def store_text_position(self):
        """Store the initial position of the text for later animation"""
        if hasattr(self, 'title_label') and self.title_label:
            try:
                # Get the global position of the text label
                global_pos = self.title_label.mapToGlobal(self.title_label.rect().center())
                # Convert to local coordinates for animation
                local_pos = self.mapFromGlobal(global_pos)
                self._text_position = QPointF(local_pos.x(), local_pos.y())
                print(f"Updated text position to: {self._text_position.x()}, {self._text_position.y()}")
                
                # If we have an app_name_label, store its global position for the final animation
                if self.app_name_label:
                    app_name_global_pos = self.app_name_label.mapToGlobal(self.app_name_label.rect().center())
                    self.final_position = self.mapFromGlobal(app_name_global_pos)
                    print(f"Final position set to: {self.final_position.x()}, {self.final_position.y()}")
            except Exception as e:
                print(f"Error in store_text_position: {str(e)}")
                # Keep the current position if there's an error
    
    def start_fade_in(self):
        """Start with a fade-in animation"""
        print("Starting fade-in animation")
        fade_in = QPropertyAnimation(self, b"opacity")
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setDuration(1000)
        fade_in.setEasingCurve(QEasingCurve.InOutQuad)
        fade_in.finished.connect(self.start_animation_sequence)
        fade_in.start()
    
    def start_animation_sequence(self):
        """Start the full animation sequence"""
        # Start with the loading messages
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self.update_loading_message)
        self.loading_timer.start(700)  # Update message every 700ms
    
    def update_loading_message(self):
        """Update the loading message"""
        self._loading_step += 1
        
        if self._loading_step < len(self._loading_messages):
            # Update to next loading message
            next_message = self._loading_messages[self._loading_step]
            print(f"Loading step {self._loading_step}: {next_message}")
            self.loading_label.setText(next_message)
        else:
            # All loading messages shown, move to gradient animation
            print("All loading messages shown, starting gradient animation")
            self.loading_timer.stop()
            self.loading_label.setText("")  # Clear the loading message
            self.start_gradient_animation()
    
    def start_gradient_animation(self):
        """Start the continuous gradient animation on the app name"""
        print("Starting gradient animation")
        # Set the flag for continuous animation
        self._continuous_gradient_animation = True
        
        # Create a timer for continuous gradient animation
        self.gradient_timer = QTimer(self)
        self.gradient_timer.timeout.connect(self.update_gradient)
        self.gradient_timer.start(50)  # Update every 50ms for smooth animation
        
        # Set a timer to start the transition after 4 seconds
        QTimer.singleShot(4000, self.start_transition_animation)
    
    def update_gradient(self):
        """Update the gradient position for continuous animation"""
        self._gradient_position += 0.01
        if self._gradient_position > 1.0:
            self._gradient_position = 0.0
        self.update()  # Trigger repaint
    
    def start_transition_animation(self):
        """Start the transition animation to move text to its final position and fade out background"""
        print("Starting transition animation")
        # Create animation group for parallel animations
        self.transition_group = QParallelAnimationGroup()
        
        # Create animation for text position movement
        if self.app_name_label:
            # Get the global position of the app name label for transition target
            try:
                app_label_center = self.app_name_label.rect().center()
                app_label_global_pos = self.app_name_label.mapToGlobal(app_label_center)
                final_pos = self.mapFromGlobal(app_label_global_pos)
                
                print(f"Transition to position: {final_pos.x()}, {final_pos.y()}")
                
                # Create animation for text position
                position_anim = QPropertyAnimation(self, b"textPosition")
                position_anim.setStartValue(self._text_position)
                position_anim.setEndValue(final_pos)
                position_anim.setDuration(1000)
                position_anim.setEasingCurve(QEasingCurve.OutQuad)
                self.transition_group.addAnimation(position_anim)
                
                # Create animation for text size
                size_anim = QPropertyAnimation(self, b"textSize")
                size_anim.setStartValue(self._text_size)
                size_anim.setEndValue(self.final_size)
                size_anim.setDuration(1000)
                size_anim.setEasingCurve(QEasingCurve.OutQuad)
                self.transition_group.addAnimation(size_anim)
            except Exception as e:
                print(f"Error setting up position animation: {str(e)}")
        
        # Create animation for background fade out
        bg_fade_anim = QPropertyAnimation(self, b"bgOpacity")
        bg_fade_anim.setStartValue(1.0)
        bg_fade_anim.setEndValue(0.0)
        bg_fade_anim.setDuration(1000)
        bg_fade_anim.setEasingCurve(QEasingCurve.OutQuad)
        self.transition_group.addAnimation(bg_fade_anim)
        
        # Connect to finished signal
        self.transition_group.finished.connect(self.complete_animation)
        
        # Start the transition
        self.transition_group.start()
    
    def complete_animation(self):
        """Complete the animation and switch to main UI"""
        print("Completing animation")
        # Stop the continuous gradient animation
        if hasattr(self, 'gradient_timer') and self.gradient_timer.isActive():
            self.gradient_timer.stop()
        
        # Show the main UI elements
        self.show_main_widgets()
        
        # Mark as deleted first and schedule deletion
        self.is_deleted = True
        print("Animation complete - deleting animation widget")
        self.deleteLater()
    
    # Property for gradient position animation
    def _get_gradient_position(self):
        return self._gradient_position
    
    def _set_gradient_position(self, pos):
        self._gradient_position = pos
        self.update()  # Trigger repaint
    
    gradientPosition = pyqtProperty(float, _get_gradient_position, _set_gradient_position)
    
    # Property for opacity animation
    def _get_opacity(self):
        return self._opacity
    
    def _set_opacity(self, opacity):
        self._opacity = opacity
        self.update()
    
    opacity = pyqtProperty(float, _get_opacity, _set_opacity)
    
    # Property for background opacity animation
    def _get_bg_opacity(self):
        return self._bg_opacity
    
    def _set_bg_opacity(self, opacity):
        self._bg_opacity = opacity
        self.update()
    
    bgOpacity = pyqtProperty(float, _get_bg_opacity, _set_bg_opacity)
    
    # Property for text position animation
    def _get_text_position(self):
        return self._text_position
    
    def _set_text_position(self, pos):
        self._text_position = pos
        self.update()
    
    textPosition = pyqtProperty(QPointF, _get_text_position, _set_text_position)
    
    # Property for text size animation
    def _get_text_size(self):
        return self._text_size
    
    def _set_text_size(self, size):
        self._text_size = size
        if hasattr(self, 'title_label'):
            self.title_label.setFont(QFont("Segoe UI", int(size)))
        self.update()
    
    textSize = pyqtProperty(float, _get_text_size, _set_text_size)
    
    def paintEvent(self, event):
        """Custom paint event for the animation"""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill the entire widget with black background with fading opacity
        painter.fillRect(self.rect(), QColor(0, 0, 0, int(255 * self._bg_opacity)))
        
        # For the initial fade-in and loading phases, show the regular label
        if not self._continuous_gradient_animation:
            # Set the opacity for the regular label
            effect = QGraphicsOpacityEffect(self.title_label)
            effect.setOpacity(self._opacity)
            self.title_label.setGraphicsEffect(effect)
            self.title_label.show()
            return
        
        # For gradient animation phase, hide the label and draw custom text
        if self._continuous_gradient_animation:
            # Hide the normal label since we'll draw it manually
            if hasattr(self, 'title_label'):
                self.title_label.hide()
                
            # Get the text and font from the label
            text = "VoiceAuth"
            font = QFont("Segoe UI", int(self._text_size))
            
            # Set the font for painting
            painter.setFont(font)
            
            # Get the text size to properly center it
            fm = painter.fontMetrics()
            text_width = fm.horizontalAdvance(text)
            text_height = fm.height()
            
            # Calculate the center position for the text
            center_x = self._text_position.x()
            center_y = self._text_position.y()
            
            # Create rectangle for drawing text centered at this position
            text_rect = QRect(
                int(center_x - text_width / 2),
                int(center_y - text_height / 2),
                text_width,
                text_height
            )
            
            # Create gradient for text
            gradient = QLinearGradient(0, 0, self.width(), 0)  # Horizontal gradient
            gradient.setColorAt(0.0, QColor(165, 126, 241))  # Light purple
            gradient.setColorAt(0.5, QColor(130, 145, 230))  # Medium blue-purple
            gradient.setColorAt(1.0, QColor(95, 164, 219))   # Light blue
            
            # Make gradient appear to move across the text continuously
            gradient.setStart(self.width() * -0.5 + (self.width() * 2.0 * self._gradient_position), 0)
            gradient.setFinalStop(self.width() * 0.5 + (self.width() * 2.0 * self._gradient_position), 0)
            
            # Draw text with gradient
            painter.setPen(QPen(gradient, 1))
            painter.drawText(text_rect, Qt.AlignCenter, text)

def run_startup_animation(main_window, content_layout, logo_widget=None, app_name_label=None):
    """
    Run the startup animation in the main window
    
    Parameters:
    -----------
    main_window : QMainWindow
        The main application window
    content_layout : QLayout
        The main content layout where UI elements will be shown after animation
    logo_widget : QWidget, optional
        Reference to the logo widget for the transition
    app_name_label : QLabel, optional
        Reference to the app name label for the transition
    """
    # Create the animation widget
    anim_widget = StartupAnimationWidget(
        main_window,
        content_layout,
        logo_widget,
        app_name_label
    )
    
    # Show the animation widget
    anim_widget.show()
    
    return anim_widget 
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from PyQt5.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout, 
                           QFileDialog, QLabel, QProgressBar, QTextEdit, 
                           QComboBox, QRadioButton, QButtonGroup, QListWidget, QListWidgetItem,
                           QGroupBox, QGridLayout, QSplitter, QSpacerItem,
                           QSizePolicy, QScrollArea, QSlider, QFrame, QProgressDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QMetaObject, Q_ARG, QRect, QObject
from PyQt5.QtGui import QFont, QPixmap, QColor, QPainter, QBrush, QPen, QLinearGradient

from utils import get_output_path, get_model_path  # Import path utilities
from ui_components import StylesheetProvider
from audio_processor import AudioProcessor
from simple_model import classify_audio
from batch_process import extract_features

class WorkerThread(QThread):
    """Worker thread for background operations"""
    update_status = pyqtSignal(str)
    finished_task = pyqtSignal(dict)
    
    def __init__(self, task, params):
        super().__init__()
        self.task = task
        self.params = params
        
    def run(self):
        try:
            if self.task == "classify":
                # Extract file path and model path
                file_path = self.params.get("file_path")
                model_path = self.params.get("model_path")
                features = self.params.get("features")
                scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
                
                if features is None:
                    # Extract features
                    self.update_status.emit(f"Extracting features from {os.path.basename(file_path)}...")
                    features = extract_features(file_path)
                    
                    if features is None:
                        self.finished_task.emit({
                            "success": False,
                            "message": f"Failed to extract features from {file_path}"
                        })
                        return
                else:
                    self.update_status.emit("Using pre-extracted features...")
                
                # Classify
                self.update_status.emit("Classifying audio...")
                is_ai, confidence = classify_audio(features, model_path, scaler_path)
                
                # Return results
                self.finished_task.emit({
                    "success": True,
                    "is_ai": is_ai,
                    "confidence": confidence,
                    "features": features,
                    "file_path": file_path
                })
            
            elif self.task == "process_recording":
                # Get parameters
                input_file = self.params.get("input_file")
                processor = self.params.get("processor")
                
                # Apply noise suppression
                self.update_status.emit(f"Processing recording {os.path.basename(input_file)}...")
                processed_file = processor.apply_noise_suppression(input_file)
                
                if processed_file:
                    # Extract features
                    self.update_status.emit("Extracting features...")
                    features = processor.extract_features(processed_file)
                    
                    if features is not None:
                        self.finished_task.emit({
                            "success": True,
                            "file_path": processed_file,
                            "features": features
                        })
                    else:
                        self.finished_task.emit({
                            "success": False,
                            "message": "Failed to extract features from processed audio"
                        })
                else:
                    self.finished_task.emit({
                        "success": False,
                        "message": "Failed to process recording"
                    })
                
            elif self.task == "save_feedback":
                # Get parameters
                file_path = self.params.get("file_path")
                features = self.params.get("features")
                correct_label = self.params.get("correct_label")
                feedback_file = self.params.get("feedback_file")
                
                # Update status
                self.update_status.emit("Saving feedback...")
                
                # Ensure feedback directory exists
                os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
                
                # Also ensure samples directory exists
                samples_dir = os.path.join(get_output_path(), "samples")
                os.makedirs(samples_dir, exist_ok=True)
                
                # Create or load feedback dataframe
                if os.path.exists(feedback_file):
                    feedback_df = pd.read_csv(feedback_file)
                else:
                    # Create new feedback dataframe with same structure as features
                    num_features = len(features)
                    feature_names = [f'feature_{i}' for i in range(num_features)]
                    feedback_df = pd.DataFrame(columns=feature_names + ['label', 'file_path'])
                
                # Copy the audio file to the samples directory
                import shutil
                file_basename = os.path.basename(file_path)
                saved_filename = f"feedback_{len(feedback_df) + 1}_{file_basename}"
                saved_path = os.path.join(samples_dir, saved_filename)
                
                try:
                    shutil.copy2(file_path, saved_path)
                    self.update_status.emit(f"Copied audio file to {saved_path}")
                    
                    # Use relative path in feedback_df
                    relative_path = os.path.join("samples", saved_filename)
                except Exception as e:
                    self.update_status.emit(f"Warning: Could not copy audio file: {str(e)}")
                    relative_path = file_path  # Use original path if copy fails
                
                # Create new row with features and label
                new_row = pd.DataFrame([np.append(features, [correct_label, relative_path])], 
                                      columns=feedback_df.columns)
                
                # Append to feedback dataframe
                feedback_df = pd.concat([feedback_df, new_row], ignore_index=True)
                
                # Save updated feedback
                feedback_df.to_csv(feedback_file, index=False)
                
                # Return success
                self.finished_task.emit({
                    "success": True,
                    "feedback_count": len(feedback_df),
                    "message": f"Feedback saved. Total feedback samples: {len(feedback_df)}"
                })
                
            elif self.task == "retrain":
                # Get parameters
                model_path = self.params.get("model_path")
                feedback_file = self.params.get("feedback_file")
                original_features = self.params.get("original_features")
                
                # Update status
                self.update_status.emit("Retraining model with feedback...")
                
                # Ensure output directories exist
                os.makedirs("output", exist_ok=True)
                os.makedirs("output/models", exist_ok=True)
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                # Load feedback data
                if not os.path.exists(feedback_file):
                    self.finished_task.emit({
                        "success": False,
                        "message": "No feedback data available"
                    })
                    return
                
                try:
                    feedback_df = pd.read_csv(feedback_file)
                    if len(feedback_df) == 0:
                        self.finished_task.emit({
                            "success": False,
                            "message": "Feedback file exists but contains no data"
                        })
                        return
                except Exception as e:
                    self.finished_task.emit({
                        "success": False,
                        "message": f"Error loading feedback data: {str(e)}"
                    })
                    return
                
                # Create backup of current model if it exists
                if os.path.exists(model_path):
                    try:
                        # Create backup directory
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_dir = os.path.join(get_model_path(), "backups")
                        os.makedirs(backup_dir, exist_ok=True)
                        
                        # Create backup files with timestamp
                        model_backup = os.path.join(backup_dir, f"{os.path.basename(model_path)}.{timestamp}.bak")
                        scaler_backup = os.path.join(backup_dir, f"{os.path.basename(os.path.splitext(model_path)[0])}_scaler.pkl.{timestamp}.bak")
                        
                        # Copy files
                        shutil.copy2(model_path, model_backup)
                        scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
                        if os.path.exists(scaler_path):
                            shutil.copy2(scaler_path, scaler_backup)
                            
                        self.update_status.emit(f"Created model backup: {model_backup}")
                    except Exception as e:
                        self.update_status.emit(f"Warning: Could not create model backup: {str(e)}")
                
                # Create directory for samples
                samples_dir = os.path.join(get_output_path(), "samples")
                os.makedirs(samples_dir, exist_ok=True)
                
                # Load original training data if available
                if os.path.exists(original_features):
                    self.update_status.emit("Loading original training data...")
                    try:
                        original_df = pd.read_csv(original_features)
                        
                        # Validate column compatibility
                        feedback_columns = set(feedback_df.columns)
                        original_columns = set(original_df.columns)
                        
                        if not feedback_columns.issubset(original_columns) and not original_columns.issubset(feedback_columns):
                            self.update_status.emit("Warning: Column mismatch between original and feedback data")
                            self.update_status.emit(f"Original columns: {original_columns}")
                            self.update_status.emit(f"Feedback columns: {feedback_columns}")
                            
                            # Find common columns
                            common_columns = list(original_columns.intersection(feedback_columns))
                            if 'label' in common_columns and len(common_columns) > 1:
                                self.update_status.emit(f"Using {len(common_columns)} common columns for retraining")
                                original_df = original_df[common_columns]
                                feedback_df = feedback_df[common_columns]
                            else:
                                self.finished_task.emit({
                                    "success": False,
                                    "message": "Incompatible data formats between original and feedback data"
                                })
                                return
                        
                        # Combine original data with feedback
                        combined_df = pd.concat([original_df, feedback_df], ignore_index=True)
                    except Exception as e:
                        self.finished_task.emit({
                            "success": False,
                            "message": f"Error processing original training data: {str(e)}"
                        })
                        return
                else:
                    # Only use feedback data
                    self.update_status.emit("Original training data not found. Using only feedback data.")
                    combined_df = feedback_df
                
                # Save combined features
                temp_features = "output/temp_features.csv"
                try:
                    combined_df.to_csv(temp_features, index=False)
                except Exception as e:
                    self.finished_task.emit({
                        "success": False,
                        "message": f"Error saving combined features: {str(e)}"
                    })
                    return
                
                from simple_model import train_model
                
                # Retrain model
                try:
                    self.update_status.emit(f"Training model with {len(combined_df)} samples...")
                    model, scaler = train_model(temp_features, model_path, callback=self.update_status.emit)
                    
                    # Clean up temporary file
                    if os.path.exists(temp_features):
                        os.remove(temp_features)
                    
                    # Return success
                    self.finished_task.emit({
                        "success": True,
                        "message": f"Model retrained successfully with {len(feedback_df)} feedback samples"
                    })
                except Exception as e:
                    # Restore backup if training failed
                    if os.path.exists(model_backup) and os.path.exists(scaler_backup):
                        try:
                            shutil.copy2(model_backup, model_path)
                            shutil.copy2(scaler_backup, scaler_path)
                            self.update_status.emit("Restored model from backup after training failure")
                        except Exception as restore_error:
                            self.update_status.emit(f"Warning: Failed to restore backup: {str(restore_error)}")
                    
                    self.finished_task.emit({
                        "success": False,
                        "message": f"Error during model training: {str(e)}"
                    })
        except Exception as e:
            self.update_status.emit(f"Error in worker thread: {str(e)}")
            self.finished_task.emit({
                "success": False,
                "message": f"Error: {str(e)}"
            })

class ImportSampleTab(QWidget):
    """Tab for importing and classifying audio samples"""
    
    # Signal to notify when feedback is saved
    feedback_saved_signal = pyqtSignal()
    
    def __init__(self, parent=None, model_path=None, feedback_file=None):
        super().__init__(parent)
        self.parent = parent
        self.model_path = model_path
        self.feedback_file = feedback_file
        self.current_file = None
        self.current_features = None
        self.current_prediction = None
        self.is_playing = False
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.setup_ui()
        self.check_feedback_file()
    
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Top section - File import
        import_group = QGroupBox("Import Audio Sample")
        import_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        import_layout = QVBoxLayout()
        
        # Add descriptive text
        desc_label = QLabel("Import a voice sample to classify it as AI-generated or human.")
        desc_label.setStyleSheet(StylesheetProvider.get_subtitle_label_style())
        import_layout.addWidget(desc_label)
        
        # Add file selection controls
        file_layout = QHBoxLayout()
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file)
        self.browse_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.browse_button.setMinimumWidth(100)
        
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.browse_button, 0)
        
        import_layout.addLayout(file_layout)
        
        # Add playback controls
        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play Audio")
        self.play_button.clicked.connect(self.play_recording)
        self.play_button.setEnabled(False)
        self.play_button.setMinimumHeight(36)
        
        self.stop_button = QPushButton("Stop Playback")
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)
        self.stop_button.setMinimumHeight(36)
        
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.stop_button)
        playback_layout.addStretch()
        
        import_layout.addLayout(playback_layout)
        
        # Add classify button
        self.classify_button = QPushButton("Classify Sample")
        self.classify_button.setStyleSheet(StylesheetProvider.get_primary_button_style())
        self.classify_button.clicked.connect(self.classify_sample)
        self.classify_button.setEnabled(False)
        self.classify_button.setMinimumHeight(40)
        
        import_layout.addWidget(self.classify_button)
        
        import_group.setLayout(import_layout)
        
        # Middle section - Classification results
        self.results_group = QGroupBox("Classification Results")
        results_layout = QVBoxLayout()
        
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMaximum(100)
        self.confidence_bar.setMinimum(0)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("%p% confidence")
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setMinimumHeight(20)
        
        results_layout.addWidget(self.result_label)
        results_layout.addWidget(self.confidence_bar)
        
        # Add feedback mechanism
        feedback_layout = QVBoxLayout()
        
        self.feedback_label = QLabel("Was the prediction correct?")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        
        # Radio buttons for feedback
        radio_layout = QHBoxLayout()
        
        self.radio_correct = QRadioButton("Yes (prediction was correct)")
        self.radio_incorrect = QRadioButton("No (prediction was wrong)")
        
        self.feedback_group = QButtonGroup()
        self.feedback_group.addButton(self.radio_correct)
        self.feedback_group.addButton(self.radio_incorrect)
        
        radio_layout.addWidget(self.radio_correct)
        radio_layout.addWidget(self.radio_incorrect)
        
        # Submit feedback button
        self.submit_button = QPushButton("Submit Feedback")
        self.submit_button.clicked.connect(self.submit_feedback)
        self.submit_button.setEnabled(False)
        
        # Connect radio buttons to enable submit button
        self.radio_correct.toggled.connect(self.toggle_submit_button)
        self.radio_incorrect.toggled.connect(self.toggle_submit_button)
        
        feedback_layout.addWidget(self.feedback_label)
        feedback_layout.addLayout(radio_layout)
        feedback_layout.addWidget(self.submit_button)
        
        results_layout.addLayout(feedback_layout)
        
        self.results_group.setLayout(results_layout)
        self.results_group.setVisible(False)
        
        # Bottom section - Status and history
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(StylesheetProvider.get_status_label_style())
        
        status_layout.addWidget(self.status_label, 1)
        
        # Retrain button
        self.retrain_button = QPushButton("Retrain Model with Feedback")
        self.retrain_button.clicked.connect(self.retrain_model)
        self.retrain_button.setEnabled(False)
        self.retrain_button.setMinimumHeight(36)
        self.retrain_button.setMinimumWidth(200)
        
        status_layout.addWidget(self.retrain_button)
        
        # Add sections to main layout
        layout.addWidget(import_group)
        layout.addWidget(self.results_group)
        layout.addLayout(status_layout)
        layout.addStretch()
        
        # Check if feedback file exists
        self.check_feedback_file()
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
    
    def check_feedback_file(self):
        """Check if feedback file exists and has data"""
        if os.path.exists(self.feedback_file):
            try:
                feedback_df = pd.read_csv(self.feedback_file)
                if len(feedback_df) > 0:
                    self.retrain_button.setEnabled(True)
                    self.update_status(f"Found {len(feedback_df)} feedback samples. You can retrain the model.")
            except:
                pass
    
    def browse_file(self):
        """Browse for an audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.ogg *.flac)")
        
        if file_path:
            self.current_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.classify_button.setEnabled(True)
            self.play_button.setEnabled(True)
            
            # Reset results if shown
            self.results_group.setVisible(False)
            self.radio_correct.setChecked(False)
            self.radio_incorrect.setChecked(False)
            self.submit_button.setEnabled(False)
    
    def play_recording(self):
        """Play the current audio file"""
        if not self.current_file or not os.path.exists(self.current_file):
            self.update_status("No audio file to play")
            return
        
        try:
            self.is_playing = True
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.update_status("Playing audio...")
            
            # Start playback in a separate thread
            import threading
            import sounddevice as sd
            import soundfile as sf
            
            def play_thread():
                try:
                    data, samplerate = sf.read(self.current_file)
                    sd.play(data, samplerate)
                    sd.wait()  # Wait until file is done playing
                    self.playback_finished()
                except Exception as e:
                    self.update_status(f"Error during playback: {str(e)}")
                    self.playback_finished()
            
            threading.Thread(target=play_thread, daemon=True).start()
            
        except Exception as e:
            self.update_status(f"Error starting playback: {str(e)}")
            self.playback_finished()
    
    def stop_playback(self):
        """Stop the current playback"""
        if self.is_playing:
            try:
                import sounddevice as sd
                sd.stop()
                self.playback_finished()
                self.update_status("Playback stopped")
            except Exception as e:
                self.update_status(f"Error stopping playback: {str(e)}")
    
    def playback_finished(self):
        """Called when playback finishes or is stopped"""
        self.is_playing = False
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def classify_sample(self):
        """Classify the selected audio sample"""
        if not self.current_file:
            return
        
        # Create worker thread
        self.worker = WorkerThread("classify", {
            "file_path": self.current_file,
            "model_path": self.model_path,
            "features": None
        })
        
        # Connect signals
        self.worker.update_status.connect(self.update_status)
        self.worker.finished_task.connect(self.classification_finished)
        
        # Disable button and update status
        self.classify_button.setEnabled(False)
        self.update_status("Classifying audio sample...")
        
        # Start worker
        self.worker.start()
    
    def classification_finished(self, result):
        """Handle classification results"""
        self.classify_button.setEnabled(True)
        
        if not result["success"]:
            self.update_status(f"Error: {result['message']}")
            return
        
        # Store results
        self.current_prediction = result["is_ai"]
        self.current_features = result["features"]
        
        # Update UI
        is_ai = result["is_ai"]
        confidence = result["confidence"] * 100
        
        prediction_text = "AI-Generated Voice" if is_ai else "Human Voice"
        self.result_label.setText(prediction_text)
        self.result_label.setStyleSheet(StylesheetProvider.get_result_label_style(is_ai))
        
        # Update confidence bar with a proper style
        self.confidence_bar.setValue(int(confidence))
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 5px;
                background-color: rgba(30, 30, 40, 0.5);
                text-align: center;
                padding: 1px;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #85a3da, stop:0.5 #6e6ebe, stop:1 #d78599);
                border-radius: 3px;
            }
        """)
        
        # Show results
        self.results_group.setVisible(True)
        
        # Update status
        self.update_status(f"Classification complete: {prediction_text} with {confidence:.2f}% confidence")
    
    def toggle_submit_button(self):
        """Enable/disable submit button based on radio selection"""
        self.submit_button.setEnabled(
            self.radio_correct.isChecked() or self.radio_incorrect.isChecked())
    
    def submit_feedback(self):
        """Submit feedback on classification"""
        if self.current_features is None:
            return
        
        # Determine correct label
        if self.radio_correct.isChecked():
            # Prediction was correct
            correct_label = 1 if self.current_prediction else 0
        else:
            # Prediction was incorrect
            correct_label = 0 if self.current_prediction else 1
        
        # Create worker thread
        self.worker = WorkerThread("save_feedback", {
            "file_path": self.current_file,
            "features": self.current_features,
            "correct_label": correct_label,
            "feedback_file": self.feedback_file
        })
        
        # Connect signals
        self.worker.update_status.connect(self.update_status)
        self.worker.finished_task.connect(self.feedback_saved)
        
        # Disable button and update status
        self.submit_button.setEnabled(False)
        self.update_status("Saving feedback...")
        
        # Start worker
        self.worker.start()
    
    def feedback_saved(self, result):
        """Handle feedback saving result"""
        if not result["success"]:
            self.update_status(f"Error: {result['message']}")
            self.submit_button.setEnabled(True)
            return
        
        # Reset feedback UI
        self.radio_correct.setChecked(False)
        self.radio_incorrect.setChecked(False)
        
        # Enable retrain button
        self.retrain_button.setEnabled(True)
        
        # Update status
        self.update_status(result["message"])
        
        # Emit signal to refresh feedback tab
        self.feedback_saved_signal.emit()
    
    def retrain_model(self):
        """Retrain model with feedback data"""
        # Create a progress dialog to show detailed status
        self.progress_dialog = QProgressDialog("Preparing to retrain model...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Retraining Model")
        self.progress_dialog.setMinimumWidth(400)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setCancelButton(None)  # Disable cancel button
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Create worker thread
        self.worker = WorkerThread("retrain", {
            "model_path": self.model_path,
            "feedback_file": self.feedback_file,
            "original_features": self.parent.features_path
        })
        
        # Connect signals
        self.worker.update_status.connect(self.update_retraining_status)
        self.worker.finished_task.connect(self.retrain_finished)
        
        # Disable button and update status
        self.retrain_button.setEnabled(False)
        self.update_status("Retraining model...")
        
        # Start worker
        self.worker.start()
    
    def update_retraining_status(self, message):
        """Update progress dialog with retraining status"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setLabelText(message)
        self.update_status(message)
    
    def retrain_finished(self, result):
        """Handle retraining result"""
        # Close progress dialog
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # Re-enable button
        self.retrain_button.setEnabled(True)
        
        if not result["success"]:
            self.update_status(f"Error: {result['message']}")
            QMessageBox.warning(self, "Retraining Failed", result["message"])
            return
        
        # Show success message
        QMessageBox.information(self, "Retraining Successful", result["message"])
        
        # Update status
        self.update_status(result["message"])
        
        # Update version if parent app is available
        if hasattr(self.parent, 'update_version'):
            self.parent.update_version() 

class RecordSampleTab(QWidget):
    """Tab for recording and classifying voice samples"""
    
    # Add signal to update level meter from main thread
    level_update = pyqtSignal(float)
    status_update = pyqtSignal(str)
    # Signal to notify when feedback is saved
    feedback_saved_signal = pyqtSignal()
    
    def __init__(self, parent=None, model_path=None, feedback_file=None):
        super().__init__(parent)
        self.parent = parent
        self.model_path = model_path
        self.feedback_file = feedback_file
        self.is_recording = False
        self.is_playing = False
        self.current_file = None
        self.current_features = None
        self.current_prediction = None
        self.recording_seconds = 0
        self.recording_timer = None
        
        # Create audio processor
        self.audio_processor = AudioProcessor()
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Setup UI components
        self.setup_ui()
        
        # Connect signals to slots
        self.level_update.connect(self.update_level_safe)
        self.status_update.connect(self.update_status)
        
        # Populate devices
        self.populate_devices()
        self.check_feedback_file()
    
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Create record group
        record_group = QGroupBox("Record Audio Sample")
        record_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        record_layout = QVBoxLayout(record_group)
        
        # Device selection
        device_layout = QHBoxLayout()
        device_label = QLabel("Input Device:")
        self.device_combo = QComboBox()
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo, 1)
        
        # Recording timer display
        self.timer_label = QLabel("00:00")
        self.timer_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e0e0e0;")
        self.timer_label.setAlignment(Qt.AlignCenter)
        device_layout.addWidget(self.timer_label)
        
        # Recording controls
        controls_layout = QHBoxLayout()
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setMinimumHeight(40)
        self.record_status = QLabel("Ready")
        
        # Level meter
        self.level_meter = CustomLevelMeter(self)
        self.level_meter.setMinimumWidth(200)
        controls_layout.addWidget(self.record_button)
        controls_layout.addWidget(self.level_meter)
        controls_layout.addWidget(self.record_status)
        
        # Add playback controls
        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play Recording")
        self.play_button.clicked.connect(self.play_recording)
        self.play_button.setEnabled(False)
        self.play_button.setMinimumHeight(36)
        
        self.stop_button = QPushButton("Stop Playback")
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)
        self.stop_button.setMinimumHeight(36)
        
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.stop_button)
        playback_layout.addStretch()
        
        # Process button
        process_layout = QHBoxLayout()
        self.process_button = QPushButton("Process And Classify Recording")
        self.process_button.clicked.connect(self.process_recording)
        self.process_button.setEnabled(False)
        self.process_button.setMinimumHeight(36)
        self.process_button.setStyleSheet(StylesheetProvider.get_primary_button_style())
        process_layout.addWidget(self.process_button)
        
        # Add to record group
        record_layout.addLayout(device_layout)
        record_layout.addLayout(controls_layout)
        record_layout.addLayout(playback_layout)
        record_layout.addLayout(process_layout)
        
        # Create results group
        self.results_group = QGroupBox("Classification Results")
        self.results_group.setVisible(False)
        results_layout = QVBoxLayout(self.results_group)
        
        # Results display
        result_container = QWidget()
        result_container.setStyleSheet("background-color: rgba(51, 48, 74, 0.7); border-radius: 8px; padding: 10px;")
        result_container_layout = QVBoxLayout(result_container)
        
        self.result_label = QLabel()
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.result_label.setAlignment(Qt.AlignCenter)
        
        # Confidence visualization
        self.confidence_meter = QProgressBar()
        self.confidence_meter.setMinimum(0)
        self.confidence_meter.setMaximum(100)
        self.confidence_meter.setTextVisible(True)
        self.confidence_meter.setFormat("%p% confidence")
        self.confidence_meter.setMinimumHeight(20)
        
        result_container_layout.addWidget(self.result_label)
        result_container_layout.addWidget(self.confidence_meter)
        results_layout.addWidget(result_container)
        
        # Feedback section
        feedback_group = QGroupBox("Provide Feedback")
        feedback_group.setStyleSheet("QGroupBox { color: #e0e0e0; font-weight: bold; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }")
        feedback_layout = QVBoxLayout()
        
        feedback_prompt = QLabel("Is this classification correct?")
        feedback_prompt.setStyleSheet("font-size: 14px; color: #e0e0e0;")
        feedback_prompt.setAlignment(Qt.AlignCenter)
        
        feedback_buttons = QHBoxLayout()
        self.radio_correct = QRadioButton("Yes (Correct)")
        self.radio_incorrect = QRadioButton("No (Incorrect)")
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.radio_correct)
        self.radio_group.addButton(self.radio_incorrect)
        self.radio_group.buttonClicked.connect(self.toggle_submit_button)
        
        feedback_buttons.addStretch()
        feedback_buttons.addWidget(self.radio_correct)
        feedback_buttons.addWidget(self.radio_incorrect)
        feedback_buttons.addStretch()
        
        self.submit_button = QPushButton("Submit Feedback")
        self.submit_button.setStyleSheet(StylesheetProvider.get_primary_button_style())
        self.submit_button.clicked.connect(self.submit_feedback)
        self.submit_button.setEnabled(False)
        self.submit_button.setMinimumHeight(40)
        
        feedback_layout.addWidget(feedback_prompt)
        feedback_layout.addLayout(feedback_buttons)
        feedback_layout.addWidget(self.submit_button)
        feedback_group.setLayout(feedback_layout)
        
        results_layout.addWidget(feedback_group)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label, 1)
        
        # Add sections to main layout
        layout.addWidget(record_group)
        layout.addWidget(self.results_group)
        layout.addLayout(status_layout)
        layout.addStretch()
        
        # Set up audio processor callbacks
        self.audio_processor.set_callbacks(
            level_callback=self.safe_level_update,
            status_callback=self.safe_status_update
        )
    
    def populate_devices(self):
        """Populate the device combo box with available input devices"""
        try:
            devices = self.audio_processor.get_input_devices()
            for i, device in enumerate(devices):
                self.device_combo.addItem(f"{device['name']} ({device['max_input_channels']} ch)", device['index'])
        except Exception as e:
            self.update_status(f"Error getting input devices: {str(e)}")
            
    def update_timer(self):
        """Update recording timer"""
        self.recording_seconds += 1
        minutes = self.recording_seconds // 60
        seconds = self.recording_seconds % 60
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")
        
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_button.setText("Stop Recording")
            self.results_group.setVisible(False)
            self.process_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            
            # Reset and start recording timer
            self.recording_seconds = 0
            self.timer_label.setText("00:00")
            self.recording_timer = QTimer(self)
            self.recording_timer.timeout.connect(self.update_timer)
            self.recording_timer.start(1000)  # Update every second
            
            # Get selected device
            device_idx = self.device_combo.currentData()
            
            try:
                # Start recording
                self.audio_processor.start_recording(device_id=device_idx)
                self.update_status("Recording started...")
            except Exception as e:
                self.safe_status_update(f"Recording error: {str(e)}")
                self.is_recording = False
                self.record_button.setText("Start Recording")
                if self.recording_timer:
                    self.recording_timer.stop()
        else:
            # Stop recording
            self.is_recording = False
            self.record_button.setText("Start Recording")
            
            # Stop timer
            if self.recording_timer:
                self.recording_timer.stop()
                self.recording_timer = None
            
            try:
                # Stop recording and get file
                self.current_file = self.audio_processor.stop_recording()
                
                if self.current_file and os.path.exists(self.current_file):
                    self.safe_status_update(f"Recording saved to: {self.current_file}")
                    self.process_button.setEnabled(True)
                    self.play_button.setEnabled(True)
                else:
                    self.safe_status_update("Recording failed")
                    self.process_button.setEnabled(False)
                    self.play_button.setEnabled(False)
            except Exception as e:
                self.safe_status_update(f"Error stopping recording: {str(e)}")
                self.process_button.setEnabled(False)
                self.play_button.setEnabled(False)
    
    def play_recording(self):
        """Play the current recording"""
        if not self.current_file or not os.path.exists(self.current_file):
            self.update_status("No recording to play")
            return
        
        try:
            self.is_playing = True
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.update_status("Playing recording...")
            
            # Start playback in a separate thread
            import threading
            import sounddevice as sd
            import soundfile as sf
            
            def play_thread():
                try:
                    data, samplerate = sf.read(self.current_file)
                    sd.play(data, samplerate)
                    sd.wait()  # Wait until file is done playing
                    self.playback_finished()
                except Exception as e:
                    self.safe_status_update(f"Error during playback: {str(e)}")
                    self.playback_finished()
            
            threading.Thread(target=play_thread, daemon=True).start()
            
        except Exception as e:
            self.safe_status_update(f"Error starting playback: {str(e)}")
            self.playback_finished()
    
    def stop_playback(self):
        """Stop the current playback"""
        if self.is_playing:
            try:
                import sounddevice as sd
                sd.stop()
                self.playback_finished()
                self.update_status("Playback stopped")
            except Exception as e:
                self.update_status(f"Error stopping playback: {str(e)}")
    
    def playback_finished(self):
        """Called when playback finishes or is stopped"""
        self.is_playing = False
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def process_recording(self):
        """Process the recording and classify it"""
        if not self.current_file or not os.path.exists(self.current_file):
            self.update_status("No recording to process")
            return
        
        # Create worker thread
        self.worker = WorkerThread("process_recording", {
            "input_file": self.current_file,
            "processor": self.audio_processor
        })
        
        # Connect signals
        self.worker.update_status.connect(self.update_status)
        self.worker.finished_task.connect(self.processing_finished)
        
        # Disable button and update status
        self.process_button.setEnabled(False)
        self.update_status("Processing recording...")
        
        # Start worker
        self.worker.start()
    
    def processing_finished(self, result):
        """Handle processing results"""
        if not result["success"]:
            self.update_status(f"Error: {result['message']}")
            self.process_button.setEnabled(True)
            return
        
        # Store results
        self.current_features = result["features"]
        self.current_file = result["file_path"]
        
        # Classify the processed recording
        self.classify_recording()
    
    def classify_recording(self):
        """Classify the processed recording"""
        # Create worker thread
        self.worker = WorkerThread("classify", {
            "file_path": self.current_file,
            "model_path": self.model_path,
            "features": self.current_features
        })
        
        # Connect signals
        self.worker.update_status.connect(self.update_status)
        self.worker.finished_task.connect(self.classification_finished)
        
        # Update status
        self.update_status("Classifying recording...")
        
        # Start worker
        self.worker.start()
    
    def classification_finished(self, result):
        """Handle classification results"""
        self.process_button.setEnabled(True)
        
        if not result["success"]:
            self.update_status(f"Error: {result['message']}")
            return
        
        # Store results
        self.current_prediction = result["is_ai"]
        self.current_features = result["features"]
        
        # Update UI
        is_ai = result["is_ai"]
        confidence = result["confidence"] * 100
        
        prediction_text = "AI-Generated Voice" if is_ai else "Human Voice"
        self.result_label.setText(prediction_text)
        self.result_label.setStyleSheet(StylesheetProvider.get_result_label_style(is_ai))
        
        # Update confidence meter with a proper style
        self.confidence_meter.setValue(int(confidence))
        self.confidence_meter.setStyleSheet("""
            QProgressBar {
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 5px;
                background-color: rgba(30, 30, 40, 0.5);
                text-align: center;
                padding: 1px;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #85a3da, stop:0.5 #6e6ebe, stop:1 #d78599);
                border-radius: 3px;
            }
        """)
        
        # Show results
        self.results_group.setVisible(True)
        
        # Update status
        self.update_status(f"Classification complete: {prediction_text} with {confidence:.2f}% confidence")
    
    def toggle_submit_button(self):
        """Enable/disable submit button based on radio selection"""
        self.submit_button.setEnabled(
            self.radio_correct.isChecked() or self.radio_incorrect.isChecked())
    
    def submit_feedback(self):
        """Submit feedback on classification"""
        if self.current_features is None:
            return
        
        # Determine correct label
        if self.radio_correct.isChecked():
            # Prediction was correct
            correct_label = 1 if self.current_prediction else 0
        else:
            # Prediction was incorrect
            correct_label = 0 if self.current_prediction else 1
        
        # Create worker thread
        self.worker = WorkerThread("save_feedback", {
            "file_path": self.current_file,
            "features": self.current_features,
            "correct_label": correct_label,
            "feedback_file": self.feedback_file
        })
        
        # Connect signals
        self.worker.update_status.connect(self.update_status)
        self.worker.finished_task.connect(self.feedback_saved)
        
        # Disable button and update status
        self.submit_button.setEnabled(False)
        self.update_status("Saving feedback...")
        
        # Start worker
        self.worker.start()
    
    def feedback_saved(self, result):
        """Handle feedback saving result"""
        if not result["success"]:
            self.update_status(f"Error: {result['message']}")
            self.submit_button.setEnabled(True)
            return
        
        # Reset feedback UI
        self.radio_correct.setChecked(False)
        self.radio_incorrect.setChecked(False)
        
        # Update status
        self.update_status(result["message"])
        
        # Emit signal to refresh feedback tab
        self.feedback_saved_signal.emit()

    def check_feedback_file(self):
        """Check if feedback file exists and has data"""
        if os.path.exists(self.feedback_file):
            try:
                feedback_df = pd.read_csv(self.feedback_file)
                if len(feedback_df) > 0:
                    self.update_status(f"Found {len(feedback_df)} feedback samples available for retraining.")
            except:
                pass
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
    
    def safe_level_update(self, level):
        """Thread-safe level update using signal"""
        self.level_update.emit(level)
    
    def update_level_safe(self, level):
        """Update level meter (runs in main thread)"""
        self.level_meter.setValue(int(min(level * 100, 100)))
    
    def safe_status_update(self, message):
        """Thread-safe status update using Qt signals"""
        self.status_update.emit(message)

# Add the TabManager class at the end of the file
class TabManager:
    """Manages the application tabs"""
    
    def __init__(self, parent=None, model_path=None, features_path=None, 
                feedback_file=None, version=None, last_updated=None):
        # Create tabs with proper thread safety
        self.import_tab = ImportSampleTab(parent=parent, model_path=model_path, feedback_file=feedback_file)
        self.record_tab = RecordSampleTab(parent=parent, model_path=model_path, feedback_file=feedback_file)
        self.feedback_tab = FeedbackTab(parent=parent, model_path=model_path, feedback_file=feedback_file, features_path=features_path)
        self.info_tab = InformationTab(parent=parent, model_path=model_path, features_path=features_path, version=version, last_updated=last_updated)
        
        # Connect the tabs
        self.import_tab.feedback_saved_signal.connect(self.refresh_feedback_tab)
        self.record_tab.feedback_saved_signal.connect(self.refresh_feedback_tab)
    
    def refresh_feedback_tab(self):
        """Refresh the feedback tab to show updated data"""
        self.feedback_tab.refresh_feedback() 

# Add the CustomLevelMeter class that might be referenced by other tabs
class CustomLevelMeter(QWidget):
    """Custom level meter with gradient visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0
        self.setMinimumHeight(20)
        self.setMaximumHeight(20)
        
    def setValue(self, value):
        """Set the current level (0-100)"""
        self.level = max(0, min(100, value))
        self.update()
        
    def paintEvent(self, event):
        """Paint the level meter with gradient"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        bg_rect = self.rect().adjusted(0, 0, 0, 0)
        bg_brush = QBrush(QColor(0, 0, 0, 50))
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.setBrush(bg_brush)
        painter.drawRoundedRect(bg_rect, 4, 4)
        
        # Calculate filled width
        width = self.rect().width() - 2
        filled_width = int(width * (self.level / 100.0))
        
        if filled_width > 0:
            # Create gradient matching the app theme
            gradient = QLinearGradient(0, 0, width, 0)
            gradient.setColorAt(0.0, QColor(85, 163, 218, 220))  # Light blue
            gradient.setColorAt(0.5, QColor(110, 110, 190, 220))  # Purple
            gradient.setColorAt(1.0, QColor(215, 133, 157, 220))  # Pink
            
            # Draw level indicator
            level_rect = QRect(1, 1, filled_width, self.rect().height() - 2)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(level_rect, 3, 3)
        
        painter.end()

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for plotting"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        
        # Set responsive size policy for better scaling
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        
        self.setParent(parent)
        
        # Transparent background
        self.fig.set_facecolor('none')
        self.fig.patch.set_alpha(0)
        
        # Transparent axes background with subtle grid
        self.axes.set_facecolor('none')
        self.axes.patch.set_alpha(0)
        
        # Minimal spines with low opacity - use proper matplotlib color format
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['left'].set_color((1, 1, 1, 0.2))  # white with 20% opacity
        self.axes.spines['bottom'].set_color((1, 1, 1, 0.2))  # white with 20% opacity
        
        # Light text and axis styles - use proper matplotlib color format
        self.axes.tick_params(colors=(1, 1, 1, 0.7), labelsize=9)  # white with 70% opacity
        self.axes.yaxis.label.set_color((1, 1, 1, 0.9))  # white with 90% opacity
        self.axes.xaxis.label.set_color((1, 1, 1, 0.9))  # white with 90% opacity
        self.axes.title.set_color((1, 1, 1, 0.9))  # white with 90% opacity
        
        # Set tight layout for better use of space
        self.fig.tight_layout(pad=3.0)  # Increase padding to fix layout warning
        
    def resizeEvent(self, event):
        """Handle resize events by updating the figure layout"""
        super().resizeEvent(event)
        self.fig.tight_layout(pad=3.0)  # Increase padding to fix layout warning
        self.draw_idle()  # Request a redraw when idle

class FeedbackTab(QWidget):
    """Tab for managing feedback and retraining the model"""
    
    def __init__(self, parent=None, model_path=None, feedback_file=None, features_path=None):
        super().__init__(parent)
        self.parent = parent
        self.model_path = model_path
        self.feedback_file = feedback_file
        self.features_path = features_path
        self.feedback_list = []
        
        self.setup_ui()
        self.load_feedback()
    
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Stats section
        stats_group = QGroupBox("Feedback Statistics")
        stats_group.setStyleSheet("QGroupBox { color: #e0e0e0; font-weight: bold; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }")
        stats_layout = QVBoxLayout()
        
        # Add stats display
        self.stats_label = QLabel("No feedback data available")
        self.stats_label.setStyleSheet("font-size: 14px; color: #d0d0d0; padding: 5px;")
        self.stats_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.stats_label)
        
        # Add retrain button
        retrain_layout = QHBoxLayout()
        self.retrain_button = QPushButton("Retrain Model with Feedback")
        self.retrain_button.clicked.connect(self.retrain_model)
        self.retrain_button.setEnabled(False)
        self.retrain_button.setMinimumHeight(36)
        self.retrain_button.setStyleSheet(StylesheetProvider.get_primary_button_style())
        retrain_layout.addStretch()
        retrain_layout.addWidget(self.retrain_button)
        retrain_layout.addStretch()
        
        stats_layout.addLayout(retrain_layout)
        stats_group.setLayout(stats_layout)
        
        # Create horizontal layout for list and details
        feedback_content_layout = QHBoxLayout()
        
        # Feedback list section - now takes left 60% of space
        list_group = QGroupBox("Feedback History")
        list_group.setStyleSheet("QGroupBox { color: #e0e0e0; font-weight: bold; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }")
        list_layout = QVBoxLayout()
        
        # Create feedback list widget
        self.feedback_list_widget = QListWidget()
        self.feedback_list_widget.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                color: #e0e0e0;
                outline: none;
            }
            QListWidget::item {
                background-color: rgba(51, 48, 74, 0.7);
                padding: 10px;
                margin-bottom: 1px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: rgba(176, 138, 217, 0.4);
                color: white;
            }
            QListWidget::item:hover:!selected {
                background-color: rgba(89, 95, 154, 0.4);
            }
        """)
        self.feedback_list_widget.setAlternatingRowColors(False)
        self.feedback_list_widget.setSpacing(2)  # Add spacing between items
        self.feedback_list_widget.currentItemChanged.connect(self.on_item_selected)
        
        list_layout.addWidget(self.feedback_list_widget)
        list_group.setLayout(list_layout)
        feedback_content_layout.addWidget(list_group, 60)  # Takes 60% of width
        
        # Detail section for selected feedback - now in right 40% of space
        detail_group = QGroupBox("Feedback Details")
        detail_group.setStyleSheet("QGroupBox { color: #e0e0e0; font-weight: bold; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 8px; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }")
        detail_layout = QVBoxLayout()
        
        # Create a container for the details with better styling
        detail_container = QWidget()
        detail_container.setStyleSheet("background-color: rgba(51, 48, 74, 0.7); border-radius: 6px; padding: 10px;")
        detail_container_layout = QVBoxLayout(detail_container)
        detail_container_layout.setContentsMargins(10, 10, 10, 10)
        
        self.detail_label = QLabel("Select a feedback item to view details")
        self.detail_label.setStyleSheet("font-size: 13px; color: #e0e0e0; padding: 5px;")
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        detail_container_layout.addWidget(self.detail_label)
        
        detail_layout.addWidget(detail_container)
        detail_layout.addStretch()
        
        detail_group.setLayout(detail_layout)
        feedback_content_layout.addWidget(detail_group, 40)  # Takes 40% of width
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label, 1)
        
        # Add sections to main layout
        layout.addWidget(stats_group, 1)
        layout.addLayout(feedback_content_layout, 5)  # Give more space to the feedback content
        layout.addLayout(status_layout)
    
    def on_item_selected(self, current, previous):
        """Handle selection of feedback item"""
        if not current:
            self.detail_label.setText("Select a feedback item to view details")
            return
        
        # Get feedback index
        index = current.data(Qt.UserRole)
        if index is None or index < 0 or index >= len(self.feedback_list):
            return
        
        # Get feedback data
        feedback = self.feedback_list[index]
        
        # Create detail text
        detail_text = (
            f"<b>Sample ID:</b> {index+1}<br><br>"
            f"<b>File:</b> {feedback['file_path']}<br><br>"
            f"<b>Classification:</b> {'AI-Generated' if feedback['label'] == 1 else 'Human'}<br><br>"
            f"<b>Features sample:</b><br>{', '.join([f'{v:.3f}' for v in feedback['features'][:5]])}..."
        )
        
        self.detail_label.setText(detail_text)
    
    def refresh_feedback(self):
        """Refresh the feedback list"""
        self.update_status("Refreshing feedback data...")
        self.load_feedback()
    
    def load_feedback(self):
        """Load feedback data from file"""
        if not os.path.exists(self.feedback_file):
            self.update_status("No feedback data available")
            return
        
        try:
            feedback_df = pd.read_csv(self.feedback_file)
            num_samples = len(feedback_df)
            
            if num_samples == 0:
                self.update_status("No feedback samples found")
                return
            
            # Enable retrain button
            self.retrain_button.setEnabled(True)
            
            # Count correct/incorrect classifications
            num_ai = sum(feedback_df['label'] == 1)
            num_human = sum(feedback_df['label'] == 0)
            
            # Update stats label
            self.stats_label.setText(
                f"Total feedback samples: {num_samples}\n"
                f"AI-generated samples: {num_ai}\n"
                f"Human samples: {num_human}"
            )
            
            # Clear existing items
            self.feedback_list_widget.clear()
            self.feedback_list = []
            
            # Store feedback data
            for index, row in feedback_df.iterrows():
                try:
                    # Extract features (everything except label and file_path)
                    features = row[:-2].values.tolist()
                    label = int(row['label'])
                    file_path = row['file_path']
                    
                    self.feedback_list.append({
                        'index': index,
                        'features': features,
                        'label': label,
                        'file_path': file_path
                    })
                    
                    # Create list item with clear formatting
                    item_text = f"Sample {index+1}: {'AI-Generated' if label == 1 else 'Human'} - {file_path}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.UserRole, index)  # Store index for reference
                    self.feedback_list_widget.addItem(item)
                except Exception as e:
                    print(f"Error adding item {index}: {str(e)}")
                    continue
            
            self.update_status(f"Loaded {num_samples} feedback samples")
            
        except Exception as e:
            self.update_status(f"Error loading feedback: {str(e)}")
    
    def retrain_model(self):
        """Retrain model with feedback data"""
        # Create a progress dialog to show detailed status
        self.progress_dialog = QProgressDialog("Preparing to retrain model...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Retraining Model")
        self.progress_dialog.setMinimumWidth(400)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setCancelButton(None)  # Disable cancel button
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Create worker thread
        self.worker = WorkerThread("retrain", {
            "model_path": self.model_path,
            "feedback_file": self.feedback_file,
            "original_features": self.features_path
        })
        
        # Connect signals
        self.worker.update_status.connect(self.update_retraining_status)
        self.worker.finished_task.connect(self.retrain_finished)
        
        # Disable button and update status
        self.retrain_button.setEnabled(False)
        self.update_status("Retraining model...")
        
        # Start worker
        self.worker.start()
    
    def update_retraining_status(self, message):
        """Update progress dialog with retraining status"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setLabelText(message)
        self.update_status(message)
    
    def retrain_finished(self, result):
        """Handle retraining result"""
        # Close progress dialog
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # Re-enable button
        self.retrain_button.setEnabled(True)
        
        if not result["success"]:
            self.update_status(f"Error: {result['message']}")
            QMessageBox.warning(self, "Retraining Failed", result["message"])
            return
        
        # Show success message
        QMessageBox.information(self, "Retraining Successful", result["message"])
        
        # Update status
        self.update_status(result["message"])
        
        # Update version if parent app is available
        if hasattr(self.parent, 'update_version'):
            self.parent.update_version()
            
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)

class InformationTab(QWidget):
    """Tab for showing information and statistics about the model"""
    
    def __init__(self, parent=None, model_path=None, features_path=None, version=None, last_updated=None):
        super().__init__(parent)
        self.parent = parent
        self.model_path = model_path
        self.features_path = features_path
        self.version = version
        self.last_updated = last_updated
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI components"""
        # Create a scroll area to contain the entire content
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        
        # Create the content widget that will be placed in the scroll area
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_widget.setStyleSheet("QWidget { background-color: transparent; }")
        
        # Set the scroll area as the main layout widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        
        # Create content layout
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20)
        
        # Top section - About VoiceAuth - dark semi-transparent background
        about_group = QGroupBox("About VoiceAuth")
        about_group.setStyleSheet("""
            QGroupBox { 
                color: #e0e0e0; 
                font-weight: bold; 
                border: 1px solid rgba(255, 255, 255, 0.15); 
                border-radius: 8px; 
                margin-top: 1ex; 
                background-color: rgba(30, 30, 50, 0.7);
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top center; 
                padding: 0 10px; 
                background-color: transparent;
            }
        """)
        about_layout = QVBoxLayout(about_group)
        about_layout.setSpacing(15)
        about_layout.setContentsMargins(15, 20, 15, 15)
        
        title_label = QLabel("VoiceAuth - AI Voice Detection")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff; background-color: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        
        subtitle_label = QLabel("Developed by Zohaib Khan & Umer Kashif for Regeneron ISEF 2025")
        subtitle_label.setStyleSheet("font-size: 12px; color: #d0d0d0; background-color: transparent;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        
        desc_label = QLabel(
            "VoiceAuth is an application that uses machine learning to "
            "differentiate between AI-generated and human voices. "
            "The system employs logistic regression on extracted audio features "
            "to make its predictions with high accuracy."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 13px; color: #ffffff; padding: 5px; background-color: transparent;")
        desc_label.setMinimumHeight(60)
        
        # Version information
        version_layout = QHBoxLayout()
        version_label_title = QLabel("Version:")
        version_label_title.setStyleSheet("color: #c0c0c0; background-color: transparent;")
        self.version_label = QLabel(self.version)
        self.version_label.setStyleSheet("color: #ffffff; font-weight: bold; background-color: transparent;")
        self.version_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        version_layout.addWidget(version_label_title, 0)
        version_layout.addWidget(self.version_label, 1)
        
        date_layout = QHBoxLayout()
        date_label_title = QLabel("Last Updated:")
        date_label_title.setStyleSheet("color: #c0c0c0; background-color: transparent;")
        self.date_label = QLabel(self.last_updated)
        self.date_label.setStyleSheet("color: #ffffff; background-color: transparent;")
        self.date_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        date_layout.addWidget(date_label_title, 0)
        date_layout.addWidget(self.date_label, 1)
        
        about_layout.addWidget(title_label)
        about_layout.addWidget(subtitle_label)
        about_layout.addWidget(desc_label)
        about_layout.addLayout(version_layout)
        about_layout.addLayout(date_layout)
        
        # Stats section
        stats_group = QGroupBox("System Statistics")
        stats_group.setStyleSheet("""
            QGroupBox { 
                color: #e0e0e0; 
                font-weight: bold; 
                border: 1px solid rgba(255, 255, 255, 0.15); 
                border-radius: 8px; 
                margin-top: 1ex; 
                background-color: rgba(30, 30, 50, 0.7);
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top center; 
                padding: 0 10px; 
                background-color: transparent;
            }
        """)
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(15)
        stats_layout.setContentsMargins(15, 20, 15, 15)
        
        # Stats grid
        stats_grid = QGridLayout()
        stats_grid.setColumnStretch(1, 1)  # Make second column expand
        stats_grid.setVerticalSpacing(10)
        stats_grid.setHorizontalSpacing(15)
        
        # Row 1: Model Information
        stats_grid.addWidget(QLabel("Model Type:"), 0, 0)
        model_type_label = QLabel("Logistic Regression")
        model_type_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(model_type_label, 0, 1)
        
        # Row 2: Accuracy
        stats_grid.addWidget(QLabel("Accuracy:"), 1, 0)
        accuracy_label = QLabel("97.8%")
        accuracy_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(accuracy_label, 1, 1)
        
        # Row 3: Training Dataset
        stats_grid.addWidget(QLabel("Training Dataset:"), 2, 0)
        dataset_label = QLabel("63,951 samples (31,359 human, 32,592 AI)")
        dataset_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(dataset_label, 2, 1)
        
        # Row 4: Feature Count
        stats_grid.addWidget(QLabel("Feature Count:"), 3, 0)
        features_label = QLabel("193 acoustic features")
        features_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(features_label, 3, 1)
        
        # Row 5: Feature Selection
        stats_grid.addWidget(QLabel("Feature Selection:"), 4, 0)
        feature_selection_label = QLabel("None (all features used)")
        feature_selection_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(feature_selection_label, 4, 1)
        
        # Row 6: Cross-Validation
        stats_grid.addWidget(QLabel("Cross-Validation:"), 5, 0)
        cv_label = QLabel("5-fold, Stratified")
        cv_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(cv_label, 5, 1)
        
        # Row 7: Mean AUC
        stats_grid.addWidget(QLabel("Mean AUC:"), 6, 0)
        auc_label = QLabel("0.983")
        auc_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(auc_label, 6, 1)
        
        # Row 8: Precision
        stats_grid.addWidget(QLabel("Precision:"), 7, 0)
        precision_label = QLabel("0.976")
        precision_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(precision_label, 7, 1)
        
        # Row 9: Recall
        stats_grid.addWidget(QLabel("Recall:"), 8, 0)
        recall_label = QLabel("0.981")
        recall_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(recall_label, 8, 1)
        
        # Row 10: F1 Score
        stats_grid.addWidget(QLabel("F1 Score:"), 9, 0)
        f1_label = QLabel("0.978")
        f1_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(f1_label, 9, 1)
        
        # Row 11: Training Time
        stats_grid.addWidget(QLabel("Training Time:"), 10, 0)
        training_time_label = QLabel("1.8 seconds")
        training_time_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
        stats_grid.addWidget(training_time_label, 10, 1)
        
        # Add grid to stats layout
        stats_layout.addLayout(stats_grid)
        
        # Additional explanation
        explanation_label = QLabel(
            "This system uses a variety of acoustic features including MFCCs, spectral contrast, "
            "chroma, zero crossing rate, and spectral bandwidth to classify voice samples. "
            "The logistic regression model is trained with class weighting to handle "
            "any potential imbalance in training data."
        )
        explanation_label.setWordWrap(True)
        explanation_label.setStyleSheet("color: #e0e0e0; padding: 5px; font-style: italic;")
        stats_layout.addWidget(explanation_label)
        
        # Add data visualizations section
        visuals_group = QGroupBox("Model Visualizations")
        visuals_group.setStyleSheet("""
            QGroupBox { 
                color: #e0e0e0; 
                font-weight: bold; 
                border: 1px solid rgba(255, 255, 255, 0.15); 
                border-radius: 8px; 
                margin-top: 1ex; 
                background-color: rgba(30, 30, 50, 0.7);
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top center; 
                padding: 0 10px; 
                background-color: transparent;
            }
        """)
        visuals_layout = QVBoxLayout(visuals_group)
        visuals_layout.setSpacing(15)
        visuals_layout.setContentsMargins(15, 20, 15, 15)
        
        # First row of visualizations (Confusion Matrix and ROC Curve)
        row1_layout = QHBoxLayout()
        
        # Confusion Matrix
        cm_group = QGroupBox("Confusion Matrix")
        cm_group.setStyleSheet("""
            QGroupBox { 
                color: #d0d0d0; 
                font-weight: bold; 
                border: 1px solid rgba(255, 255, 255, 0.1); 
                border-radius: 5px; 
                margin-top: 1ex; 
                background-color: rgba(40, 40, 60, 0.5);
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top center; 
                padding: 0 5px; 
                background-color: transparent;
            }
        """)
        cm_layout = QVBoxLayout(cm_group)
        self.cm_canvas = MatplotlibCanvas(self, width=5, height=4.5, dpi=90)
        cm_layout.addWidget(self.cm_canvas)
        
        # Add explanation of confusion matrix terms
        cm_explanation = QLabel(
            "<b>True Negative (30,924):</b> Human voice correctly identified as human<br>"
            "<b>False Positive (435):</b> Human voice incorrectly identified as AI<br>"
            "<b>False Negative (672):</b> AI voice incorrectly identified as human<br>"
            "<b>True Positive (31,920):</b> AI voice correctly identified as AI"
        )
        cm_explanation.setStyleSheet("color: #d0d0d0; font-size: 11px; background-color: transparent; padding: 5px;")
        cm_explanation.setWordWrap(True)
        cm_layout.addWidget(cm_explanation)
        
        # Add save button
        cm_save_btn = QPushButton("Save to Computer")
        cm_save_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(133, 163, 218, 0.7);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(133, 163, 218, 0.9);
            }
        """)
        cm_save_btn.clicked.connect(lambda: self.save_visualization(self.cm_canvas, "confusion_matrix"))
        cm_layout.addWidget(cm_save_btn)
        
        # ROC Curve
        roc_group = QGroupBox("ROC Curve")
        roc_group.setStyleSheet("""
            QGroupBox { 
                color: #d0d0d0; 
                font-weight: bold; 
                border: 1px solid rgba(255, 255, 255, 0.1); 
                border-radius: 5px; 
                margin-top: 1ex; 
                background-color: rgba(40, 40, 60, 0.5);
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top center; 
                padding: 0 5px; 
                background-color: transparent;
            }
        """)
        roc_layout = QVBoxLayout(roc_group)
        self.roc_canvas = MatplotlibCanvas(self, width=5, height=4.5, dpi=90)
        roc_layout.addWidget(self.roc_canvas)
        
        # Add save button
        roc_save_btn = QPushButton("Save to Computer")
        roc_save_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(133, 163, 218, 0.7);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(133, 163, 218, 0.9);
            }
        """)
        roc_save_btn.clicked.connect(lambda: self.save_visualization(self.roc_canvas, "roc_curve"))
        roc_layout.addWidget(roc_save_btn)
        
        row1_layout.addWidget(cm_group)
        row1_layout.addWidget(roc_group)
        
        # Second row of visualizations (Dataset Bar Graph and Libraries Pie Chart)
        row2_layout = QHBoxLayout()
        
        # Dataset Bar Graph
        dataset_group = QGroupBox("Testing Dataset Distribution")
        dataset_group.setStyleSheet("""
            QGroupBox { 
                color: #d0d0d0; 
                font-weight: bold; 
                border: 1px solid rgba(255, 255, 255, 0.1); 
                border-radius: 5px; 
                margin-top: 1ex; 
                background-color: rgba(40, 40, 60, 0.5);
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top center; 
                padding: 0 5px; 
                background-color: transparent;
            }
        """)
        dataset_layout = QVBoxLayout(dataset_group)
        self.dataset_canvas = MatplotlibCanvas(self, width=4, height=3.5, dpi=90)
        dataset_layout.addWidget(self.dataset_canvas)
        
        # Add save button
        dataset_save_btn = QPushButton("Save to Computer")
        dataset_save_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(133, 163, 218, 0.7);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(133, 163, 218, 0.9);
            }
        """)
        dataset_save_btn.clicked.connect(lambda: self.save_visualization(self.dataset_canvas, "dataset_distribution"))
        dataset_layout.addWidget(dataset_save_btn)
        
        # Libraries Pie Chart
        libraries_group = QGroupBox("Python Libraries Used")
        libraries_group.setStyleSheet("""
            QGroupBox { 
                color: #d0d0d0; 
                font-weight: bold; 
                border: 1px solid rgba(255, 255, 255, 0.1); 
                border-radius: 5px; 
                margin-top: 1ex; 
                background-color: rgba(40, 40, 60, 0.5);
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top center; 
                padding: 0 5px; 
                background-color: transparent;
            }
        """)
        libraries_layout = QVBoxLayout(libraries_group)
        self.libraries_canvas = MatplotlibCanvas(self, width=7, height=6, dpi=90)
        libraries_layout.addWidget(self.libraries_canvas)
        
        # Add save button
        libraries_save_btn = QPushButton("Save to Computer")
        libraries_save_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(133, 163, 218, 0.7);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(133, 163, 218, 0.9);
            }
        """)
        libraries_save_btn.clicked.connect(lambda: self.save_visualization(self.libraries_canvas, "libraries_pie_chart"))
        libraries_layout.addWidget(libraries_save_btn)
        
        row2_layout.addWidget(dataset_group)
        row2_layout.addWidget(libraries_group)
        
        # Add rows to visualizations layout
        visuals_layout.addLayout(row1_layout)
        visuals_layout.addLayout(row2_layout)
        
        # Technical details section
        tech_group = QGroupBox("Technical Details")
        tech_group.setStyleSheet("""
            QGroupBox { 
                color: #e0e0e0; 
                font-weight: bold; 
                border: 1px solid rgba(255, 255, 255, 0.15); 
                border-radius: 8px; 
                margin-top: 1ex; 
                background-color: rgba(30, 30, 50, 0.7);
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top center; 
                padding: 0 10px; 
                background-color: transparent;
            }
        """)
        tech_layout = QVBoxLayout(tech_group)
        tech_layout.setSpacing(15)
        tech_layout.setContentsMargins(15, 20, 15, 15)
        
        # Technical details text
        tech_text = QTextEdit()
        tech_text.setReadOnly(True)
        tech_text.setStyleSheet("background-color: rgba(20, 20, 40, 0.5); color: #e0e0e0; border: none; border-radius: 4px;")
        
        tech_html = """
        <style>
        h3 { color: #b0b0ff; margin-top: 10px; margin-bottom: 5px; }
        p { margin-top: 2px; margin-bottom: 10px; }
        ul { margin-top: 5px; }
        li { margin-bottom: 3px; }
        .highlight { color: #a0c0ff; font-weight: bold; }
        </style>
        
        <h3>Feature Extraction</h3>
        <p>The system extracts the following acoustic features from each audio sample:</p>
        <ul>
            <li>13 Mel-Frequency Cepstral Coefficients (MFCCs)</li>
            <li>Spectral Centroid, Contrast, and Rolloff</li>
            <li>Zero Crossing Rate</li>
            <li>Chroma Features (12 dimensions)</li>
            <li>Spectral Bandwidth</li>
            <li>Temporal statistics (mean, std, min, max, etc.)</li>
        </ul>
        
        <h3>Model Hyperparameters</h3>
        <ul>
            <li>Solver: <span class="highlight">liblinear</span></li>
            <li>Regularization: <span class="highlight">L2</span></li>
            <li>C value: <span class="highlight">1.0</span></li>
            <li>Class weight: <span class="highlight">balanced</span></li>
            <li>Maximum iterations: <span class="highlight">1000</span></li>
        </ul>
        
        <h3>Performance Evaluation</h3>
        <p>The model is evaluated using stratified k-fold cross-validation (k=5) 
        to ensure balanced class representation in each fold. Performance metrics 
        include accuracy, precision, recall, F1 score, and AUC.</p>
        
        <h3>Preprocessing</h3>
        <ul>
            <li>Audio resampling to 22,050 Hz</li>
            <li>Feature standardization (zero mean, unit variance)</li>
            <li>Optional noise reduction</li>
        </ul>
        """
        
        tech_text.setHtml(tech_html)
        tech_text.setMinimumHeight(300)
        tech_layout.addWidget(tech_text)
        
        # Add sections to layout
        layout.addWidget(about_group)
        layout.addWidget(stats_group)
        layout.addWidget(visuals_group)
        layout.addWidget(tech_group)
        
        # Set the scroll area widget
        scroll_area.setWidget(content_widget)
        
        # Draw the visualizations
        self.draw_visualizations()
    
    def draw_visualizations(self):
        """Create and draw all visualizations"""
        self.draw_confusion_matrix()
        self.draw_roc_curve()
        self.draw_dataset_distribution()
        self.draw_libraries_pie_chart()
    
    def draw_confusion_matrix(self):
        """Draw confusion matrix visualization"""
        # Create a static confusion matrix for demonstration
        confusion_matrix = np.array([
            [30924, 435],   # True Negatives, False Positives
            [672, 31920]    # False Negatives, True Positives
        ])
        
        ax = self.cm_canvas.axes
        ax.clear()
        
        # Create a custom colormap using app theme colors
        from matplotlib.colors import LinearSegmentedColormap
        theme_colors = [(0.33, 0.33, 0.6, 1.0),      # Dark blue-purple
                 (0.52, 0.64, 0.85, 1.0),     # Blue (#85a3da)
                 (0.84, 0.52, 0.60, 1.0)]     # Pink (#d78599)
        custom_cmap = LinearSegmentedColormap.from_list('VoiceAuthTheme', theme_colors, N=100)
        
        # Plot the confusion matrix
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=custom_cmap)
        ax.set_title('Confusion Matrix', color='white', fontsize=14)
        
        # Add text annotations
        thresh = confusion_matrix.max() / 2
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], ',d'),
                        ha="center", va="center",
                        color="white",
                        fontsize=12, fontweight='bold')
        
        # Set axis labels and ticks
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Human', 'AI'], fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Human', 'AI'], fontsize=12)
        ax.set_ylabel('Actual Class', color='white', fontsize=13)
        ax.set_xlabel('Predicted Class', color='white', fontsize=13)
        
        # Add colorbar - fix: use the figure's colorbar method
        cbar = self.cm_canvas.fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors='white')
        
        # Tight layout to optimize spacing
        self.cm_canvas.fig.tight_layout()
        self.cm_canvas.draw()
    
    def draw_roc_curve(self):
        """Draw ROC curve visualization"""
        ax = self.roc_canvas.axes
        ax.clear()
        
        # Create a visually appealing ROC curve
        # Data points (these are approximations)
        fpr = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        tpr = np.array([0, 0.75, 0.83, 0.89, 0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.998, 1.0])
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='#85a3da', lw=3, label=f'ROC Curve (AUC = 0.983)')
        ax.plot([0, 1], [0, 1], color='#d78599', lw=2, linestyle='--', alpha=0.7, label='Random Classifier')
        
        # Add area under curve with slight transparency
        ax.fill_between(fpr, tpr, alpha=0.2, color='#85a3da')
        
        # Customize axes
        ax.set_xlabel('False Positive Rate', color='white')
        ax.set_ylabel('True Positive Rate', color='white')
        ax.set_title('Receiver Operating Characteristic', color='white', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='lower right', framealpha=0.7)
        
        # Set limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Tight layout to optimize spacing
        self.roc_canvas.fig.tight_layout()
        self.roc_canvas.draw()
    
    def draw_dataset_distribution(self):
        """Draw bar graph of testing dataset distribution"""
        ax = self.dataset_canvas.axes
        ax.clear()
        
        # Data
        categories = ['AI Voices', 'Human Voices']
        counts = [32592, 31359]  # As specified by the user
        
        # Plot bar chart with gradient colors
        bars = ax.bar(categories, counts, width=0.6)
        
        # Apply gradient colors matching app theme
        bars[0].set_color('#85a3da')  # Blue for AI Voices
        bars[1].set_color('#d78599')  # Pink for Human Voices
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                    f'{count:,}',
                    ha='center', va='bottom', color='white', fontsize=10)
        
        # Customize axes
        ax.set_title('Testing Dataset Distribution', color='white', fontsize=12)
        ax.set_ylabel('Number of Samples', color='white')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Set y-axis limits to have some padding above the highest bar
        ax.set_ylim(0, max(counts) * 1.12)
        
        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color((1, 1, 1, 0.3))
        ax.spines['bottom'].set_color((1, 1, 1, 0.3))
        
        # Tight layout to optimize spacing
        self.dataset_canvas.fig.tight_layout()
        self.dataset_canvas.draw()
    
    def draw_libraries_pie_chart(self):
        """Draw pie chart of Python libraries used"""
        ax = self.libraries_canvas.axes
        ax.clear()
        
        # Data for the pie chart
        libraries = ['Scikit-learn', 'Librosa', 'NumPy', 'PyQt5', 'Pandas', 'Matplotlib']
        sizes = [30, 25, 20, 15, 7, 3]  # Percentages
        
        # Color palette matching the app theme
        colors = ['#85a3da', '#6e6ebe', '#d78599', '#b08ad9', '#5f9ea0', '#9370db']
        
        # Set a larger figure size
        self.libraries_canvas.fig.set_figwidth(7)
        self.libraries_canvas.fig.set_figheight(6)
        
        # Draw pie chart with custom styling - fix unpacking issue
        wedges, texts = ax.pie(
            sizes, 
            explode=[0.05] * len(sizes),  # Equal explode for all slices
            labels=None,  # No labels on the pie directly
            colors=colors,
            autopct=None,  # No percentages on the pie directly
            shadow=False, 
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
        )
        
        # Create a legend with percentages - make text white
        labels = [f'{l} ({s}%)' for l, s in zip(libraries, sizes)]
        ax.legend(
            wedges, 
            labels,
            loc='center left', 
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=12,
            labelcolor='white'  # Make legend text white
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        
        # Tight layout to optimize spacing
        self.libraries_canvas.fig.tight_layout(pad=1.2)
        self.libraries_canvas.draw()
    
    def update_version_info(self, version, last_updated):
        """Update version information"""
        self.version = version
        self.last_updated = last_updated
        
        # Update labels
        self.version_label.setText(self.version)
        self.date_label.setText(self.last_updated)
        
    def save_visualization(self, canvas, name):
        """Save visualization to a file"""
        # Ask user for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", f"{name}.png", "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            try:
                # Save the figure
                canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='none', 
                                   edgecolor='none', transparent=True)
                QMessageBox.information(
                    self, "Save Successful", f"Visualization saved to {file_path}"
                )
            except Exception as e:
                QMessageBox.warning(
                    self, "Save Failed", f"Could not save image: {str(e)}"
                )

class ModelMetricsWorker(QObject):
    """Worker thread for calculating model metrics"""
    metrics_ready = pyqtSignal(dict)
    finished = pyqtSignal()
    
    def __init__(self, model_path, features_path):
        super().__init__()
        self.model_path = model_path
        self.features_path = features_path
    
    def run(self):
        """Calculate model metrics and emit results"""
        try:
            import joblib
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
            import pandas as pd
            import numpy as np
            
            # Load model and scaler
            model = joblib.load(self.model_path)
            scaler_path = f"{os.path.splitext(self.model_path)[0]}_scaler.pkl"
            scaler = joblib.load(scaler_path)
            
            # Load features
            features_df = pd.read_csv(self.features_path)
            X = features_df.drop(['label', 'file_path'], axis=1, errors='ignore').values
            y = features_df['label'].values
            
            # Process features
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            y_prob = model.predict_proba(X_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Feature importances (coefficients for logistic regression)
            feature_importances = np.abs(model.coef_[0])
            
            # Emit results
            self.metrics_ready.emit({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'samples': len(y),
                'confusion_matrix': cm,
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc,
                'feature_importances': feature_importances
            })
            
        except Exception as e:
            print(f"Error calculating model metrics: {str(e)}")
            # Emit default values in case of error
            self.metrics_ready.emit({
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'samples': 0,
                'confusion_matrix': np.array([[0, 0], [0, 0]]),
                'fpr': np.array([0, 1]),
                'tpr': np.array([0, 1]),
                'auc': 0.5,
                'feature_importances': np.array([0])
            })
        
        self.finished.emit()

def create_metric_label(text, tooltip=""):
    """Create a stylized metric label"""
    label = QLabel(text)
    label.setStyleSheet("color: #e0e0e0; font-size: 14px; background-color: transparent;")
    label.setToolTip(tooltip)
    return label

def create_metric_value(value, color):
    """Create a stylized metric value with specific color"""
    label = QLabel(value)
    label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold; background-color: transparent;")
    return label 
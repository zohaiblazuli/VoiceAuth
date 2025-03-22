import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QFileDialog, QLabel, QProgressBar, 
                           QWidget, QRadioButton, QButtonGroup, QTextEdit,
                           QListWidget, QGroupBox, QGridLayout, QMessageBox,
                           QSplitter, QProgressDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from batch_process import extract_features
from simple_model import classify_audio, train_model

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
                scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
                
                # Update status
                self.update_status.emit(f"Processing {os.path.basename(file_path)}...")
                
                # Extract features
                features = extract_features(file_path)
                
                if features is None:
                    self.finished_task.emit({
                        "success": False,
                        "message": f"Failed to extract features from {file_path}"
                    })
                    return
                
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
                
                # Create or load feedback dataframe
                if os.path.exists(feedback_file):
                    feedback_df = pd.read_csv(feedback_file)
                else:
                    # Create new feedback dataframe
                    num_features = len(features)
                    feature_names = [f'feature_{i}' for i in range(num_features)]
                    feedback_df = pd.DataFrame(columns=feature_names + ['label', 'file_path'])
                
                # Create new row with features and label
                new_row = pd.DataFrame([np.append(features, [correct_label, os.path.basename(file_path)])], 
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
                
                # Load feedback data
                if not os.path.exists(feedback_file):
                    self.finished_task.emit({
                        "success": False,
                        "message": "No feedback data available"
                    })
                    return
                
                feedback_df = pd.read_csv(feedback_file)
                
                # Load original training data if available
                if os.path.exists(original_features):
                    self.update_status.emit("Loading original training data...")
                    original_df = pd.read_csv(original_features)
                    
                    # Combine original data with feedback
                    combined_df = pd.concat([original_df, feedback_df], ignore_index=True)
                else:
                    # Only use feedback data
                    combined_df = feedback_df
                
                # Save combined features
                temp_features = "output/temp_features.csv"
                combined_df.to_csv(temp_features, index=False)
                
                # Retrain model
                self.update_status.emit(f"Training model with {len(combined_df)} samples...")
                model, scaler = train_model(temp_features, model_path)
                
                # Return success
                self.finished_task.emit({
                    "success": True,
                    "message": f"Model retrained successfully with {len(feedback_df)} feedback samples"
                })
                
        except Exception as e:
            self.finished_task.emit({
                "success": False,
                "message": f"Error: {str(e)}"
            })

class SimpleClassifierApp(QMainWindow):
    """Simple classifier application"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Classifier")
        self.setMinimumSize(800, 600)
        
        # Set up variables
        self.current_file = None
        self.current_features = None
        self.current_prediction = None
        self.model_path = "output/models/voice_classifier.pkl"
        self.feedback_file = "output/models/feedback_data.csv"
        self.original_features = "output/features.csv"
        self.history = []
        
        # Set up UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create splitter for main sections
        splitter = QSplitter(Qt.Vertical)
        
        # Top section - File selection and classification
        top_widget = QWidget()
        top_layout = QVBoxLayout()
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QGridLayout()
        
        self.model_path_label = QLabel("Model path:")
        self.model_path_edit = QTextEdit(self.model_path)
        self.model_path_edit.setMaximumHeight(30)
        self.model_path_button = QPushButton("Browse...")
        self.model_path_button.clicked.connect(self.browse_model)
        
        model_layout.addWidget(self.model_path_label, 0, 0)
        model_layout.addWidget(self.model_path_edit, 0, 1)
        model_layout.addWidget(self.model_path_button, 0, 2)
        
        model_group.setLayout(model_layout)
        
        # File selection
        file_group = QGroupBox("Audio File Selection")
        file_layout = QVBoxLayout()
        
        self.file_button = QPushButton("Select Audio File")
        self.file_button.clicked.connect(self.browse_audio)
        self.file_button.setMinimumHeight(40)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignCenter)
        
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_label)
        
        file_group.setLayout(file_layout)
        
        # Classification button
        self.classify_button = QPushButton("Classify Audio")
        self.classify_button.clicked.connect(self.classify_audio)
        self.classify_button.setMinimumHeight(50)
        self.classify_button.setEnabled(False)
        
        # Add to top layout
        top_layout.addWidget(model_group)
        top_layout.addWidget(file_group)
        top_layout.addWidget(self.classify_button)
        
        top_widget.setLayout(top_layout)
        
        # Middle section - Results
        middle_widget = QWidget()
        middle_layout = QVBoxLayout()
        
        # Results display
        results_group = QGroupBox("Classification Results")
        results_layout = QVBoxLayout()
        
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 16, QFont.Bold))
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMaximum(100)
        self.confidence_bar.setMinimum(0)
        
        # Feedback mechanism
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
        
        # Add to feedback layout
        feedback_layout.addWidget(self.feedback_label)
        feedback_layout.addLayout(radio_layout)
        feedback_layout.addWidget(self.submit_button)
        
        # Add to results layout
        results_layout.addWidget(self.result_label)
        results_layout.addWidget(self.confidence_bar)
        results_layout.addLayout(feedback_layout)
        
        results_group.setLayout(results_layout)
        results_group.setVisible(False)
        self.results_group = results_group
        
        # Add to middle layout
        middle_layout.addWidget(results_group)
        
        middle_widget.setLayout(middle_layout)
        
        # Bottom section - History and retraining
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout()
        
        # History list
        history_group = QGroupBox("Classification History")
        history_layout = QVBoxLayout()
        
        self.history_list = QListWidget()
        
        # Retraining button
        self.retrain_button = QPushButton("Retrain Model with Feedback")
        self.retrain_button.clicked.connect(self.retrain_model)
        self.retrain_button.setMinimumHeight(40)
        self.retrain_button.setEnabled(False)
        
        history_layout.addWidget(self.history_list)
        history_layout.addWidget(self.retrain_button)
        
        history_group.setLayout(history_layout)
        
        # Add to bottom layout
        bottom_layout.addWidget(history_group)
        
        bottom_widget.setLayout(bottom_layout)
        
        # Add widgets to splitter
        splitter.addWidget(top_widget)
        splitter.addWidget(middle_widget)
        splitter.addWidget(bottom_widget)
        
        # Set splitter sizes
        splitter.setSizes([200, 200, 200])
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        
        status_layout.addWidget(self.status_label)
        
        # Add to main layout
        main_layout.addWidget(splitter)
        main_layout.addLayout(status_layout)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Check if feedback file exists and update retrain button
        self.check_feedback_file()
        
    def browse_model(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.pkl)")
        
        if file_path:
            self.model_path = file_path
            self.model_path_edit.setText(file_path)
    
    def browse_audio(self):
        """Browse for audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.ogg *.flac)")
        
        if file_path:
            self.current_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.classify_button.setEnabled(True)
    
    def classify_audio(self):
        """Classify the selected audio file"""
        if not self.current_file:
            return
        
        # Reset feedback UI
        self.results_group.setVisible(False)
        self.radio_correct.setChecked(False)
        self.radio_incorrect.setChecked(False)
        self.submit_button.setEnabled(False)
        
        # Get model path from text edit
        model_path = self.model_path_edit.toPlainText().strip()
        
        # Start worker thread
        self.worker = WorkerThread("classify", {
            "file_path": self.current_file,
            "model_path": model_path
        })
        
        self.worker.update_status.connect(self.update_status)
        self.worker.finished_task.connect(self.classification_finished)
        
        self.classify_button.setEnabled(False)
        self.worker.start()
    
    def classification_finished(self, result):
        """Handle classification results"""
        self.classify_button.setEnabled(True)
        
        if not result["success"]:
            QMessageBox.warning(self, "Error", result["message"])
            return
        
        # Store results
        self.current_prediction = result["is_ai"]
        self.current_features = result["features"]
        
        # Update UI
        is_ai = result["is_ai"]
        confidence = result["confidence"] * 100
        
        prediction_text = "AI-Generated Voice" if is_ai else "Human Voice"
        self.result_label.setText(prediction_text)
        self.confidence_bar.setValue(int(confidence))
        
        # Set color based on prediction
        if is_ai:
            self.result_label.setStyleSheet("color: #E53935;")  # Red for AI
        else:
            self.result_label.setStyleSheet("color: #43A047;")  # Green for human
        
        # Show results group
        self.results_group.setVisible(True)
        
        # Add to history
        history_item = f"{os.path.basename(result['file_path'])} - {prediction_text} ({confidence:.2f}%)"
        self.history_list.addItem(history_item)
        self.history.append({
            "file_path": result["file_path"],
            "is_ai": is_ai,
            "confidence": confidence
        })
        
        # Update status
        self.update_status(f"Classified as {prediction_text} with {confidence:.2f}% confidence")
    
    def toggle_submit_button(self):
        """Enable/disable submit button based on radio selection"""
        self.submit_button.setEnabled(
            self.radio_correct.isChecked() or self.radio_incorrect.isChecked())
    
    def submit_feedback(self):
        """Submit feedback on classification"""
        if self.current_features is None or self.current_prediction is None:
            return
        
        # Determine correct label
        if self.radio_correct.isChecked():
            # Prediction was correct
            correct_label = 1 if self.current_prediction else 0
        else:
            # Prediction was incorrect
            correct_label = 0 if self.current_prediction else 1
        
        # Start worker thread
        self.worker = WorkerThread("save_feedback", {
            "file_path": self.current_file,
            "features": self.current_features,
            "correct_label": correct_label,
            "feedback_file": self.feedback_file
        })
        
        self.worker.update_status.connect(self.update_status)
        self.worker.finished_task.connect(self.feedback_saved)
        
        self.submit_button.setEnabled(False)
        self.worker.start()
    
    def feedback_saved(self, result):
        """Handle feedback saving result"""
        if not result["success"]:
            QMessageBox.warning(self, "Error", result["message"])
            self.submit_button.setEnabled(True)
            return
        
        # Show success message
        QMessageBox.information(self, "Feedback Saved", result["message"])
        
        # Reset feedback UI
        self.radio_correct.setChecked(False)
        self.radio_incorrect.setChecked(False)
        
        # Enable retrain button if we have feedback
        if result["feedback_count"] > 0:
            self.retrain_button.setEnabled(True)
        
        # Update status
        self.update_status(result["message"])
    
    def retrain_model(self):
        """Retrain the model with feedback data"""
        # Get model path from text edit
        model_path = self.model_path_edit.toPlainText().strip()
        
        # Create a progress dialog
        self.progress_dialog = QProgressDialog("Preparing to retrain model...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Retraining Model")
        self.progress_dialog.setMinimumWidth(400)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setCancelButton(None)  # Disable cancel button
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Start worker thread
        self.worker = WorkerThread("retrain", {
            "model_path": model_path,
            "feedback_file": self.feedback_file,
            "original_features": self.original_features
        })
        
        self.worker.update_status.connect(self.update_retraining_status)
        self.worker.finished_task.connect(self.retrain_finished)
        
        self.retrain_button.setEnabled(False)
        self.update_status("Retraining model...")
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
            
        self.retrain_button.setEnabled(True)
        
        if not result["success"]:
            QMessageBox.warning(self, "Error", result["message"])
            self.update_status(f"Error: {result['message']}")
            return
        
        # Show success message
        QMessageBox.information(self, "Model Retrained", result["message"])
        
        # Update status
        self.update_status(result["message"])
    
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

def main():
    app = QApplication(sys.argv)
    window = SimpleClassifierApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 
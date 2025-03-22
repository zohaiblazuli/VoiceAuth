import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QFileDialog, QLabel, QProgressBar, 
                           QWidget, QTabWidget, QTextEdit, QComboBox, QSpinBox,
                           QDoubleSpinBox, QCheckBox, QGroupBox, QGridLayout,
                           QSplitter, QFrame, QMessageBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
import json
import joblib

from feature_extraction import extract_features, process_dataset, save_features, load_features
from model import VoiceClassifier, train_model_from_features

class WorkerThread(QThread):
    """Worker thread for long-running operations"""
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)
    finished_task = pyqtSignal(dict)
    
    def __init__(self, task, params):
        super().__init__()
        self.task = task
        self.params = params
        
    def run(self):
        try:
            if self.task == "extract_features":
                self.update_status.emit("Extracting features from dataset...")
                dataset_path = self.params.get("dataset_path")
                ai_folder = self.params.get("ai_folder")
                human_folder = self.params.get("human_folder")
                output_path = self.params.get("output_path")
                n_jobs = self.params.get("n_jobs", -1)
                
                # Process dataset
                features_df = process_dataset(
                    dataset_path=dataset_path,
                    ai_folder=ai_folder,
                    human_folder=human_folder,
                    n_jobs=n_jobs
                )
                
                # Save features
                save_features(features_df, output_path)
                
                # Return results
                self.finished_task.emit({
                    "success": True,
                    "features_path": output_path,
                    "message": f"Features extracted and saved to {output_path}"
                })
                
            elif self.task == "train_model":
                self.update_status.emit("Training model...")
                features_path = self.params.get("features_path")
                model_path = self.params.get("model_path")
                history_path = self.params.get("history_path")
                use_feature_selection = self.params.get("use_feature_selection", True)
                n_jobs = self.params.get("n_jobs", -1)
                
                # Train model
                classifier, metrics = train_model_from_features(
                    features_path=features_path,
                    model_output_path=model_path,
                    history_path=history_path,
                    use_feature_selection=use_feature_selection,
                    n_jobs=n_jobs
                )
                
                # Return results
                self.finished_task.emit({
                    "success": True,
                    "model_path": model_path,
                    "metrics": metrics,
                    "message": f"Model trained and saved to {model_path}"
                })
                
            elif self.task == "classify_audio":
                self.update_status.emit("Classifying audio file...")
                file_path = self.params.get("file_path")
                model_path = self.params.get("model_path")
                
                # Extract features
                features = extract_features(file_path)
                
                # Load model
                classifier = VoiceClassifier()
                classifier.load_model(model_path)
                
                # Make prediction
                predictions, probabilities = classifier.predict(features)
                
                # Determine result
                is_ai = bool(predictions[0])
                confidence = probabilities[0][1] if is_ai else probabilities[0][0]
                
                # Return results
                self.finished_task.emit({
                    "success": True,
                    "is_ai": is_ai,
                    "confidence": confidence,
                    "features": features,
                    "file_path": file_path,
                    "message": f"Classification complete: {'AI-generated' if is_ai else 'Human'} voice"
                })
                
            elif self.task == "update_model":
                self.update_status.emit("Updating model with feedback...")
                
                features = self.params.get("features")
                correct_label = self.params.get("correct_label")
                original_label = self.params.get("original_label")
                model_path = self.params.get("model_path")
                feedback_file = self.params.get("feedback_file")
                file_path = self.params.get("file_path")
                force_retrain = self.params.get("force_retrain", False)
                
                # If not force_retrain and we have new feedback to add
                if not force_retrain and features is not None and correct_label is not None:
                    # Save feedback to CSV file for future retraining
                    feedback_dir = os.path.dirname(feedback_file)
                    os.makedirs(feedback_dir, exist_ok=True)
                    
                    # Check if feedback file exists
                    if os.path.exists(feedback_file):
                        # Load existing feedback data
                        feedback_df = pd.read_csv(feedback_file)
                    else:
                        # Create new feedback dataframe with same structure as features
                        num_features = len(features)
                        feature_names = [f'feature_{i}' for i in range(num_features)]
                        feedback_df = pd.DataFrame(columns=feature_names + ['label', 'file_path'])
                    
                    # Create a new row with the features and correct label
                    new_row = pd.DataFrame([np.append(features, [correct_label, os.path.basename(file_path)])], 
                                         columns=feedback_df.columns)
                    
                    # Append to feedback dataframe
                    feedback_df = pd.concat([feedback_df, new_row], ignore_index=True)
                    
                    # Save updated feedback data
                    feedback_df.to_csv(feedback_file, index=False)
                else:
                    # Just load existing feedback data for forced retraining
                    if os.path.exists(feedback_file):
                        feedback_df = pd.read_csv(feedback_file)
                    else:
                        self.finished_task.emit({
                            "success": False,
                            "message": "No feedback data available for retraining."
                        })
                        return
                
                # Check if we have enough feedback samples to retrain or force_retrain is True
                if force_retrain or len(feedback_df) >= 5:  # Arbitrary threshold, can be adjusted
                    self.update_status.emit("Retraining model with feedback data...")
                    
                    # Load original training data
                    original_features_path = os.path.splitext(model_path)[0] + "_features.csv"
                    if os.path.exists(original_features_path):
                        original_df = pd.read_csv(original_features_path)
                        
                        # Combine original data with feedback data
                        combined_df = pd.concat([original_df, feedback_df], ignore_index=True)
                    else:
                        combined_df = feedback_df
                    
                    # Save combined features temporarily
                    temp_features_path = os.path.splitext(model_path)[0] + "_combined_features.csv"
                    combined_df.to_csv(temp_features_path, index=False)
                    
                    # Retrain model
                    classifier = VoiceClassifier()
                    metrics = classifier.train(combined_df, use_feature_selection=True)
                    
                    # Save updated model
                    classifier.save_model(model_path, os.path.splitext(model_path)[0] + "_history.json")
                    
                    self.finished_task.emit({
                        "success": True,
                        "model_updated": True,
                        "message": f"Model updated with feedback and retrained (accuracy: {metrics['accuracy']:.4f})"
                    })
                elif not force_retrain:
                    self.finished_task.emit({
                        "success": True,
                        "model_updated": False,
                        "message": f"Feedback saved. Need {5 - len(feedback_df)} more samples to auto-retrain model."
                    })
                
        except Exception as e:
            self.update_status.emit(f"Error: {str(e)}")
            self.finished_task.emit({
                "success": False,
                "message": f"Error: {str(e)}"
            })

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for plotting"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        
class VoiceClassifierApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Classifier")
        self.setMinimumSize(800, 600)
        
        # Initialize variables
        self.features_path = None
        self.model_path = None
        self.current_audio_path = None
        self.current_features = None
        self.current_prediction = None
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create tabs
        self.tabs = QTabWidget()
        
        # Create tabs for different functionalities
        self.setup_preprocessing_tab()
        self.setup_training_tab()
        self.setup_classification_tab()
        
        # Add tabs to the tab widget
        self.tabs.addTab(self.preprocessing_tab, "Feature Extraction")
        self.tabs.addTab(self.training_tab, "Model Training")
        self.tabs.addTab(self.classification_tab, "Voice Classification")
        
        # Create status bar
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.progress_bar)
        
        # Add widgets to main layout
        main_layout.addWidget(self.tabs)
        main_layout.addLayout(status_layout)
        
        # Set layout to main widget
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
    def setup_preprocessing_tab(self):
        self.preprocessing_tab = QWidget()
        layout = QVBoxLayout()
        
        # Dataset selection
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QGridLayout()
        
        # Dataset path
        self.dataset_path_label = QLabel("Dataset path:")
        self.dataset_path_edit = QTextEdit()
        self.dataset_path_edit.setMaximumHeight(50)
        self.dataset_path_edit.setText(r"C:\Project Resources\dataset\raw")  # Preset the dataset path
        self.dataset_path_button = QPushButton("Browse...")
        self.dataset_path_button.clicked.connect(self.browse_dataset)
        
        # Dataset structure
        self.ai_folder_label = QLabel("AI-generated folder name:")
        self.ai_folder_edit = QTextEdit("ai")  # Changed from "ai_generated" to "ai"
        self.ai_folder_edit.setMaximumHeight(30)
        
        self.human_folder_label = QLabel("Human folder name:")
        self.human_folder_edit = QTextEdit("human")
        self.human_folder_edit.setMaximumHeight(30)
        
        # Add widgets to dataset layout
        dataset_layout.addWidget(self.dataset_path_label, 0, 0)
        dataset_layout.addWidget(self.dataset_path_edit, 0, 1)
        dataset_layout.addWidget(self.dataset_path_button, 0, 2)
        dataset_layout.addWidget(self.ai_folder_label, 1, 0)
        dataset_layout.addWidget(self.ai_folder_edit, 1, 1)
        dataset_layout.addWidget(self.human_folder_label, 2, 0)
        dataset_layout.addWidget(self.human_folder_edit, 2, 1)
        
        dataset_group.setLayout(dataset_layout)
        
        # Feature extraction options
        feature_group = QGroupBox("Feature Extraction Options")
        feature_layout = QGridLayout()
        
        self.parallel_label = QLabel("Number of parallel jobs:")
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setMinimum(-1)
        self.parallel_spin.setMaximum(16)
        self.parallel_spin.setValue(-1)
        self.parallel_spin.setSpecialValueText("Auto")
        
        self.output_path_label = QLabel("Output features path:")
        self.output_path_edit = QTextEdit("output/features.csv")
        self.output_path_edit.setMaximumHeight(30)
        self.output_path_button = QPushButton("Browse...")
        self.output_path_button.clicked.connect(self.browse_output_path)
        
        # Add widgets to feature layout
        feature_layout.addWidget(self.parallel_label, 0, 0)
        feature_layout.addWidget(self.parallel_spin, 0, 1)
        feature_layout.addWidget(self.output_path_label, 1, 0)
        feature_layout.addWidget(self.output_path_edit, 1, 1)
        feature_layout.addWidget(self.output_path_button, 1, 2)
        
        feature_group.setLayout(feature_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.extract_button = QPushButton("Extract Features")
        self.extract_button.setMinimumHeight(40)
        self.extract_button.clicked.connect(self.start_feature_extraction)
        
        button_layout.addWidget(self.extract_button)
        
        # Add everything to main layout
        layout.addWidget(dataset_group)
        layout.addWidget(feature_group)
        layout.addLayout(button_layout)
        layout.addStretch()
        
        self.preprocessing_tab.setLayout(layout)
        
    def setup_training_tab(self):
        self.training_tab = QWidget()
        layout = QVBoxLayout()
        
        # Input features
        input_group = QGroupBox("Input Features")
        input_layout = QGridLayout()
        
        self.features_path_label = QLabel("Features path:")
        self.features_path_edit = QTextEdit("output/features.csv")
        self.features_path_edit.setMaximumHeight(30)
        self.features_path_button = QPushButton("Browse...")
        self.features_path_button.clicked.connect(self.browse_features_path)
        
        input_layout.addWidget(self.features_path_label, 0, 0)
        input_layout.addWidget(self.features_path_edit, 0, 1)
        input_layout.addWidget(self.features_path_button, 0, 2)
        
        input_group.setLayout(input_layout)
        
        # Training options
        training_group = QGroupBox("Training Options")
        training_layout = QGridLayout()
        
        self.feature_selection_check = QCheckBox("Use feature selection")
        self.feature_selection_check.setChecked(True)
        
        self.training_parallel_label = QLabel("Number of parallel jobs:")
        self.training_parallel_spin = QSpinBox()
        self.training_parallel_spin.setMinimum(-1)
        self.training_parallel_spin.setMaximum(16)
        self.training_parallel_spin.setValue(-1)
        self.training_parallel_spin.setSpecialValueText("Auto")
        
        self.model_path_label = QLabel("Output model path:")
        self.model_path_edit = QTextEdit("output/models/voice_classifier.pkl")
        self.model_path_edit.setMaximumHeight(30)
        self.model_path_button = QPushButton("Browse...")
        self.model_path_button.clicked.connect(self.browse_model_path)
        
        training_layout.addWidget(self.feature_selection_check, 0, 0, 1, 2)
        training_layout.addWidget(self.training_parallel_label, 1, 0)
        training_layout.addWidget(self.training_parallel_spin, 1, 1)
        training_layout.addWidget(self.model_path_label, 2, 0)
        training_layout.addWidget(self.model_path_edit, 2, 1)
        training_layout.addWidget(self.model_path_button, 2, 2)
        
        training_group.setLayout(training_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Train Model")
        self.train_button.setMinimumHeight(40)
        self.train_button.clicked.connect(self.start_model_training)
        
        button_layout.addWidget(self.train_button)
        
        # Results display
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        # Add everything to main layout
        layout.addWidget(input_group)
        layout.addWidget(training_group)
        layout.addLayout(button_layout)
        layout.addWidget(results_group)
        
        self.training_tab.setLayout(layout)
        
    def setup_classification_tab(self):
        self.classification_tab = QWidget()
        layout = QVBoxLayout()
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QGridLayout()
        
        self.classify_model_path_label = QLabel("Model path:")
        self.classify_model_path_edit = QTextEdit("output/models/voice_classifier.pkl")
        self.classify_model_path_edit.setMaximumHeight(30)
        self.classify_model_path_button = QPushButton("Browse...")
        self.classify_model_path_button.clicked.connect(self.browse_classify_model_path)
        
        model_layout.addWidget(self.classify_model_path_label, 0, 0)
        model_layout.addWidget(self.classify_model_path_edit, 0, 1)
        model_layout.addWidget(self.classify_model_path_button, 0, 2)
        
        model_group.setLayout(model_layout)
        
        # Audio file selection
        audio_group = QGroupBox("Audio File")
        audio_layout = QGridLayout()
        
        self.audio_path_label = QLabel("Audio file path:")
        self.audio_path_edit = QTextEdit("")
        self.audio_path_edit.setMaximumHeight(30)
        self.audio_path_button = QPushButton("Browse...")
        self.audio_path_button.clicked.connect(self.browse_audio_file)
        
        audio_layout.addWidget(self.audio_path_label, 0, 0)
        audio_layout.addWidget(self.audio_path_edit, 0, 1)
        audio_layout.addWidget(self.audio_path_button, 0, 2)
        
        audio_group.setLayout(audio_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.classify_button = QPushButton("Classify Audio")
        self.classify_button.setMinimumHeight(40)
        self.classify_button.clicked.connect(self.start_classification)
        
        button_layout.addWidget(self.classify_button)
        
        # Results display
        results_display = QGroupBox("Classification Results")
        results_layout = QVBoxLayout()
        
        self.classification_result_label = QLabel("")
        self.classification_result_label.setAlignment(Qt.AlignCenter)
        self.classification_result_label.setFont(QFont("Arial", 14, QFont.Bold))
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMaximum(100)
        self.confidence_bar.setMinimum(0)
        self.confidence_bar.setValue(0)
        
        results_layout.addWidget(self.classification_result_label)
        results_layout.addWidget(self.confidence_bar)
        
        # Feedback mechanism
        feedback_group = QGroupBox("Provide Feedback")
        feedback_layout = QVBoxLayout()
        
        self.feedback_label = QLabel("Was the prediction correct?")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        
        # Add radio buttons for feedback
        feedback_radio_layout = QHBoxLayout()
        
        self.feedback_correct = QRadioButton("Yes, prediction was correct")
        self.feedback_incorrect = QRadioButton("No, this is actually:")
        
        # Group radio buttons
        self.feedback_group = QButtonGroup()
        self.feedback_group.addButton(self.feedback_correct)
        self.feedback_group.addButton(self.feedback_incorrect)
        
        feedback_radio_layout.addWidget(self.feedback_correct)
        feedback_radio_layout.addWidget(self.feedback_incorrect)
        
        # Add dropdown for corrected class when prediction is wrong
        actual_class_layout = QHBoxLayout()
        self.actual_class_label = QLabel("Actual class:")
        self.actual_class_combo = QComboBox()
        self.actual_class_combo.addItem("Human")
        self.actual_class_combo.addItem("AI-generated")
        self.actual_class_combo.setEnabled(False)
        
        actual_class_layout.addWidget(self.actual_class_label)
        actual_class_layout.addWidget(self.actual_class_combo)
        
        # Connect feedback radio buttons to enable/disable class dropdown
        self.feedback_incorrect.toggled.connect(lambda checked: self.actual_class_combo.setEnabled(checked))
        
        # Submit feedback button
        self.submit_feedback_button = QPushButton("Submit Feedback")
        self.submit_feedback_button.clicked.connect(self.submit_feedback)
        self.submit_feedback_button.setEnabled(False)
        
        # Force retrain button
        retrain_layout = QHBoxLayout()
        self.force_retrain_button = QPushButton("Force Retrain Model")
        self.force_retrain_button.clicked.connect(self.force_retrain_model)
        self.force_retrain_button.setEnabled(False)
        retrain_layout.addWidget(self.force_retrain_button)
        
        # Add widgets to feedback layout
        feedback_layout.addWidget(self.feedback_label)
        feedback_layout.addLayout(feedback_radio_layout)
        feedback_layout.addLayout(actual_class_layout)
        feedback_layout.addWidget(self.submit_feedback_button)
        feedback_layout.addLayout(retrain_layout)
        
        feedback_group.setLayout(feedback_layout)
        feedback_group.setVisible(False)
        self.feedback_group_widget = feedback_group
        
        results_layout.addWidget(feedback_group)
        results_display.setLayout(results_layout)
        
        # Add everything to main layout
        layout.addWidget(model_group)
        layout.addWidget(audio_group)
        layout.addLayout(button_layout)
        layout.addWidget(results_display)
        layout.addStretch()
        
        self.classification_tab.setLayout(layout)
    
    def browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if folder:
            self.dataset_path_edit.setText(folder)
    
    def browse_output_path(self):
        file, _ = QFileDialog.getSaveFileName(self, "Save Features File", "", "CSV Files (*.csv)")
        if file:
            self.output_path_edit.setText(file)
    
    def browse_features_path(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Features File", "", "CSV Files (*.csv)")
        if file:
            self.features_path_edit.setText(file)
            self.features_path = file
    
    def browse_model_path(self):
        file, _ = QFileDialog.getSaveFileName(self, "Save Model File", "", "Pickle Files (*.pkl)")
        if file:
            self.model_path_edit.setText(file)
            self.model_path = file
    
    def browse_classify_model_path(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Pickle Files (*.pkl)")
        if file:
            self.classify_model_path_edit.setText(file)
    
    def browse_audio_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3 *.ogg *.flac)")
        if file:
            self.audio_path_edit.setText(file)
            self.current_audio_path = file
    
    def start_feature_extraction(self):
        dataset_path = self.dataset_path_edit.toPlainText().strip()
        ai_folder = self.ai_folder_edit.toPlainText().strip()
        human_folder = self.human_folder_edit.toPlainText().strip()
        output_path = self.output_path_edit.toPlainText().strip()
        n_jobs = self.parallel_spin.value()
        
        if not os.path.exists(dataset_path):
            QMessageBox.warning(self, "Error", "Dataset path does not exist.")
            return
        
        if not os.path.exists(os.path.join(dataset_path, ai_folder)):
            QMessageBox.warning(self, "Error", f"AI folder '{ai_folder}' does not exist in dataset path.")
            return
        
        if not os.path.exists(os.path.join(dataset_path, human_folder)):
            QMessageBox.warning(self, "Error", f"Human folder '{human_folder}' does not exist in dataset path.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Start worker thread
        self.worker = WorkerThread("extract_features", {
            "dataset_path": dataset_path,
            "ai_folder": ai_folder,
            "human_folder": human_folder,
            "output_path": output_path,
            "n_jobs": n_jobs
        })
        
        self.worker.update_status.connect(self.update_status)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished_task.connect(self.feature_extraction_finished)
        
        self.extract_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.worker.start()
    
    def start_model_training(self):
        features_path = self.features_path_edit.toPlainText().strip()
        model_path = self.model_path_edit.toPlainText().strip()
        use_feature_selection = self.feature_selection_check.isChecked()
        n_jobs = self.training_parallel_spin.value()
        
        if not os.path.exists(features_path):
            QMessageBox.warning(self, "Error", "Features file does not exist.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Start worker thread
        self.worker = WorkerThread("train_model", {
            "features_path": features_path,
            "model_path": model_path,
            "history_path": f"{os.path.splitext(model_path)[0]}_history.json",
            "use_feature_selection": use_feature_selection,
            "n_jobs": n_jobs
        })
        
        self.worker.update_status.connect(self.update_status)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished_task.connect(self.model_training_finished)
        
        self.train_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.worker.start()
        
        # Save a copy of the features for potential retraining
        features_backup_path = f"{os.path.splitext(model_path)[0]}_features.csv"
        if os.path.exists(features_path) and not os.path.exists(features_backup_path):
            import shutil
            shutil.copy(features_path, features_backup_path)
    
    def start_classification(self):
        model_path = self.classify_model_path_edit.toPlainText().strip()
        audio_path = self.audio_path_edit.toPlainText().strip()
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return
        
        if not os.path.exists(audio_path):
            QMessageBox.warning(self, "Error", "Audio file does not exist.")
            return
        
        # Reset feedback UI
        self.feedback_group_widget.setVisible(False)
        self.feedback_correct.setChecked(False)
        self.feedback_incorrect.setChecked(False)
        self.actual_class_combo.setEnabled(False)
        self.submit_feedback_button.setEnabled(False)
        
        # Start worker thread
        self.worker = WorkerThread("classify_audio", {
            "model_path": model_path,
            "file_path": audio_path
        })
        
        self.worker.update_status.connect(self.update_status)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished_task.connect(self.classification_finished)
        
        self.classify_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.worker.start()
    
    def submit_feedback(self):
        """Handle user feedback submission"""
        if not self.current_features is not None or self.current_prediction is None:
            QMessageBox.warning(self, "Error", "No classification available for feedback.")
            return
        
        model_path = self.classify_model_path_edit.toPlainText().strip()
        
        # Determine correct label based on feedback
        if self.feedback_correct.isChecked():
            # Prediction was correct
            correct_label = 1 if self.current_prediction else 0
        else:
            # Prediction was incorrect, get the corrected class
            correct_label = 1 if self.actual_class_combo.currentText() == "AI-generated" else 0
        
        # Start worker thread to update model
        self.worker = WorkerThread("update_model", {
            "features": self.current_features,
            "correct_label": correct_label,
            "original_label": 1 if self.current_prediction else 0,
            "model_path": model_path,
            "feedback_file": os.path.join(os.path.dirname(model_path), "feedback_data.csv"),
            "file_path": self.current_audio_path
        })
        
        self.worker.update_status.connect(self.update_status)
        self.worker.finished_task.connect(self.model_update_finished)
        
        self.submit_feedback_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.worker.start()
    
    def force_retrain_model(self):
        """Force model retraining with current feedback data"""
        model_path = self.classify_model_path_edit.toPlainText().strip()
        feedback_file = os.path.join(os.path.dirname(model_path), "feedback_data.csv")
        
        if not os.path.exists(feedback_file):
            QMessageBox.warning(self, "Error", "No feedback data available for retraining.")
            return
        
        feedback_df = pd.read_csv(feedback_file)
        if len(feedback_df) == 0:
            QMessageBox.warning(self, "Error", "No feedback data available for retraining.")
            return
        
        # Start worker thread for forced retraining
        self.update_status(f"Forcing model retraining with {len(feedback_df)} feedback samples...")
        
        # Load original training data
        original_features_path = os.path.splitext(model_path)[0] + "_features.csv"
        
        self.worker = WorkerThread("update_model", {
            "features": None,  # Not adding new features, just retraining
            "correct_label": None,
            "original_label": None,
            "model_path": model_path,
            "feedback_file": feedback_file,
            "file_path": None,
            "force_retrain": True  # Flag to force retraining
        })
        
        self.worker.update_status.connect(self.update_status)
        self.worker.finished_task.connect(self.model_update_finished)
        
        self.force_retrain_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.worker.start()
    
    def update_status(self, message):
        self.status_label.setText(message)
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def feature_extraction_finished(self, result):
        self.extract_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        if result["success"]:
            QMessageBox.information(self, "Success", result["message"])
            self.features_path = result["features_path"]
            self.features_path_edit.setText(self.features_path)
            self.tabs.setCurrentIndex(1)  # Switch to training tab
        else:
            QMessageBox.warning(self, "Error", result["message"])
    
    def model_training_finished(self, result):
        self.train_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        if result["success"]:
            QMessageBox.information(self, "Success", result["message"])
            self.model_path = result["model_path"]
            self.classify_model_path_edit.setText(self.model_path)
            
            # Display metrics
            metrics = result["metrics"]
            results_text = f"Training Results:\n\n"
            results_text += f"Accuracy: {metrics['accuracy']:.4f}\n"
            results_text += f"Precision: {metrics['precision']:.4f}\n"
            results_text += f"Recall: {metrics['recall']:.4f}\n"
            results_text += f"F1 Score: {metrics['f1']:.4f}\n"
            
            self.results_text.setText(results_text)
            
            # Switch to classification tab
            self.tabs.setCurrentIndex(2)
        else:
            QMessageBox.warning(self, "Error", result["message"])
    
    def classification_finished(self, result):
        self.classify_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        if result["success"]:
            is_ai = result["is_ai"]
            confidence = result["confidence"] * 100
            
            # Store current features and prediction for feedback
            self.current_features = result["features"]
            self.current_prediction = is_ai
            
            # Update result display
            self.classification_result_label.setText("AI-Generated Voice" if is_ai else "Human Voice")
            self.confidence_bar.setValue(int(confidence))
            
            # Change color based on classification
            if is_ai:
                self.classification_result_label.setStyleSheet("color: #E53935;")  # Red for AI
            else:
                self.classification_result_label.setStyleSheet("color: #43A047;")  # Green for human
            
            # Show feedback options
            self.feedback_group_widget.setVisible(True)
            self.submit_feedback_button.setEnabled(True)
            
            # Set the opposite class in the dropdown
            self.actual_class_combo.setCurrentIndex(0 if is_ai else 1)
            
            # Check if there's feedback data to enable force retrain button
            model_path = self.classify_model_path_edit.toPlainText().strip()
            feedback_file = os.path.join(os.path.dirname(model_path), "feedback_data.csv")
            if os.path.exists(feedback_file):
                try:
                    feedback_df = pd.read_csv(feedback_file)
                    if len(feedback_df) > 0:
                        self.force_retrain_button.setEnabled(True)
                except:
                    self.force_retrain_button.setEnabled(False)
            else:
                self.force_retrain_button.setEnabled(False)
        else:
            QMessageBox.warning(self, "Error", result["message"])
    
    def model_update_finished(self, result):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        if result["success"]:
            QMessageBox.information(self, "Feedback Submitted", result["message"])
            
            # Enable force retrain button if there's feedback data
            model_path = self.classify_model_path_edit.toPlainText().strip()
            feedback_file = os.path.join(os.path.dirname(model_path), "feedback_data.csv")
            if os.path.exists(feedback_file):
                try:
                    feedback_df = pd.read_csv(feedback_file)
                    if len(feedback_df) > 0:
                        self.force_retrain_button.setEnabled(True)
                except:
                    pass
            
            # Hide feedback section after submission
            self.feedback_group_widget.setVisible(False)
            
            # Reset feedback UI elements
            self.feedback_correct.setChecked(False)
            self.feedback_incorrect.setChecked(False)
            self.submit_feedback_button.setEnabled(False)
        else:
            QMessageBox.warning(self, "Error", result["message"])
            self.force_retrain_button.setEnabled(True)  # Re-enable in case of error


def main():
    app = QApplication(sys.argv)
    
    # Create main window
    window = VoiceClassifierApp()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
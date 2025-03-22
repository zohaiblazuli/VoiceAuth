import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os
import json
import joblib
import time
from tqdm import tqdm

class VoiceClassifier:
    def __init__(self, n_jobs=-1, random_state=42):
        """
        Initialize the voice classifier
        
        Parameters:
        -----------
        n_jobs : int
            Number of jobs for parallel processing
        random_state : int
            Random seed for reproducibility
        """
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_selector = None
        self.pipeline = None
        self.training_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'train_time': [],
            'best_params': None
        }
    
    def preprocess_features(self, X):
        """
        Preprocess features with scaling
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
            
        Returns:
        --------
        X_scaled : numpy array
            Scaled feature matrix
        """
        return self.scaler.transform(X)
    
    def train(self, features_df, test_size=0.2, use_feature_selection=True):
        """
        Train the logistic regression model
        
        Parameters:
        -----------
        features_df : pandas DataFrame
            DataFrame with features and labels
        test_size : float
            Proportion of data to use for testing
        use_feature_selection : bool
            Whether to use feature selection
            
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        start_time = time.time()
        
        # Split data into features and labels
        X = features_df.drop(['label', 'file_path'], axis=1, errors='ignore').values
        y = features_df['label'].values
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create pipeline
        if use_feature_selection:
            # Setup feature selection
            base_model = LogisticRegression(solver='liblinear', random_state=self.random_state)
            self.feature_selector = SelectFromModel(estimator=base_model)
            
            # Setup final pipeline
            pipeline_steps = [
                ('feature_selection', self.feature_selector),
                ('classifier', LogisticRegression(random_state=self.random_state, n_jobs=self.n_jobs))
            ]
        else:
            pipeline_steps = [
                ('classifier', LogisticRegression(random_state=self.random_state, n_jobs=self.n_jobs))
            ]
        
        self.pipeline = Pipeline(steps=pipeline_steps)
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear'],
            'classifier__class_weight': [None, 'balanced']
        }
        
        grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=self.n_jobs, verbose=1
        )
        
        # Fit the model with grid search
        print("Training model with grid search...")
        grid_search.fit(X_train_scaled, y_train)
        
        # Get the best model
        self.pipeline = grid_search.best_estimator_
        self.model = self.pipeline.named_steps['classifier']
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Save training history
        train_time = time.time() - start_time
        self.training_history['accuracy'].append(metrics['accuracy'])
        self.training_history['precision'].append(metrics['precision'])
        self.training_history['recall'].append(metrics['recall'])
        self.training_history['f1'].append(metrics['f1'])
        self.training_history['train_time'].append(train_time)
        self.training_history['best_params'] = grid_search.best_params_
        
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        return metrics
    
    def predict(self, features):
        """
        Predict the class of a sample
        
        Parameters:
        -----------
        features : numpy array
            Feature vector or matrix
            
        Returns:
        --------
        predictions : numpy array
            Predicted class labels (0=human, 1=AI)
        probabilities : numpy array
            Class probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet, call train() first")
        
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.pipeline.predict(features_scaled)
        probabilities = self.pipeline.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def save_model(self, model_path, history_path=None):
        """Save the trained model to disk"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet, call train() first")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.pipeline, model_path)
        joblib.dump(self.scaler, f"{os.path.splitext(model_path)[0]}_scaler.pkl")
        
        if history_path:
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=4)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path, scaler_path=None):
        """Load a trained model from disk"""
        if not scaler_path:
            scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl"
        
        self.pipeline = joblib.load(model_path)
        if 'classifier' in self.pipeline.named_steps:
            self.model = self.pipeline.named_steps['classifier']
        else:
            self.model = self.pipeline
            
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        print(f"Model loaded from {model_path}")
    
    def plot_feature_importance(self, feature_names, output_path=None):
        """Plot feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet, call train() first")
        
        if hasattr(self.model, 'coef_'):
            # For logistic regression
            importance = np.abs(self.model.coef_[0])
            
            # If feature selection was used, map back to original features
            if self.feature_selector:
                selected_indices = self.feature_selector.get_support()
                full_importance = np.zeros(len(feature_names))
                full_importance[selected_indices] = importance
                importance = full_importance
            
            # Get top 20 features
            if len(importance) > 20:
                top_indices = np.argsort(importance)[-20:]
                importance = importance[top_indices]
                feature_names = np.array(feature_names)[top_indices]
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance)), importance)
            plt.yticks(range(len(importance)), feature_names)
            plt.xlabel('Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path)
                print(f"Feature importance plot saved to {output_path}")
            else:
                plt.show()


def train_model_from_features(features_path, model_output_path, history_path=None, 
                             use_feature_selection=True, n_jobs=-1):
    """
    Train a model from pre-extracted features
    
    Parameters:
    -----------
    features_path : str
        Path to the CSV file with extracted features
    model_output_path : str
        Path to save the trained model
    history_path : str, optional
        Path to save the training history
    use_feature_selection : bool
        Whether to use feature selection
    n_jobs : int
        Number of jobs for parallel processing
    
    Returns:
    --------
    classifier : VoiceClassifier
        Trained classifier
    metrics : dict
        Performance metrics
    """
    # Load features
    print(f"Loading features from {features_path}")
    features_df = pd.read_csv(features_path)
    
    # Initialize and train classifier
    classifier = VoiceClassifier(n_jobs=n_jobs)
    metrics = classifier.train(features_df, use_feature_selection=use_feature_selection)
    
    # Save model
    classifier.save_model(model_output_path, history_path)
    
    # Plot feature importance if feature names are available
    if 'file_path' in features_df.columns and 'label' in features_df.columns:
        feature_names = features_df.drop(['file_path', 'label'], axis=1).columns.tolist()
        plot_path = f"{os.path.splitext(model_output_path)[0]}_feature_importance.png"
        classifier.plot_feature_importance(feature_names, plot_path)
    
    return classifier, metrics 
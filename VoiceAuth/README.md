# Voice Classifier Application

A desktop application that uses logistic regression to differentiate between AI-generated and human voices based on audio samples.

## Features

- Feature extraction from audio files (WAV, MP3, OGG, FLAC)
- Logistic regression model training with hyperparameter optimization
- Real-time voice classification
- User-friendly GUI interface
- Hardware acceleration through parallel processing
- Feature selection for better performance
- Visualizations of feature importance
- Feedback mechanism to improve model accuracy over time
- On-demand model retraining with collected feedback

## Installation

1. Install Python 3.8+ if not already installed
2. Clone this repository
3. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### Preparing Your Dataset

Organize your dataset in the following structure:
```
dataset_folder/
├── ai_generated/  (folder containing AI-generated voice samples)
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── human/  (folder containing human voice samples)
    ├── sample1.wav
    ├── sample2.wav
    └── ...
```

### Running the Application

Run the main application:

```
python app.py
```

### Step 1: Feature Extraction

1. In the "Feature Extraction" tab, browse to select your dataset folder
2. Verify that folder names for AI and human samples are correct
3. Set the number of parallel jobs for processing (-1 uses all available cores)
4. Set the output path for the extracted features
5. Click "Extract Features" to start the process

### Step 2: Model Training

1. In the "Model Training" tab, the features file from the previous step should be automatically loaded
2. Select whether to use feature selection
3. Set the number of parallel jobs for training
4. Set the output path for the trained model
5. Click "Train Model" to start training

### Step 3: Voice Classification

1. In the "Voice Classification" tab, the trained model from the previous step should be automatically loaded
2. Browse to select an audio file to classify
3. Click "Classify Audio" to analyze the file
4. View the results showing whether the voice is AI-generated or human, along with confidence level

### Step 4: Provide Feedback

After classification, you can provide feedback to improve the model:

1. Select "Yes, prediction was correct" if the classification was accurate
2. Or select "No, this is actually:" and choose the correct class from the dropdown if the prediction was wrong
3. Click "Submit Feedback" to send your feedback
4. The model will automatically update after collecting sufficient feedback samples (currently set to 5 samples)
5. Alternatively, click "Force Retrain Model" to immediately retrain the model with all collected feedback without waiting for 5 samples

## Technical Details

### Feature Extraction

The application extracts various audio features including:
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Spectral Centroid
- Spectral Contrast
- Spectral Rolloff
- Zero Crossing Rate
- Chroma Features
- Spectral Bandwidth
- Tempo and Beat Features
- Mel Spectrogram

### Model Architecture

- Standard scaling for feature normalization
- Optional feature selection using SelectFromModel
- Logistic regression classifier with hyperparameter tuning
- Grid search for finding optimal parameters
- Performance evaluation using accuracy, precision, recall, and F1-score

### Feedback Mechanism

The application includes an adaptive learning system that:
- Collects user feedback on classification results
- Stores correctly labeled samples
- Automatically retrains the model when sufficient feedback data is collected (5 samples by default)
- Allows manual triggering of retraining with any number of feedback samples
- Combines original training data with feedback data for improved accuracy
- Updates the model in real-time without requiring manual retraining

## Hardware Acceleration

The application uses parallel processing for both feature extraction and model training to utilize all available CPU cores, significantly speeding up processing time for large datasets.

## Requirements

- Python 3.8+
- NumPy
- Pandas
- scikit-learn
- librosa
- matplotlib
- soundfile
- PyQt5
- tqdm
- joblib

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
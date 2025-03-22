# Voice Classifier - Simplified Approach

This is a simplified version of the voice classifier that is designed to handle large datasets with limited memory. It breaks down the processing into three separate steps that can be run independently.

## Memory Error Solution

If you're experiencing memory errors with the full application, use these simplified scripts to process your dataset in batches.

## Step 1: Feature Extraction

Extract features from audio files in small batches to reduce memory usage:

```
python batch_process.py --dataset "C:\Project Resources\dataset\raw" --ai_folder ai --human_folder human --batch_size 100 --output output/features.csv
```

Parameters:
- `--dataset`: Path to the dataset directory (default: "C:\Project Resources\dataset\raw")
- `--ai_folder`: Name of the folder containing AI-generated voices (default: "ai")
- `--human_folder`: Name of the folder containing human voices (default: "human")
- `--batch_size`: Number of files to process in each batch (default: 100, reduce if memory issues persist)
- `--output`: Path to save the extracted features (default: "output/features.csv")

## Step 2: Model Training

Train a logistic regression model using the extracted features:

```
python simple_model.py --features output/features.csv --output output/models/voice_classifier.pkl --test_size 0.2
```

Parameters:
- `--features`: Path to the CSV file containing extracted features (default: "output/features.csv")
- `--output`: Path to save the trained model (default: "output/models/voice_classifier.pkl")
- `--test_size`: Proportion of data to use for testing (default: 0.2)

## Step 3: Classification

Classify a single audio file and provide feedback:

```
python classify.py --audio "path/to/your/audio_file.wav" --model output/models/voice_classifier.pkl
```

Parameters:
- `--audio`: Path to the audio file to classify (required)
- `--model`: Path to the trained model (default: "output/models/voice_classifier.pkl")

## Features

- **Batch Processing**: Process large datasets piece by piece to avoid memory errors
- **Reduced Feature Set**: Uses a minimal set of features to reduce memory requirements
- **Simple Model**: Uses a standard logistic regression model without complex hyperparameter tuning
- **Interactive Feedback**: Allows you to provide feedback on classification results

## Technical Details

### Reduced Feature Set

To minimize memory usage, we extract only the most essential audio features:
- MFCCs (reduced number of coefficients)
- Spectral Centroid (brightness of sound)
- Zero Crossing Rate (noisiness)
- Tempo Feature (rhythm information)
- Simplified Mel Spectrogram (reduced dimensionality)

### Memory Optimization

- Processing files in small batches
- Immediately saving results to disk and clearing memory
- Using simple models with fewer parameters
- Avoiding parallel processing that can consume excessive memory

## System Requirements

- Python 3.6+
- At least 4GB of RAM
- Sufficient disk space for extracted features

## Troubleshooting

If you still experience memory errors:
1. Reduce the batch size (try 50 or even 25)
2. Close other applications while processing
3. Increase your system's page file size
4. Consider processing on a machine with more RAM 
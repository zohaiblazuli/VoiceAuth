import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import argparse

def extract_features(file_path, n_mfcc=20, n_mels=64, frames=64):
    """Extract audio features with reduced memory usage"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Extract minimum essential features to reduce memory usage
        # 1. MFCCs (reduced coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_scaled = np.mean(mfccs.T, axis=0)  # 1D array
        
        # 2. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_scaled = np.mean(centroid.T, axis=0).reshape(-1)  # Ensure 1D
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_scaled = np.mean(zcr.T, axis=0).reshape(-1)  # Ensure 1D
        
        # 4. Tempo feature
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_feature = np.array([float(tempo)])  # Force scalar as 1D array
        except:
            tempo_feature = np.array([0.0])  # Default value if tempo extraction fails
        
        # 5. Simplified Mel Spectrogram (reduced size)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Take mean along time axis to ensure 1D array
        mel_spec_features = np.mean(mel_spec_db, axis=1).reshape(-1)
        
        # Print shapes for debugging
        # print(f"mfccs_scaled shape: {mfccs_scaled.shape}")
        # print(f"centroid_scaled shape: {centroid_scaled.shape}")
        # print(f"zcr_scaled shape: {zcr_scaled.shape}")
        # print(f"tempo_feature shape: {tempo_feature.shape}")
        # print(f"mel_spec_features shape: {mel_spec_features.shape}")
        
        # Combine all features
        feature_vector = np.concatenate((
            mfccs_scaled,  # Already 1D
            centroid_scaled,  # Ensure 1D with reshape(-1)
            zcr_scaled,  # Ensure 1D with reshape(-1)
            tempo_feature,  # Already 1D array with single value
            mel_spec_features  # Ensure 1D with reshape(-1)
        ))
        
        return feature_vector
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def process_files_in_batches(dataset_path, ai_folder='ai', human_folder='human', 
                            batch_size=100, output_file='output/features.csv'):
    """Process dataset in small batches to conserve memory"""
    # Get all file paths
    ai_path = os.path.join(dataset_path, ai_folder)
    human_path = os.path.join(dataset_path, human_folder)
    
    ai_files = [os.path.join(ai_path, f) for f in os.listdir(ai_path) 
                if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    human_files = [os.path.join(human_path, f) for f in os.listdir(human_path) 
                  if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    
    # Create lists for files and labels
    all_files = ai_files + human_files
    all_labels = [1] * len(ai_files) + [0] * len(human_files)
    
    # Initialize empty dataframe for features
    feature_df = None
    feature_columns = None
    
    # Process in batches
    total_batches = (len(all_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        print(f"Processing batch {batch_idx+1}/{total_batches}")
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_files))
        
        batch_files = all_files[start_idx:end_idx]
        batch_labels = all_labels[start_idx:end_idx]
        
        features_list = []
        files_processed = []
        labels_processed = []
        
        for i, (file_path, label) in enumerate(tqdm(zip(batch_files, batch_labels), 
                                                   total=len(batch_files),
                                                   desc=f"Batch {batch_idx+1}")):
            features = extract_features(file_path)
            if features is not None:
                features_list.append(features)
                files_processed.append(file_path)
                labels_processed.append(label)
        
        if not features_list:
            print(f"No features extracted in batch {batch_idx+1}, skipping")
            continue
        
        # Create feature matrix for this batch
        X_batch = np.array(features_list)
        
        # Get feature names if not already determined
        if feature_columns is None:
            num_features = X_batch.shape[1]
            feature_columns = [f'feature_{i}' for i in range(num_features)]
        
        # Create batch dataframe
        batch_df = pd.DataFrame(X_batch, columns=feature_columns)
        batch_df['label'] = labels_processed
        batch_df['file_path'] = files_processed
        
        # If first batch, create the file, otherwise append
        if batch_idx == 0:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            batch_df.to_csv(output_file, index=False)
        else:
            batch_df.to_csv(output_file, mode='a', header=False, index=False)
        
        # Free memory
        del X_batch, batch_df, features_list, files_processed, labels_processed
        
    print(f"All batches processed. Features saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio dataset in batches to avoid memory errors")
    parser.add_argument("--dataset", type=str, default=r"C:\Project Resources\dataset\raw", 
                      help="Path to dataset directory")
    parser.add_argument("--ai_folder", type=str, default="ai", 
                      help="Name of the folder containing AI-generated voices")
    parser.add_argument("--human_folder", type=str, default="human", 
                      help="Name of the folder containing human voices")
    parser.add_argument("--batch_size", type=int, default=100, 
                      help="Number of files to process in each batch")
    parser.add_argument("--output", type=str, default="output/features.csv", 
                      help="Output path for the features CSV file")
    
    args = parser.parse_args()
    
    print("Starting batch processing with the following settings:")
    print(f"Dataset path: {args.dataset}")
    print(f"AI folder: {args.ai_folder}")
    print(f"Human folder: {args.human_folder}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output file: {args.output}")
    print("\nProcessing will be done sequentially to minimize memory usage.")
    
    process_files_in_batches(
        dataset_path=args.dataset,
        ai_folder=args.ai_folder,
        human_folder=args.human_folder,
        batch_size=args.batch_size,
        output_file=args.output
    ) 
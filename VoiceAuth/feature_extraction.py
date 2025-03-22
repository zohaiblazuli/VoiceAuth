import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import os
import joblib
from concurrent.futures import ProcessPoolExecutor

def extract_features(file_path, n_mfcc=40, n_mels=128, frames=128):
    """
    Extract audio features from a file using librosa
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    n_mfcc : int
        Number of MFCC coefficients to extract
    n_mels : int
        Number of Mel bands to use
    frames : int
        Number of frames to consider for fixed-length feature extraction
        
    Returns:
    --------
    feature_vector : numpy array
        Combined features in a 1D array
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Extract features
        # 1. MFCCs (timbre and phonetic content)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # 2. Spectral Centroid (brightness of sound)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_scaled = np.mean(centroid.T, axis=0)
        
        # 3. Spectral Contrast (valleys/peaks in spectrum)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_scaled = np.mean(contrast.T, axis=0)
        
        # 4. Spectral Rolloff (amount of high frequency)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_scaled = np.mean(rolloff.T, axis=0)
        
        # 5. Zero Crossing Rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_scaled = np.mean(zcr.T, axis=0)
        
        # 6. Chroma Features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_scaled = np.mean(chroma.T, axis=0)
        
        # 7. Spectral Bandwidth (width of frequency band)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_scaled = np.mean(bandwidth.T, axis=0)
        
        # 8. Tempo and Beat Features - ensure it's a scalar
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_feature = np.array([tempo])  # Make sure it's a 1D array with single value
        
        # 9. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_resized = librosa.util.fix_length(mel_spec_db, size=frames, axis=1)
        mel_spec_flat = mel_spec_resized.flatten()[:500]  # Limit size
        
        # Combine all features
        feature_vector = np.concatenate((
            mfccs_scaled, 
            centroid_scaled, 
            contrast_scaled, 
            rolloff_scaled,
            zcr_scaled,
            chroma_scaled,
            bandwidth_scaled,
            tempo_feature,  # Changed from [tempo] to tempo_feature
            mel_spec_flat
        ))
        
        return feature_vector
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def process_file(args):
    """Helper function for parallel processing"""
    file_path, label, n_mfcc, n_mels, frames = args
    features = extract_features(file_path, n_mfcc, n_mels, frames)
    return features, label, file_path

def process_dataset(dataset_path, ai_folder='ai_generated', human_folder='human', 
                   n_mfcc=40, n_mels=128, frames=128, n_jobs=-1):
    """
    Process an entire dataset and extract features
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset folder
    ai_folder : str
        Name of the folder containing AI-generated samples
    human_folder : str
        Name of the folder containing human samples
    n_mfcc, n_mels, frames : parameters for feature extraction
    n_jobs : int
        Number of parallel jobs (-1 for all available cores)
        
    Returns:
    --------
    features_df : pandas DataFrame
        DataFrame with extracted features and labels
    """
    # Get all file paths
    ai_path = os.path.join(dataset_path, ai_folder)
    human_path = os.path.join(dataset_path, human_folder)
    
    ai_files = [os.path.join(ai_path, f) for f in os.listdir(ai_path) 
                if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    human_files = [os.path.join(human_path, f) for f in os.listdir(human_path) 
                  if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    
    # Prepare arguments for parallel processing
    ai_args = [(f, 1, n_mfcc, n_mels, frames) for f in ai_files]  # 1 for AI
    human_args = [(f, 0, n_mfcc, n_mels, frames) for f in human_files]  # 0 for human
    all_args = ai_args + human_args
    
    features_list = []
    file_paths = []
    labels = []
    
    # Use parallel processing to extract features
    with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        results = list(tqdm(executor.map(process_file, all_args), 
                          total=len(all_args), 
                          desc="Extracting features"))
    
    # Process results
    for features, label, file_path in results:
        if features is not None:
            features_list.append(features)
            labels.append(label)
            file_paths.append(file_path)
    
    # Create feature matrix
    X = np.array(features_list)
    y = np.array(labels)
    
    # Get feature names
    num_features = X.shape[1]
    feature_names = [f'feature_{i}' for i in range(num_features)]
    
    # Create DataFrame
    features_df = pd.DataFrame(X, columns=feature_names)
    features_df['label'] = y
    features_df['file_path'] = file_paths
    
    return features_df

def save_features(features_df, output_path):
    """Save extracted features to a CSV file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    
def load_features(input_path):
    """Load extracted features from a CSV file"""
    return pd.read_csv(input_path) 
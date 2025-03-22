import os
import pandas as pd
import sys

def check_feature_extraction_progress(output_file="output/features.csv"):
    """Check the progress of batch feature extraction"""
    if not os.path.exists(output_file):
        print("\nFeature extraction has not started or no files have been processed yet.")
        return
    
    try:
        # Read the current feature file
        features_df = pd.read_csv(output_file)
        num_processed = len(features_df)
        
        # Count AI and human samples
        if 'label' in features_df.columns:
            ai_count = sum(features_df['label'] == 1)
            human_count = sum(features_df['label'] == 0)
        else:
            ai_count = "unknown"
            human_count = "unknown"
        
        print("\n===== Feature Extraction Progress =====")
        print(f"Total files processed: {num_processed}")
        print(f"AI samples: {ai_count}")
        print(f"Human samples: {human_count}")
        
        # Rough estimate of file size
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Current features file size: {file_size_mb:.2f} MB")
        
        print("=====================================")
        print("Feature extraction is still in progress. Run this script again to check updated progress.")
        
    except Exception as e:
        print(f"\nError checking progress: {str(e)}")
        print("The features file might be currently being written or is incomplete.")

if __name__ == "__main__":
    output_file = "output/features.csv"
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    check_feature_extraction_progress(output_file) 
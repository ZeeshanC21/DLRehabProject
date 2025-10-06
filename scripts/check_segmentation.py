# scripts/check_segmentation.py
import pandas as pd
from pathlib import Path

def check_segmentation_file(dataset_path="../data/raw/REHAB24-6"):
    """Check the actual column names in Segmentation.csv"""
    
    seg_file = Path(dataset_path) / "Segmentation.csv"
    
    if seg_file.exists():
        df = pd.read_csv(seg_file)
        
        print("Segmentation.csv Analysis")
        print("=" * 40)
        print(f"Total rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Check for possible exercise column names
        possible_exercise_cols = [col for col in df.columns if 'exercise' in col.lower()]
        possible_correctness_cols = [col for col in df.columns if any(word in col.lower() for word in ['correct', 'label', 'class'])]
        
        print(f"\nPossible exercise columns: {possible_exercise_cols}")
        print(f"Possible correctness columns: {possible_correctness_cols}")
        
        # Show unique values for each column
        for col in df.columns:
            unique_vals = df[col].unique()
            if len(unique_vals) < 20:  # Only show if reasonable number of unique values
                print(f"\n{col} unique values: {unique_vals}")
        
    else:
        print(f"Segmentation.csv not found at: {seg_file}")

if __name__ == "__main__":
    check_segmentation_file()
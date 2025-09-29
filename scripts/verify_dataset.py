# scripts/verify_dataset_final.py
import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt

def verify_rehab24_structure(dataset_path="../data/raw/REHAB24-6"):
    """Verify the actual REHAB24-6 structure with correct CSV parsing"""
    
    dataset_path = Path(dataset_path)
    
    print("🔍 REHAB24-6 Dataset Verification")
    print("=" * 50)
    
    # 1. Check exercise directories
    exercises = ['Ex1', 'Ex2', 'Ex3', 'Ex4', 'Ex5', 'Ex6']
    exercise_names = {
        'Ex1': 'Arm Abduction',
        'Ex2': 'Arm VW', 
        'Ex3': 'Push-ups',
        'Ex4': 'Leg Abduction',
        'Ex5': 'Leg Lunge',
        'Ex6': 'Squats'
    }
    
    stats = {}
    
    for ex in exercises:
        print(f"\n📁 {ex} ({exercise_names[ex]}):")
        
        # Check videos
        video_dir = dataset_path / "videos" / ex
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4"))
            print(f"  🎥 Videos: {len(videos)}")
            stats[f'{ex}_videos'] = len(videos)
        
        # Check 2D joints (.npy files)
        joints_2d_dir = dataset_path / "2d_joints" / ex
        if joints_2d_dir.exists():
            joints_2d = list(joints_2d_dir.glob("*.npy"))
            print(f"  🦴 2D Joints: {len(joints_2d)}")
            stats[f'{ex}_2d_joints'] = len(joints_2d)
            
            # Sample a joint file to check shape
            if joints_2d:
                sample_joints = np.load(joints_2d[0])
                print(f"    📊 Sample shape: {sample_joints.shape}")
    
    # 2. Check segmentation file with correct delimiter
    seg_file = dataset_path / "Segmentation.csv"
    if seg_file.exists():
        # Load with semicolon delimiter
        seg_df = pd.read_csv(seg_file, sep=';')
        
        print(f"\n📋 Segmentation Info (Fixed):")
        print(f"  📊 Total rows: {len(seg_df)}")
        print(f"  📊 Columns: {list(seg_df.columns)}")
        
        # Show exercise statistics
        print(f"  📊 Exercises by ID: {seg_df['exercise_id'].value_counts().sort_index().to_dict()}")
        print(f"  📊 Correctness: {seg_df['correctness'].value_counts().to_dict()}")
        
        # Show per-exercise correctness
        print(f"\n  📊 Per-Exercise Correctness:")
        for ex_id in sorted(seg_df['exercise_id'].unique()):
            ex_data = seg_df[seg_df['exercise_id'] == ex_id]
            correct = len(ex_data[ex_data['correctness'] == 1])
            incorrect = len(ex_data[ex_data['correctness'] == 0])
            print(f"    Ex{ex_id} ({exercise_names[f'Ex{ex_id}']}): {correct} correct, {incorrect} incorrect")
        
        # Clean data stats
        clean_data = seg_df[seg_df['mocap_erroneous'] == 0]
        print(f"  📊 Clean data (no mocap errors): {len(clean_data)}/{len(seg_df)} ({len(clean_data)/len(seg_df)*100:.1f}%)")
    
    # 3. Check MediaPipe output
    mp_output = Path("../mediapipe_output")
    if mp_output.exists():
        print(f"\n🤖 MediaPipe Output:")
        frames_dirs = list((mp_output / "frames").glob("PM_*"))
        keypoint_files = list((mp_output / "frames").glob("*_keypoints.npy"))
        print(f"  📁 Frame directories: {len(frames_dirs)}")
        print(f"  🔑 Keypoint files: {len(keypoint_files)}")
        
        if keypoint_files:
            sample_kp = np.load(keypoint_files[0])
            print(f"    📊 MP keypoints shape: {sample_kp.shape}")
    
    print(f"\n✅ Dataset verification complete!")
    return stats

if __name__ == "__main__":
    stats = verify_rehab24_structure()
# scripts/dataset_loader_fixed.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import json

class RehabDataset(Dataset):
    """Unified dataset loader for REHAB24-6 + MediaPipe data"""
    
    def __init__(self, dataset_path="../data/raw/REHAB24-6", 
                 mediapipe_path="../mediapipe_output", 
                 use_mediapipe=True, use_3d=False,
                 exercises=[1, 2, 3, 4, 5, 6]):  # Using exercise_id numbers
        
        self.dataset_path = Path(dataset_path)
        self.mediapipe_path = Path(mediapipe_path)
        self.use_mediapipe = use_mediapipe
        self.use_3d = use_3d
        
        # Load segmentation data with correct delimiter
        self.segmentation = pd.read_csv(self.dataset_path / "Segmentation.csv", sep=';')
        
        # Filter by exercises and clean data
        self.segmentation = self.segmentation[
            self.segmentation['exercise_id'].isin(exercises)
        ].reset_index(drop=True)
        
        # Remove erroneous mocap data
        self.segmentation = self.segmentation[
            self.segmentation['mocap_erroneous'] == 0
        ].reset_index(drop=True)
        
        # Exercise mapping (exercise_id to 0-based index)
        self.exercise_to_id = {ex: i for i, ex in enumerate(sorted(exercises))}
        self.id_to_exercise = {i: ex for ex, i in self.exercise_to_id.items()}
        
        # Exercise names mapping
        self.exercise_names = {
            1: 'Arm Abduction',
            2: 'Arm VW',
            3: 'Push-ups', 
            4: 'Leg Abduction',
            5: 'Leg Lunge',
            6: 'Squats'
        }
        
        print(f"Loaded {len(self.segmentation)} exercise repetitions")
        print(f"Exercises: {[(ex, self.exercise_names[ex]) for ex in exercises]}")
        print(f"Using {'MediaPipe + ' if use_mediapipe else ''}{'3D' if use_3d else '2D'} joints")
    
    def __len__(self):
        return len(self.segmentation)
    
    def __getitem__(self, idx):
        row = self.segmentation.iloc[idx]
        
        # Get basic info
        video_id = row['video_id']
        exercise_id = row['exercise_id']
        first_frame = row['first_frame']
        last_frame = row['last_frame']
        correctness = row['correctness']  # Already 0/1 in the data
        exercise_idx = self.exercise_to_id[exercise_id]  # Convert to 0-based index
        
        # Load joint data
        joints_data = None
        mediapipe_data = None
        
        # 1. Load REHAB24-6 joints
        ex_name = f"Ex{exercise_id}"
        
        if self.use_3d:
            joint_dir = self.dataset_path / "3d_joints" / ex_name
            # Find matching 3D file
            possible_files = list(joint_dir.glob(f"{video_id}*.npy"))
        else:
            joint_dir = self.dataset_path / "2d_joints" / ex_name
            # Find matching 2D file (could be c17 or c18)
            possible_files = list(joint_dir.glob(f"{video_id}*.npy"))
        
        if possible_files:
            joint_file = possible_files[0]  # Take first match
            joints_data = np.load(joint_file)
            
            # Extract the specific repetition frames
            if len(joints_data.shape) == 3:  # (frames, joints, coords)
                # Adjust frame indices (make sure they're within bounds)
                total_frames = joints_data.shape[0]
                start_idx = max(0, min(first_frame, total_frames - 1))
                end_idx = max(start_idx + 1, min(last_frame + 1, total_frames))
                joints_data = joints_data[start_idx:end_idx]
        
        # 2. Load MediaPipe data if requested
        if self.use_mediapipe:
            mp_files = list(self.mediapipe_path.glob(f"frames/{video_id}*_keypoints.npy"))
            if mp_files:
                mediapipe_data = np.load(mp_files[0])
                # Extract repetition frames
                if len(mediapipe_data) > last_frame:
                    start_idx = max(0, first_frame)
                    end_idx = min(len(mediapipe_data), last_frame + 1)
                    mediapipe_data = mediapipe_data[start_idx:end_idx]
        
        # 3. Prepare return data
        sample = {
            'exercise_id': torch.tensor(exercise_idx, dtype=torch.long),
            'correctness': torch.tensor(correctness, dtype=torch.float32),
            'video_id': video_id,
            'original_exercise_id': exercise_id,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'duration': last_frame - first_frame + 1
        }
        
        # Add joint data
        if joints_data is not None:
            # Reshape to (frames, features) for LSTM
            if len(joints_data.shape) == 3:  # (frames, joints, coords)
                joints_data = joints_data.reshape(joints_data.shape[0], -1)
            sample['rehab_joints'] = torch.tensor(joints_data, dtype=torch.float32)
        
        if mediapipe_data is not None:
            # MediaPipe data is already (frames, features)
            sample['mediapipe_joints'] = torch.tensor(mediapipe_data, dtype=torch.float32)
        
        return sample
    
    def get_exercise_stats(self):
        """Get statistics about the dataset"""
        stats = {}
        for ex_id in sorted(self.segmentation['exercise_id'].unique()):
            ex_data = self.segmentation[self.segmentation['exercise_id'] == ex_id]
            stats[f'Ex{ex_id}_{self.exercise_names[ex_id]}'] = {
                'total': len(ex_data),
                'correct': len(ex_data[ex_data['correctness'] == 1]),
                'incorrect': len(ex_data[ex_data['correctness'] == 0]),
                'avg_duration': ex_data.apply(lambda x: x['last_frame'] - x['first_frame'] + 1, axis=1).mean()
            }
        return stats

def create_dataloaders(dataset_path="../data/raw/REHAB24-6", 
                       batch_size=8, 
                       train_split=0.7,
                       val_split=0.15):
    """Create train/val/test dataloaders"""
    
    # Create full dataset
    dataset = RehabDataset(dataset_path)
    
    # Split indices
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Shuffle indices for random split
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Custom collate function to handle variable length sequences
    def collate_fn(batch):
        # Find max sequence length in batch
        max_len_rehab = 0
        max_len_mp = 0
        
        for sample in batch:
            if 'rehab_joints' in sample:
                max_len_rehab = max(max_len_rehab, sample['rehab_joints'].shape[0])
            if 'mediapipe_joints' in sample:
                max_len_mp = max(max_len_mp, sample['mediapipe_joints'].shape[0])
        
        # Pad sequences
        batched = {}
        for key in batch[0].keys():
            if key in ['rehab_joints', 'mediapipe_joints']:
                # Handle sequence data with padding
                sequences = []
                for sample in batch:
                    if key in sample:
                        seq = sample[key]
                        # Pad to max length
                        if key == 'rehab_joints' and max_len_rehab > 0:
                            pad_len = max_len_rehab - seq.shape[0]
                            if pad_len > 0:
                                seq = torch.cat([seq, torch.zeros(pad_len, seq.shape[1])], dim=0)
                        elif key == 'mediapipe_joints' and max_len_mp > 0:
                            pad_len = max_len_mp - seq.shape[0]
                            if pad_len > 0:
                                seq = torch.cat([seq, torch.zeros(pad_len, seq.shape[1])], dim=0)
                        sequences.append(seq)
                    else:
                        # Create zero tensor if data missing
                        if key == 'rehab_joints':
                            sequences.append(torch.zeros(max_len_rehab, 52))  # 26 joints * 2 coords
                        else:
                            sequences.append(torch.zeros(max_len_mp, 99))  # 33 landmarks * 3 coords
                
                if sequences:
                    batched[key] = torch.stack(sequences)
            else:
                # Handle non-sequence data
                if torch.is_tensor(batch[0][key]):
                    batched[key] = torch.stack([sample[key] for sample in batch])
                else:
                    batched[key] = [sample[key] for sample in batch]
        
        return batched
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, dataset

# Test the dataset loader
if __name__ == "__main__":
    # Test dataset loading
    dataset = RehabDataset()
    print(f"Dataset size: {len(dataset)}")
    
    # Print stats
    stats = dataset.get_exercise_stats()
    for ex, stat in stats.items():
        print(f"{ex}: {stat}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"{key}: {value.shape} ({value.dtype})")
        else:
            print(f"{key}: {value}")
    
    # Test dataloader
    print(f"\nTesting dataloader...")
    train_loader, val_loader, test_loader, _ = create_dataloaders(batch_size=4)
    
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {len(value)} items")
        break
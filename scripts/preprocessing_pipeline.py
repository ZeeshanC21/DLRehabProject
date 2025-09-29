# preprocessing_pipeline.py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
import json
import os

class RehabPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.joint_names = self.load_joint_names()
        
    def load_joint_names(self):
        with open("joints_names.txt", 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def normalize_keypoints(self, keypoints_2d):
        """Normalize keypoints to unit scale based on torso"""
        # Use shoulder-hip distance as reference
        left_shoulder = keypoints_2d[:, 1*2:1*2+2]   # joint_1 
        left_hip = keypoints_2d[:, 7*2:7*2+2]        # joint_7
        
        torso_length = np.linalg.norm(left_shoulder - left_hip, axis=1, keepdims=True)
        torso_length[torso_length == 0] = 1  # Avoid division by zero
        
        normalized = keypoints_2d.copy()
        for i in range(0, keypoints_2d.shape[1], 2):
            normalized[:, i:i+2] = keypoints_2d[:, i:i+2] / torso_length
            
        return normalized
    
    def smooth_keypoints(self, keypoints, window_length=7, polyorder=3):
        """Apply Savitzky-Golay filter to smooth keypoint trajectories"""
        if len(keypoints) < window_length:
            return keypoints
            
        smoothed = keypoints.copy()
        for joint_idx in range(keypoints.shape[1]):
            smoothed[:, joint_idx] = savgol_filter(
                keypoints[:, joint_idx], window_length, polyorder
            )
        return smoothed
    
    def compute_joint_angles(self, keypoints_2d):
        """Compute key joint angles for exercise assessment"""
        angles = {}
        
        # Define joint triplets for angle calculation (parent, joint, child)
        angle_joints = {
            'left_elbow': (1, 3, 5),      # shoulder, elbow, wrist
            'right_elbow': (2, 4, 6),
            'left_knee': (7, 9, 11),      # hip, knee, ankle  
            'right_knee': (8, 10, 12),
            'left_shoulder': (7, 1, 3),   # hip, shoulder, elbow
            'right_shoulder': (8, 2, 4),
        }
        
        for angle_name, (p1, p2, p3) in angle_joints.items():
            angles[angle_name] = []
            
            for frame in keypoints_2d:
                # Extract joint coordinates
                joint1 = frame[p1*2:p1*2+2]
                joint2 = frame[p2*2:p2*2+2] 
                joint3 = frame[p3*2:p3*2+2]
                
                # Calculate vectors
                v1 = joint1 - joint2
                v2 = joint3 - joint2
                
                # Calculate angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
                angles[angle_name].append(angle)
        
        return {k: np.array(v) for k, v in angles.items()}
    
    def compute_alignment_error(self, keypoints, reference_template):
        """Compute alignment error against reference 'correct' template"""
        # Use Procrustes analysis or DTW for temporal alignment
        from scipy.spatial.distance import cdist
        
        # Simple MSE alignment error
        if len(keypoints) != len(reference_template):
            # Resample to match lengths
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, 1, len(keypoints))
            new_indices = np.linspace(0, 1, len(reference_template))
            
            aligned_keypoints = []
            for joint_idx in range(keypoints.shape[1]):
                f = interp1d(old_indices, keypoints[:, joint_idx], kind='linear')
                aligned_keypoints.append(f(new_indices))
            keypoints = np.array(aligned_keypoints).T
        
        # Compute per-joint MSE
        mse_per_joint = np.mean((keypoints - reference_template) ** 2, axis=0)
        
        # Overall alignment score (lower is better)
        alignment_error = np.mean(mse_per_joint)
        
        return alignment_error, mse_per_joint
    
    def process_exercise_sequence(self, joints_csv_path, segmentation_row):
        """Process a single exercise repetition"""
        # Load joint data
        df = pd.read_csv(joints_csv_path)
        
        # Extract repetition frames
        start_frame = segmentation_row['StartFrame']
        end_frame = segmentation_row['EndFrame']
        
        rep_data = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)]
        
        # Convert to keypoints array (N_frames x N_joints*2)
        keypoint_cols = [col for col in df.columns if col.startswith('joint_') and ('_x' in col or '_y' in col)]
        keypoints = rep_data[keypoint_cols].values
        
        # Preprocessing pipeline
        normalized = self.normalize_keypoints(keypoints)
        smoothed = self.smooth_keypoints(normalized)
        angles = self.compute_joint_angles(smoothed)
        
        # Package results
        processed_data = {
            'exercise_type': segmentation_row['ExerciseType'],
            'correctness': segmentation_row['Correctness'],
            'raw_keypoints': keypoints,
            'normalized_keypoints': normalized,
            'smoothed_keypoints': smoothed,
            'joint_angles': angles,
            'duration_frames': end_frame - start_frame,
            'metadata': {
                'recording': segmentation_row['RecordingName'],
                'rep_id': segmentation_row['RepetitionID'],
                'direction': segmentation_row['Direction']
            }
        }
        
        return processed_data

# Usage example
def process_dataset():
    processor = RehabPreprocessor("./")
    segmentation = pd.read_csv("Segmentation.csv")
    
    processed_exercises = []
    
    for _, seg_row in segmentation.iterrows()[:10]:  # Process first 10 for demo
        recording_name = seg_row['RecordingName']
        joints_path = f"2d_joints/{recording_name}_2d_joints.csv"
        
        if os.path.exists(joints_path):
            processed = processor.process_exercise_sequence(joints_path, seg_row)
            processed_exercises.append(processed)
            
            # Save individual processed file
            output_path = f"processed/{recording_name}_rep{seg_row['RepetitionID']}.npz"
            os.makedirs("processed", exist_ok=True)
            
            np.savez_compressed(output_path,
                raw_keypoints=processed['raw_keypoints'],
                normalized_keypoints=processed['normalized_keypoints'], 
                smoothed_keypoints=processed['smoothed_keypoints'],
                joint_angles=processed['joint_angles'],
                metadata=processed['metadata']
            )
            
            print(f"Processed: {recording_name} rep {seg_row['RepetitionID']}")
    
    return processed_exercises

# Run preprocessing
if __name__ == "__main__":
    processed_data = process_dataset()
    print(f"Processed {len(processed_data)} exercise repetitions")
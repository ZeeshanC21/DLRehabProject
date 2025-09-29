# conversion_utils.py
import numpy as np
import pandas as pd

# REHAB24-6 joint names (26 joints) to COCO-17 mapping
REHAB24_TO_COCO17_MAP = {
    # Map REHAB24-6 joint indices to COCO-17 indices
    0: 0,    # Head -> nose
    1: 5,    # LeftShoulder -> left_shoulder  
    2: 6,    # RightShoulder -> right_shoulder
    3: 7,    # LeftElbow -> left_elbow
    4: 8,    # RightElbow -> right_elbow
    5: 9,    # LeftWrist -> left_wrist
    6: 10,   # RightWrist -> right_wrist
    7: 11,   # LeftHip -> left_hip
    8: 12,   # RightHip -> right_hip
    9: 13,   # LeftKnee -> left_knee
    10: 14,  # RightKnee -> right_knee
    11: 15,  # LeftAnkle -> left_ankle
    12: 16,  # RightAnkle -> right_ankle
}

def rehab24_to_coco17(rehab_joints_csv):
    """Convert REHAB24-6 2D joints to COCO-17 format"""
    df = pd.read_csv(rehab_joints_csv)
    
    coco_keypoints = []
    for _, row in df.iterrows():
        frame_keypoints = np.zeros(17 * 3)  # 17 joints * (x,y,conf)
        
        for rehab_idx, coco_idx in REHAB24_TO_COCO17_MAP.items():
            if f'joint_{rehab_idx}_x' in df.columns:
                x = row[f'joint_{rehab_idx}_x']
                y = row[f'joint_{rehab_idx}_y']
                conf = 1.0  # REHAB24-6 doesn't have confidence
                
                frame_keypoints[coco_idx * 3] = x
                frame_keypoints[coco_idx * 3 + 1] = y  
                frame_keypoints[coco_idx * 3 + 2] = conf
        
        coco_keypoints.append(frame_keypoints)
    
    return np.array(coco_keypoints)

# MediaPipe 33 to COCO-17 mapping
MEDIAPIPE33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

def mediapipe33_to_coco17(mp_landmarks):
    """Convert MediaPipe 33 landmarks to COCO-17"""
    coco_keypoints = np.zeros(17 * 3)
    
    for i, mp_idx in enumerate(MEDIAPIPE33_TO_COCO17):
        if mp_idx < len(mp_landmarks):
            coco_keypoints[i*3] = mp_landmarks[mp_idx].x
            coco_keypoints[i*3+1] = mp_landmarks[mp_idx].y  
            coco_keypoints[i*3+2] = mp_landmarks[mp_idx].visibility
    
    return coco_keypoints
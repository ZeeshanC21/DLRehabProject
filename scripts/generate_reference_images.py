# scripts/generate_reference_images.py
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def create_stick_figure_from_joints(joints_2d, frame_idx=None, title="Reference Pose"):
    """Create a clean stick figure from REHAB24-6 joint data"""
    
    # REHAB24-6 has 26 joints in format (frames, 26, 2)
    if frame_idx is None:
        # Use middle frame as reference
        frame_idx = joints_2d.shape[0] // 2
    
    # Extract single frame
    if len(joints_2d.shape) == 3:
        joints = joints_2d[frame_idx]  # (26, 2)
    else:
        joints = joints_2d
    
    # Create white canvas
    img_size = 800
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Normalize joint positions to fit canvas
    x_coords = joints[:, 0]
    y_coords = joints[:, 1]
    
    # Remove zero/invalid points
    valid_mask = (x_coords != 0) & (y_coords != 0)
    if not valid_mask.any():
        return img
    
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    
    # Scale to canvas
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    padding = 100
    scale = min((img_size - 2*padding) / (x_max - x_min), 
                (img_size - 2*padding) / (y_max - y_min))
    
    # Normalize and center
    normalized_joints = joints.copy()
    normalized_joints[:, 0] = (joints[:, 0] - x_min) * scale + padding
    normalized_joints[:, 1] = (joints[:, 1] - y_min) * scale + padding
    
    # Define skeleton connections (approximate body structure)
    connections = [
        # Head to shoulders
        (0, 1), (0, 2),  # head to left/right shoulder
        # Arms
        (1, 3), (3, 5),  # left shoulder -> elbow -> wrist
        (2, 4), (4, 6),  # right shoulder -> elbow -> wrist
        # Torso
        (1, 7), (2, 8),  # shoulders to hips
        (7, 8),          # hip connection
        # Legs
        (7, 9), (9, 11),  # left hip -> knee -> ankle
        (8, 10), (10, 12), # right hip -> knee -> ankle
    ]
    
    # Draw skeleton
    for joint1_idx, joint2_idx in connections:
        if joint1_idx < len(normalized_joints) and joint2_idx < len(normalized_joints):
            pt1 = normalized_joints[joint1_idx]
            pt2 = normalized_joints[joint2_idx]
            
            if pt1[0] != 0 and pt1[1] != 0 and pt2[0] != 0 and pt2[1] != 0:
                cv2.line(img, 
                        (int(pt1[0]), int(pt1[1])), 
                        (int(pt2[0]), int(pt2[1])),
                        (0, 100, 255), 4)  # Orange lines
    
    # Draw joints
    for i, joint in enumerate(normalized_joints):
        if joint[0] != 0 and joint[1] != 0:
            cv2.circle(img, (int(joint[0]), int(joint[1])), 8, (255, 0, 0), -1)  # Blue dots
            # cv2.putText(img, str(i), (int(joint[0])+10, int(joint[1])), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add title
    cv2.putText(img, title, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "Reference Pose from Dataset", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    return img

def generate_all_reference_images(dataset_path="../data/raw/REHAB24-6"):
    """Generate reference images for all 6 exercises"""
    
    dataset_path = Path(dataset_path)
    output_dir = Path("reference_images")
    output_dir.mkdir(exist_ok=True)
    
    exercises = {
        'Ex1': 'Arm Abduction',
        'Ex2': 'Arm VW',
        'Ex3': 'Push-ups',
        'Ex4': 'Leg Abduction',
        'Ex5': 'Leg Lunge',
        'Ex6': 'Squats'
    }
    
    # Load segmentation to find "correct" examples
    seg_df = pd.read_csv(dataset_path / "Segmentation.csv", sep=';')
    
    print("Generating reference images from dataset...\n")
    
    for ex_id, (ex_dir, ex_name) in enumerate(exercises.items(), 1):
        print(f"Processing {ex_name}...")
        
        # Find a correct example from segmentation
        correct_examples = seg_df[
            (seg_df['exercise_id'] == ex_id) & 
            (seg_df['correctness'] == 1) &
            (seg_df['mocap_erroneous'] == 0)
        ]
        
        if len(correct_examples) == 0:
            print(f"No correct examples found for {ex_name}")
            continue
        
        # Get the first correct example
        example = correct_examples.iloc[0]
        video_id = example['video_id']
        first_frame = example['first_frame']
        last_frame = example['last_frame']
        
        # Find the corresponding joint file
        joints_dir = dataset_path / "2d_joints" / ex_dir
        joint_files = list(joints_dir.glob(f"{video_id}*.npy"))
        
        if not joint_files:
            print(f"No joint file found for {video_id}")
            continue
        
        # Load joint data
        joints_data = np.load(joint_files[0])
        
        # Use middle frame of the repetition
        mid_frame = (first_frame + last_frame) // 2
        if mid_frame < len(joints_data):
            frame_joints = joints_data[mid_frame]
        else:
            frame_joints = joints_data[len(joints_data) // 2]
        
        # Create stick figure
        stick_img = create_stick_figure_from_joints(
            frame_joints.reshape(1, 26, 2), 
            frame_idx=0,
            title=ex_name
        )
        
        # Save
        output_path = output_dir / f"{ex_dir}_{ex_name.replace(' ', '_')}_reference.png"
        cv2.imwrite(str(output_path), stick_img)
        print(f"Saved: {output_path}")
        
        # Also create a version with video frame if available
        video_dir = dataset_path / "videos" / ex_dir
        video_files = list(video_dir.glob(f"{video_id}*.mp4"))
        
        if video_files:
            cap = cv2.VideoCapture(str(video_files[0]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, video_frame = cap.read()
            cap.release()
            
            if ret:
                # Resize video frame to match stick figure
                video_frame_resized = cv2.resize(video_frame, (800, 800))
                
                # Create side-by-side comparison
                combined = np.hstack([stick_img, video_frame_resized])
                
                output_path_combined = output_dir / f"{ex_dir}_{ex_name.replace(' ', '_')}_with_video.png"
                cv2.imwrite(str(output_path_combined), combined)
                print(f"Saved combined: {output_path_combined}")
    
    print(f"\nAll reference images saved to '{output_dir}' directory!")
    return output_dir

def create_reference_pose_grid():
    """Create a single image with all 6 exercise reference poses"""
    
    ref_dir = Path("reference_images")
    if not ref_dir.exists():
        print("Reference images not found. Run generate_all_reference_images() first.")
        return
    
    # Load all reference images
    ref_images = []
    exercises = ['Ex1_Arm_Abduction', 'Ex2_Arm_VW', 'Ex3_Push-ups', 
                 'Ex4_Leg_Abduction', 'Ex5_Leg_Lunge', 'Ex6_Squats']
    
    for ex in exercises:
        img_path = ref_dir / f"{ex}_reference.png"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            # Resize for grid
            img = cv2.resize(img, (400, 400))
            ref_images.append(img)
    
    # Create 2x3 grid
    row1 = np.hstack(ref_images[:3])
    row2 = np.hstack(ref_images[3:])
    grid = np.vstack([row1, row2])
    
    # Save grid
    output_path = ref_dir / "all_exercises_reference.png"
    cv2.imwrite(str(output_path), grid)
    print(f"Reference grid saved: {output_path}")
    
    return grid

if __name__ == "__main__":
    # Generate all reference images
    output_dir = generate_all_reference_images()
    
    # Create combined grid
    create_reference_pose_grid()
    
    print("\nYou can now use these images in the real-time coach!")
    print("   Images are stored in 'reference_images/' folder")
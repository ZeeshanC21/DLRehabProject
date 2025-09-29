# extract_mediapipe.py
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path
import sys

def extract_mediapipe_from_video(video_path, output_dir):
    """Extract MediaPipe keypoints from video"""
    print(f"Starting extraction for: {video_path}")
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None
    
    video_name = Path(video_path).stem
    print(f"Processing video: {video_name}")
    
    frames_dir = output_dir / "frames" / video_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    keypoints_sequence = []
    frame_idx = 0
    
    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame (optional - comment out if you don't need frames)
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        # Extract MediaPipe keypoints
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Convert to COCO-17 format
            # MediaPipe to COCO-17 mapping
            mp_to_coco17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            
            keypoints_frame = []
            for i in mp_to_coco17:
                if i < len(results.pose_landmarks.landmark):
                    lm = results.pose_landmarks.landmark[i]
                    x = lm.x * frame.shape[1]  # Convert to pixel coords
                    y = lm.y * frame.shape[0]
                    conf = lm.visibility
                    keypoints_frame.extend([x, y, conf])
                else:
                    keypoints_frame.extend([0, 0, 0])
            
            keypoints_sequence.append({
                "frame_id": frame_idx,
                "keypoints": keypoints_frame  # 17 keypoints * 3 = 51 values
            })
        else:
            # No pose detected - add zeros
            keypoints_sequence.append({
                "frame_id": frame_idx,
                "keypoints": [0] * 51  # 17 keypoints * 3 = 51 values
            })
        
        frame_idx += 1
        
        # Progress update every 100 frames
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    # Save as COCO-like JSON
    coco_data = {
        "video_name": video_name,
        "total_frames": frame_idx,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "keypoints_sequence": keypoints_sequence,
        "keypoint_names": [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
    }
    
    json_path = output_dir / f"{video_name}_mediapipe_coco17.json"
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    # Save as numpy array
    np_keypoints = np.array([kp["keypoints"] for kp in keypoints_sequence])
    npy_path = output_dir / f"{video_name}_keypoints.npy"
    np.save(npy_path, np_keypoints)
    
    cap.release()
    print(f"Extraction completed. Processed {frame_idx} frames")
    return json_path, np_keypoints.shape

def find_video_directory():
    """Find the correct video directory"""
    # Try different possible paths
    possible_paths = [
        Path("../data/raw/REHAB24-6/videos/"),  # From scripts/ directory
        Path("data/raw/REHAB24-6/videos/"),     # From root directory
        Path("./data/raw/REHAB24-6/videos/"),   # Explicit relative path
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"Found video directory: {path.resolve()}")
            return path
    
    # If none found, let user specify
    print("Could not find video directory automatically.")
    print("Current working directory:", Path.cwd())
    print("\nPlease enter the path to your videos directory:")
    custom_path = input("Video directory path: ").strip()
    
    if custom_path:
        path = Path(custom_path)
        if path.exists():
            return path
        else:
            print(f"Path {path} does not exist!")
    
    return None

def main(start_index=1):
    print("MediaPipe Pose Extraction Pipeline")
    print("=" * 40)

    # Find video directory
    video_dir = find_video_directory()
    if video_dir is None:
        print("Could not locate video directory. Exiting.")
        sys.exit(1)

    # Create output directory
    output_dir = Path("../mediapipe_output")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    # Find all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.rglob(ext)))

    video_files = sorted(video_files)  # ensure consistent order

    print(f"\nFound {len(video_files)} video files:")
    for i, video_file in enumerate(video_files, 1):
        print(f"{i:2d}. {video_file.name}")

    if not video_files:
        print("No video files found!")
        return

    # Process each video starting from start_index
    print(f"\nProcessing videos starting at index {start_index}/{len(video_files)}...")
    print("=" * 40)

    successful = 0
    failed = 0

    for i, video_path in enumerate(video_files[start_index - 1:], start=start_index):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_path.name}")
        try:
            json_path, shape = extract_mediapipe_from_video(video_path, output_dir)
            if json_path and shape:
                print(f"✓ Success: {json_path.name}, Shape: {shape}")
                successful += 1
            else:
                print(f"✗ Failed: {video_path.name}")
                failed += 1
        except Exception as e:
            print(f"✗ Error processing {video_path.name}: {str(e)}")
            failed += 1

    print(f"\n" + "=" * 40)
    print(f"Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    # Allow optional command line argument
    start = 1
    if len(sys.argv) > 1:
        try:
            start = int(sys.argv[1])
        except ValueError:
            print("Invalid start index, using default 1")
    main(start_index=start)


if __name__ == "__main__":
    main()
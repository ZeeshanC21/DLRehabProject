# scripts/realtime_inference.py
import cv2
import mediapipe as mp
import numpy as np
import torch
import math
from exercise_model import ImprovedExerciseModel
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

class ExerciseCoach:
    """Exercise coaching system with PIP reference image"""
    
    def __init__(self, model_path='best_improved_exercise_model.pth'):
        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = ImprovedExerciseModel(
            input_dim=52,
            hidden_dim=256,
            num_layers=3,
            num_exercises=6,
            dropout=0.4,
            use_attention=True
        )
        self.correctness_history = []
        self.history_size = 10
        
        # Load model
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Exercise definitions
        self.exercises = {
            1: {
                'name': 'Arm Abduction',
                'ref_file': 'Ex1_Arm_Abduction_reference.png',
                'key_angles': {
                    'left_shoulder': {'target': 90, 'tolerance': 15, 'name': 'L Shoulder'},
                    'right_shoulder': {'target': 90, 'tolerance': 15, 'name': 'R Shoulder'}
                },
                'instructions': [
                    'Stand straight, feet shoulder-width',
                    'Raise arms sideways to shoulder height',
                    'Keep arms straight',
                    'Target: 90 degrees at shoulders'
                ]
            },
            2: {
                'name': 'Arm VW',
                'ref_file': 'Ex2_Arm_VW_reference.png',
                'key_angles': {
                    'left_elbow': {'target': 90, 'tolerance': 15, 'name': 'L Elbow'},
                    'right_elbow': {'target': 90, 'tolerance': 15, 'name': 'R Elbow'}
                },
                'instructions': [
                    'Start arms in V (straight up)',
                    'Bend elbows to W shape',
                    'Keep upper arms still',
                    'Target: 90 degree elbow bend'
                ]
            },
            3: {
                'name': 'Push-ups',
                'ref_file': 'Ex3_Push-ups_reference.png',
                'key_angles': {
                    'left_elbow': {'target': 90, 'tolerance': 15, 'name': 'L Elbow'},
                    'right_elbow': {'target': 90, 'tolerance': 15, 'name': 'R Elbow'}
                },
                'instructions': [
                    'Hands on table, shoulder-width',
                    'Body straight, head to heels',
                    'Lower until elbows at 90 degrees',
                    'Push back up'
                ]
            },
            4: {
                'name': 'Leg Abduction',
                'ref_file': 'Ex4_Leg_Abduction_reference.png',
                'key_angles': {
                    'left_hip': {'target': 45, 'tolerance': 10, 'name': 'L Hip'},
                    'right_hip': {'target': 45, 'tolerance': 10, 'name': 'R Hip'}
                },
                'instructions': [
                    'Stand on one leg (use support)',
                    'Keep leg straight',
                    'Lift other leg sideways',
                    'Target: 45 degree angle at hip'
                ]
            },
            5: {
                'name': 'Leg Lunge',
                'ref_file': 'Ex5_Leg_Lunge_reference.png',
                'key_angles': {
                    'left_knee': {'target': 90, 'tolerance': 15, 'name': 'L Knee'},
                    'right_knee': {'target': 90, 'tolerance': 15, 'name': 'R Knee'}
                },
                'instructions': [
                    'Step forward with one leg',
                    'Lower back knee to ground',
                    'Front knee at 90 degrees',
                    'Keep torso upright'
                ]
            },
            6: {
                'name': 'Squats',
                'ref_file': 'Ex6_Squats_reference.png',
                'key_angles': {
                    'left_knee': {'target': 90, 'tolerance': 15, 'name': 'L Knee'},
                    'right_knee': {'target': 90, 'tolerance': 15, 'name': 'R Knee'}
                },
                'instructions': [
                    'Feet shoulder-width apart',
                    'Back straight, chest up',
                    'Lower until thighs parallel',
                    'Target: 90 degree knee bend'
                ]
            }
        }
        
        self.keypoint_buffer = []
        self.buffer_size = 30
    
    def load_reference_image(self, exercise_id):
        """Load reference image for PIP display"""
        ref_dir = Path("reference_images")
        ref_file = self.exercises[exercise_id]['ref_file']
        ref_path = ref_dir / ref_file
        
        if ref_path.exists():
            img = cv2.imread(str(ref_path))
            img = cv2.resize(img, (200, 300))  # Small size for corner
            return img
        else:
            # Placeholder if image not found
            placeholder = np.ones((300, 200, 3), dtype=np.uint8) * 240
            cv2.putText(placeholder, "Reference", (40, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(placeholder, "Not Found", (35, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return placeholder
    
    def mediapipe_to_rehab_joints(self, landmarks):
        """Convert MediaPipe to REHAB format"""
        if not landmarks:
            return np.zeros(52)
        
        joint_mapping = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
                        9, 10, 7, 8, 17, 18, 19, 20, 21, 22, 29, 30, 31]
        
        joints_2d = []
        for joint_idx in joint_mapping[:26]:
            if joint_idx < len(landmarks.landmark):
                lm = landmarks.landmark[joint_idx]
                joints_2d.extend([lm.x, lm.y])
            else:
                joints_2d.extend([0.0, 0.0])
        
        return np.array(joints_2d)
    
    def extract_keypoints(self, frame):
        """Extract keypoints from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            keypoints = self.mediapipe_to_rehab_joints(results.pose_landmarks)
            return keypoints, results.pose_landmarks
        return None, None
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        try:
            a, b, c = np.array(p1), np.array(p2), np.array(p3)
            ba, bc = a - b, c - b
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            return math.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        except:
            return 0.0
    
    def get_angles(self, landmarks):
        """Calculate all angles"""
        def get_lm(idx):
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                return [lm.x, lm.y]
            return [0, 0]
        
        angles = {}
        try:
            angles['left_elbow'] = self.calculate_angle(get_lm(11), get_lm(13), get_lm(15))
            angles['right_elbow'] = self.calculate_angle(get_lm(12), get_lm(14), get_lm(16))
            angles['left_shoulder'] = self.calculate_angle(get_lm(23), get_lm(11), get_lm(13))
            angles['right_shoulder'] = self.calculate_angle(get_lm(24), get_lm(12), get_lm(14))
            angles['left_knee'] = self.calculate_angle(get_lm(23), get_lm(25), get_lm(27))
            angles['right_knee'] = self.calculate_angle(get_lm(24), get_lm(26), get_lm(28))
            angles['left_hip'] = self.calculate_angle(get_lm(11), get_lm(23), get_lm(25))
            angles['right_hip'] = self.calculate_angle(get_lm(12), get_lm(24), get_lm(26))
        except:
            pass
        return angles
    
    def predict_correctness(self, keypoint_sequence):
        """Predict correctness using YOUR TRAINED MODEL"""
        if len(keypoint_sequence) < 10:
            return None
        
        sequence = np.array(keypoint_sequence[-30:])
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
        
        x = torch.tensor(sequence, dtype=torch.float32).to(self.device)
        x = torch.nan_to_num(x, nan=0.0)
        
        with torch.no_grad():
            outputs = self.model(x)
            correctness_prob = outputs['correctness'].item()
        
        self.correctness_history.append(correctness_prob)
        if len(self.correctness_history) > self.history_size:
            self.correctness_history.pop(0)
        
        # Return smoothed average
        smoothed_prob = np.mean(self.correctness_history)
        return smoothed_prob
    
    def generate_feedback(self, angles, exercise_config, correctness_prob):
        """Generate feedback"""
        feedback = []
        
        if correctness_prob is not None:
            if correctness_prob > 0.65:
                feedback.append(('EXCELLENT FORM!', (0, 255, 0)))
            elif correctness_prob > 0.45:
                feedback.append(('Good - Minor tweaks', (0, 255, 255)))
            else:
                feedback.append(('Needs correction', (0, 0, 255)))
            
            feedback.append((f'Score: {correctness_prob:.0%}', (255, 255, 255)))
        
        for angle_key, angle_config in exercise_config['key_angles'].items():
            if angle_key in angles:
                current = angles[angle_key]
                target = angle_config['target']
                tolerance = angle_config['tolerance']
                name = angle_config['name']
                
                diff = abs(current - target)
                
                if diff <= tolerance:
                    feedback.append((f'{name}: {current:.0f}deg Perfect!', (0, 255, 0)))
                elif diff <= tolerance * 1.5:
                    direction = 'more' if current < target else 'less'
                    feedback.append((f'{name}: {current:.0f}deg Open {direction}', (0, 255, 255)))
                else:
                    direction = 'more' if current < target else 'less'
                    feedback.append((f'{name}: {current:.0f}deg->{target}deg', (0, 0, 255)))
        
        return feedback
    
    def overlay_reference_pip(self, frame, reference_img):
        """Overlay reference image in top-right corner"""
        h, w = frame.shape[:2]
        ref_h, ref_w = reference_img.shape[:2]
        
        # Position: top-right corner with margin
        x_offset = w - ref_w - 10
        y_offset = 10
        
        # Add white border
        bordered_ref = cv2.copyMakeBorder(reference_img, 3, 3, 3, 3,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        ref_h, ref_w = bordered_ref.shape[:2]
        
        # Overlay on frame
        frame[y_offset:y_offset+ref_h, x_offset:x_offset+ref_w] = bordered_ref
        
        # Add label
        label_bg_y = y_offset + ref_h + 5
        cv2.rectangle(frame, (x_offset, label_bg_y),
                     (x_offset + ref_w, label_bg_y + 25), (0, 0, 0), -1)
        cv2.putText(frame, "REFERENCE", (x_offset + 20, label_bg_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_ui(self, frame, landmarks, angles, feedback, exercise_name, reference_img):
        """Draw complete UI with PIP reference"""
        h, w, _ = frame.shape
        
        # Draw skeleton
        if landmarks:
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=2)
            )
        
        # Draw header
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, f'Exercise: {exercise_name}', (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Overlay reference image
        frame = self.overlay_reference_pip(frame, reference_img)
        
        # Draw feedback on left
        y_start = 90
        for i, (message, color) in enumerate(feedback):
            y_pos = y_start + i * 40
            
            # Background
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (10, y_pos - 30),
                         (text_size[0] + 20, y_pos + 5), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, message, (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw angles on joints
        if landmarks:
            angle_display = [
                ('left_elbow', 13), ('right_elbow', 14),
                ('left_knee', 25), ('right_knee', 26)
            ]
            
            for angle_key, landmark_idx in angle_display:
                if angle_key in angles and landmark_idx < len(landmarks.landmark):
                    pos = landmarks.landmark[landmark_idx]
                    px, py = int(pos.x * w), int(pos.y * h)
                    
                    text = f'{angles[angle_key]:.0f}deg'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (px + 5, py - 35),
                                (px + text_size[0] + 15, py - 5), (0, 0, 0), -1)
                    cv2.putText(frame, text, (px + 10, py - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Controls at bottom
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Press 'q' to quit | 'r' to reset | 's' to save",
                   (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def run_exercise_session(self, exercise_id):
        """Run exercise session with PIP reference"""
        exercise = self.exercises[exercise_id]
        
        print(f"\n{'='*60}")
        print(f"Starting: {exercise['name']}")
        print(f"\nInstructions:")
        for i, instruction in enumerate(exercise['instructions'], 1):
            print(f"   {i}. {instruction}")
        print(f"{'='*60}\n")
        print("Loading reference image from dataset...")
        
        # Load reference
        reference_img = self.load_reference_image(exercise_id)
        
        print("Reference loaded! Starting camera...")
        print("You'll see the reference in top-right corner")
        print("\nPress Enter when ready...")
        input()
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.keypoint_buffer = []
        
        print("Camera started!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            keypoints, landmarks = self.extract_keypoints(frame)
            
            if keypoints is not None:
                self.keypoint_buffer.append(keypoints)
                if len(self.keypoint_buffer) > self.buffer_size:
                    self.keypoint_buffer.pop(0)
                
                angles = self.get_angles(landmarks)
                
                correctness = None
                if len(self.keypoint_buffer) >= 10:
                    correctness = self.predict_correctness(self.keypoint_buffer)
                
                feedback = self.generate_feedback(angles, exercise, correctness)
            else:
                feedback = [("No pose detected", (0, 0, 255)),
                           ("Stand in camera view", (0, 0, 255))]
                angles = {}
            
            # Draw UI
            display_frame = self.draw_ui(frame, landmarks, angles, feedback,
                                        exercise['name'], reference_img)
            
            cv2.imshow(f"Rehabilitation Coach - {exercise['name']}", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.keypoint_buffer = []
                print("Buffer reset!")
            elif key == ord('s'):
                filename = f"exercise_screenshot_{exercise_id}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Session completed!")

def show_menu():
    """Show exercise menu"""
    exercises = {
        1: 'Arm Abduction', 2: 'Arm VW', 3: 'Push-ups',
        4: 'Leg Abduction', 5: 'Leg Lunge', 6: 'Squats'
    }
    
    print("\n" + "="*60)
    print("REHABILITATION EXERCISE COACH")
    print("="*60)
    print("\nSelect an exercise:")
    print("-" * 60)
    
    for ex_id, ex_name in exercises.items():
        print(f"  {ex_id}. {ex_name}")
    
    print(f"  0. Exit")
    print("-" * 60)
    
    while True:
        try:
            choice = int(input("\nEnter your choice (0-6): "))
            if choice == 0:
                return None
            if 1 <= choice <= 6:
                return choice
            print("Invalid choice. Please enter 0-6.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    """Main function"""
    print("Loading model...")
    coach = ExerciseCoach()
    
    # Check for reference images
    ref_dir = Path("reference_images")
    if not ref_dir.exists() or not any(ref_dir.glob("*.png")):
        print("\nWarning: Reference images not found!")
        print("Run 'python generate_reference_images.py' first to create them.")
        print("Continuing anyway (will show placeholders)...\n")
    
    while True:
        exercise_id = show_menu()
        
        if exercise_id is None:
            print("\nThank you for using Rehabilitation Coach!")
            break
        
        coach.run_exercise_session(exercise_id)
        
        print("\n" + "="*60)
        input("Press Enter to return to menu...")

if __name__ == "__main__":
    main()
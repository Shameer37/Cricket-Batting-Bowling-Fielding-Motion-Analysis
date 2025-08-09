import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# ===== CONFIGURATION =====
# IMPORTANT: Replace these with the actual paths to your video files.
# The script will loop through this list to process each video.
# In Google Colab, you would first mount your Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')
VIDEO_PATHS = [
    "/content/drive/MyDrive/Cricket_Analysis_project/Videos/Video-1.mp4",
    "/content/drive/MyDrive/Cricket_Analysis_project/Videos/Video-2.mp4",
    "/content/drive/MyDrive/Cricket_Analysis_project/Videos/Video-3.mp4",
    "/content/drive/MyDrive/Cricket_Analysis_project/Videos/Video-4.mp4",
    "/content/drive/MyDrive/Cricket_Analysis_project/Videos/Video-5.mp4",
]

# Set output directory and create it if it doesn't exist
OUTPUT_DIR = "final_analysis_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

FPS = 20 # Frames per second for the animation output

# ===== JOINT MAPPING AND SKELETON CONNECTIONS =====
# This is a common 33-joint model from pose estimation libraries like MediaPipe.
JOINT_MAP = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6, 'left_ear': 7,
    'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10, 'left_shoulder': 11,
    'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15,
    'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
    'right_index': 20, 'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23,
    'right_hip': 24, 'left_knee': 25, 'right_knee': 26, 'left_ankle': 27,
    'right_ankle': 28, 'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31,
    'right_foot_index': 32
}

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (10, 9), (9, 7), (10, 8),
    (11, 12), (11, 13), (13, 15), (15, 17), (17, 19), (19, 21), # Left arm, hand
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 22), # Right arm, hand
    (11, 23), (12, 24), (23, 24), # Torso, hips
    (23, 25), (25, 27), (27, 29), (29, 31), # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32)  # Right leg
]

# ===== IDEAL BIOMECHANICS FOR CORRECTION MODELS =====
# Define a simple dictionary of ideal angles for key poses.
# These values are examples and should be based on biomechanical research.
IDEAL_ANGLES = {
    "Bowler": {
        "release_angle": 175,  # Close to a straight arm
        "release_frame": 0.5  # Example: check at 50% of the bowling action
    },
    "Batsman": {
        "back_angle": 90, # A relatively straight back for a stable stance
        "max_bat_angle": 90 # Bat should not be parallel to the ground
    },
    "Fielder": {
        "spread_angle": 100, # A wide stance for better stability
        "knee_bend": 120 # A low posture for a ready position
    }
}


# ===== REAL VIDEO PROCESSING FUNCTION (REPLACES DUMMY GENERATION) =====
def process_video_to_keypoints(video_path):
    """
    Processes a video file to extract 3D pose keypoints using MediaPipe.
    """
    print(f"✅ Starting to process video: {video_path}")
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    all_frames_keypoints = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Error: Could not open video file {video_path}")
        return None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            
            if results.pose_landmarks:
                frame_keypoints = {}
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    joint_name = next((name for name, index in JOINT_MAP.items() if index == idx), f'joint_{idx}')
                    frame_keypoints[f'x_{joint_name}'] = landmark.x
                    frame_keypoints[f'y_{joint_name}'] = landmark.y
                    frame_keypoints[f'z_{joint_name}'] = landmark.z
                all_frames_keypoints.append(frame_keypoints)
            
    cap.release()
    print(f"✅ Finished processing video. Extracted {len(all_frames_keypoints)} frames.")
    
    if all_frames_keypoints:
        return pd.DataFrame(all_frames_keypoints)
    else:
        print("⚠️ No pose landmarks were detected in the video.")
        return None

# ===== AI MODEL TRAINING AND CLASSIFICATION =====
def train_role_classifier(X_train, y_train):
    """Trains a simple SVM classifier for role prediction."""
    print("🧠 Training the SVM model...")
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    print("✅ Model training complete.")
    return model

def create_training_data(keypoint_df):
    """
    Simulates a small, manually-labeled training dataset.
    In a real scenario, you would manually label a few frames.
    """
    print("📊 Creating dummy training data...")
    # Example: label the first 10 frames as 'Batsman', the next 10 as 'Bowler', etc.
    num_frames = len(keypoint_df)
    labels = pd.Series(['Batsman'] * (num_frames // 3) + ['Bowler'] * (num_frames // 3) + ['Fielder'] * (num_frames - 2 * (num_frames // 3)))
    
    features = keypoint_df.values
    
    return features, labels

def load_or_train_model(keypoint_df):
    """Loads a pre-trained model or trains a new one."""
    model_path = 'role_classifier.joblib'
    if os.path.exists(model_path):
        print("🧠 Loading pre-trained model...")
        return joblib.load(model_path)
    else:
        features, labels = create_training_data(keypoint_df)
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        model = train_role_classifier(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"✅ Model accuracy on test data: {accuracy_score(y_test, y_pred):.2f}")
        joblib.dump(model, model_path)
        print(f"💾 Model saved to {model_path}")
        return model

# ===== BIOMECHANICAL ANGLE CALCULATION FUNCTIONS (unchanged) =====
def calculate_angle(A, B, C):
    """Calculates the angle (in degrees) between three 3D points centered at B."""
    BA = A - B
    BC = C - B
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_back_angle(frame_coords, joint_map):
    """Calculates the back angle for a batsman using shoulders and hips."""
    left_shoulder = frame_coords[joint_map['left_shoulder']]
    right_shoulder = frame_coords[joint_map['right_shoulder']]
    left_hip = frame_coords[joint_map['left_hip']]
    right_hip = frame_coords[joint_map['right_hip']]
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    hip_midpoint = (left_hip + right_hip) / 2
    vertical_vector = np.array([0, 1, 0])
    torso_vector = shoulder_midpoint - hip_midpoint
    cosine_angle = np.dot(torso_vector, vertical_vector) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return 90 - np.degrees(angle)

def calculate_spread_angle(frame_coords, joint_map):
    """Calculates the spread angle for a fielder using ankles and hips."""
    left_ankle = frame_coords[joint_map['left_ankle']]
    right_ankle = frame_coords[joint_map['right_ankle']]
    left_hip = frame_coords[joint_map['left_hip']]
    right_hip = frame_coords[joint_map['right_hip']]
    hip_midpoint = (left_hip + right_hip) / 2
    left_leg_vector = left_ankle - hip_midpoint
    right_leg_vector = right_ankle - hip_midpoint
    cosine_angle = np.dot(left_leg_vector, right_leg_vector) / (np.linalg.norm(left_leg_vector) * np.linalg.norm(right_leg_vector))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# ===== CORRECTION ANALYSIS FUNCTION =====
def analyze_for_corrections(role, angles, ideal_angles):
    """
    Analyzes biomechanical angles against ideal standards and provides corrections.
    """
    corrections = []
    if role == "Bowler":
        if abs(angles['right_elbow_angle'] - ideal_angles['Bowler']['release_angle']) > 5:
            corrections.append("Bowler: Arm is not straight enough at release. Strive for a straighter elbow.")
    elif role == "Batsman":
        if angles['back_angle'] < (ideal_angles['Batsman']['back_angle'] - 10):
            corrections.append("Batsman: Back angle is too low. Maintain an upright posture for better balance.")
    elif role == "Fielder":
        if angles['spread_angle'] < (ideal_angles['Fielder']['spread_angle'] - 10):
            corrections.append("Fielder: Stance is too narrow. Widen your feet for a more stable base.")
    return corrections

# ===== MAIN EXECUTION LOOP =====
def run_analysis_pipeline(video_path, role_model):
    """
    Main function to run the full analysis pipeline for a single video.
    """
    video_name = os.path.basename(video_path).split('.')[0]
    print(f"\n--- Starting analysis for {video_name} ---")

    keypoint_df = process_video_to_keypoints(video_path)
    
    if keypoint_df is None or keypoint_df.empty:
        print(f"--- Skipping analysis for {video_name} due to no data. ---")
        return

    coord_cols = [c for c in keypoint_df.columns if c.startswith(('x_', 'y_', 'z_'))]
    keypoint_df[coord_cols] = keypoint_df[coord_cols].interpolate(limit_direction="both").astype(float)
    coords = keypoint_df[coord_cols].values
    num_joints = len(coord_cols) // 3

    roles = role_model.predict(coords)
    right_elbow_angles, back_angles, spread_angles, all_corrections = [], [], [], []

    for i in range(len(coords)):
        frame_coords_flat = coords[i]
        frame_coords = frame_coords_flat.reshape(num_joints, 3)
        current_role = roles[i]

        try:
            elbow_angle = calculate_angle(frame_coords[JOINT_MAP['right_shoulder']], frame_coords[JOINT_MAP['right_elbow']], frame_coords[JOINT_MAP['right_wrist']])
            back_angle = calculate_back_angle(frame_coords, JOINT_MAP)
            spread_angle = calculate_spread_angle(frame_coords, JOINT_MAP)
        except KeyError:
            print("⚠️ Could not find all necessary joints for angle calculation.")
            elbow_angle, back_angle, spread_angle = np.nan, np.nan, np.nan

        right_elbow_angles.append(elbow_angle)
        back_angles.append(back_angle)
        spread_angles.append(spread_angle)

        angles = {'right_elbow_angle': elbow_angle, 'back_angle': back_angle, 'spread_angle': spread_angle}
        corrections = analyze_for_corrections(current_role, angles, IDEAL_ANGLES)
        all_corrections.append(corrections)

    print(f"✅ Processed {len(roles)} frames for {video_name}.")

    output_video_path = os.path.join(OUTPUT_DIR, f"{video_name}_3d_pose_with_roles_and_corrections.mp4")
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    role_colors = {"Bowler": "red", "Batsman": "green", "Fielder": "blue"}
    
    def update_3d(frame):
        ax_3d.clear()
        xs, ys, zs = coords[frame, 0::3], coords[frame, 1::3], coords[frame, 2::3]
        role = roles[frame]
        color = role_colors.get(role, "gray")
        
        # Highlight joints that need correction
        highlighted_joints = []
        if all_corrections[frame]:
            for correction in all_corrections[frame]:
                if "elbow" in correction:
                    highlighted_joints.extend([JOINT_MAP['right_shoulder'], JOINT_MAP['right_elbow'], JOINT_MAP['right_wrist']])
                if "back" in correction:
                    highlighted_joints.extend([JOINT_MAP['left_shoulder'], JOINT_MAP['right_shoulder'], JOINT_MAP['left_hip'], JOINT_MAP['right_hip']])
                if "stance" in correction or "feet" in correction:
                    highlighted_joints.extend([JOINT_MAP['left_hip'], JOINT_MAP['right_hip'], JOINT_MAP['left_ankle'], JOINT_MAP['right_ankle']])

        ax_3d.scatter(xs, ys, zs, c=color, s=20)
        ax_3d.scatter(xs[highlighted_joints], ys[highlighted_joints], zs[highlighted_joints], c='yellow', s=100, edgecolors='black')

        for (i, j) in CONNECTIONS:
            if i < num_joints and j < num_joints:
                line_color = 'yellow' if i in highlighted_joints or j in highlighted_joints else color
                ax_3d.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], c=line_color)

        ax_3d.set_xlim(np.nanmin(coords[:, 0::3])-0.5, np.nanmax(coords[:, 0::3])+0.5)
        ax_3d.set_ylim(np.nanmin(coords[:, 1::3])-0.5, np.nanmax(coords[:, 1::3])+0.5)
        ax_3d.set_zlim(np.nanmin(coords[:, 2::3])-0.5, np.nanmax(coords[:, 2::3])+0.5)
        
        title_text = f"Frame {frame} - Role: {role}\n"
        if all_corrections[frame]:
            title_text += "\n".join(all_corrections[frame])
        ax_3d.set_title(title_text)
    
    anim = FuncAnimation(fig_3d, update_3d, frames=len(coords), interval=1000/FPS)
    anim.save(output_video_path, fps=FPS, extra_args=['-vcodec', 'libx264'])
    print(f"✅ 3D animation with corrections saved to {output_video_path}")
    plt.close(fig_3d)

    output_graph_path = os.path.join(OUTPUT_DIR, f"{video_name}_angle_over_time_segmented.png")
    fig_angle, ax_angle = plt.subplots(figsize=(12, 6))
    frames = np.arange(len(roles))
    bowler_frames = frames[roles == "Bowler"]
    batsman_frames = frames[roles == "Batsman"]
    fielder_frames = frames[roles == "Fielder"]
    ax_angle.plot(bowler_frames, np.array(right_elbow_angles)[bowler_frames], label='Bowler - Right Elbow Angle', color=role_colors['Bowler'], linestyle='-')
    ax_angle.plot(batsman_frames, np.array(back_angles)[batsman_frames], label='Batsman - Back Angle', color=role_colors['Batsman'], linestyle='-')
    ax_angle.plot(fielder_frames, np.array(spread_angles)[fielder_frames], label='Fielder - Spread Angle', color=role_colors['Fielder'], linestyle='-')
    ax_angle.set_title(f"Key Angles Over Time by Role for {video_name}")
    ax_angle.set_xlabel("Frame Number")
    ax_angle.set_ylabel("Angle (degrees)")
    ax_angle.grid(True)
    ax_angle.legend()
    plt.tight_layout()
    plt.savefig(output_graph_path)
    print(f"✅ Segmented angle-over-time graph saved to {output_graph_path}")
    plt.close(fig_angle)

    output_csv_path = os.path.join(OUTPUT_DIR, f"{video_name}_final_analysis_report.csv")
    report_df = pd.DataFrame({
        'frame_number': frames,
        'role': roles,
        'right_elbow_angle': right_elbow_angles,
        'back_angle': back_angles,
        'spread_angle': spread_angles,
        'corrections': all_corrections
    })
    report_df.to_csv(output_csv_path, index=False)
    print(f"✅ Final analysis report saved to {output_csv_path}")

# Run the pipeline with the AI model
if VIDEO_PATHS:
    sample_keypoint_df = process_video_to_keypoints(VIDEO_PATHS[0])
    if sample_keypoint_df is not None:
        role_classifier_model = load_or_train_model(sample_keypoint_df)
        for path in VIDEO_PATHS:
            run_analysis_pipeline(path, role_classifier_model)
    else:
        print("⚠️ Could not generate a sample keypoint dataframe for model training. Please check your video files.")
else:
    print("\n--- No video paths provided. Please add video paths to the VIDEO_PATHS list. ---")


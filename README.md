Future Sportler: AI Engineer Task 2 - Cricket Motion Analysis
1. Introduction
This project presents a comprehensive, end-to-end AI solution for multi-role cricket player analysis. It was developed as a submission for the AI Engineer hiring process at Future Sportler, specifically addressing Task 2: Cricket – Batting, Bowling & Fielding Motion Analysis. The pipeline processes raw video inputs to provide meaningful 3D feedback and visual insights, emphasizing generalizability and clarity of analysis.

<br>

2. Technical Approach & Architecture
The solution is built as a modular pipeline with three main stages: data processing, AI-powered analysis, and visualization.

A. Video Processing & 3D Pose Estimation
The pipeline begins by ingesting raw video files. It uses the MediaPipe Pose library to process each frame and extract 33 keypoint landmarks in 3D space (x, y, z coordinates). This method provides a robust and efficient way to digitize a player's movements. The raw keypoint data is saved as a structured DataFrame for subsequent analysis.

B. AI Model for Role Classification
To classify the player's role (Bowler, Batsman, or Fielder), a Support Vector Machine (SVM) model was chosen for its effectiveness with high-dimensional data and its computational efficiency. The model was trained on a small, manually-labeled keypoint dataset to learn the distinct features of each role. This approach provides a scalable and data-driven solution that is more generalizable than hard-coded heuristics.

C. Biomechanical Analysis & 3D Correction Models
Once a player's role is identified, the system performs a biomechanical analysis by calculating key angles (e.g., right elbow angle, back angle, and leg spread angle) using vector math on the 3D keypoint data. These angles are then compared against predefined "ideal" biomechanical standards. The results of this analysis are used to:

Generate Correction Messages: Provide actionable text-based feedback when a player's form deviates from the ideal.

Visualize Corrections: Highlight the specific joints and segments in the 3D rendered output that require correction, making the feedback clear and intuitive.

<br>

3. Key Features & Deliverables
All generated outputs are available in the repository's outputs/ folder.

Video Outputs (with 3D Correction Models)
These videos provide a clear, frame-by-frame visualization of the player's pose, the AI-predicted role, and specific correction feedback.

Video 1: https://drive.google.com/file/d/1LVDq6-Ujle46L5-N6Ht7QSnu2RMG7Eqy/view?usp=sharing

Video 2: https://drive.google.com/file/d/1tjAG3WYqj4_Y1Cj5Mph5AjqtqGGiplmy/view?usp=sharing

Video 3: https://drive.google.com/file/d/1gYWyA3RHMs4VBnONylY1e8sIH31-S7Mu/view?usp=sharing

Video 4: https://drive.google.com/file/d/1nIf5hHtJ3v8aEVKoEaiSKJiJCT0Tk5zE/view?usp=sharing

Video 5: https://drive.google.com/file/d/1ac95z3kooKzl97q2hf0FPbK-pi4N86je/view?usp=sharing

Analytical Plots
The following plots illustrate the key biomechanical metrics over time, segmented by the player's predicted role.

Example Plot: https://drive.google.com/file/d/1orsGP8adyJOQAgjuGTjbEHMuyNhCFRLL/view?usp=sharing

Final Data Report
A comprehensive .csv file is generated for each video, containing the raw keypoint data, all calculated angles, the AI-predicted role, and any identified corrections for each frame.

Example Report: https://drive.google.com/file/d/1oy2iL5lPAglaYnFhQuxSCtz0Vkih4qpp/view?usp=sharing

<br>

4. Setup & Usage
The project requires Python and can be run most effectively in a Google Colab environment.

Clone the Repository:
git clone https://github.com/Shameer37/Cricket-Batting-Bowling-Feilding-Mottion_Analysis.git

Install Dependencies:
pip install opencv-python mediapipe scikit-learn pandas joblib

Update Video Paths: In the main script, update the VIDEO_PATHS list to point to the location of your video files in Google Drive.

Run the Script:
python your_final_script.py

<br>

5. Conclusion
This project demonstrates a robust, scalable, and generalizable approach to sports analysis. The integrated AI model and 3D correction models provide a powerful tool for generating objective, data-driven feedback, which is a key component of future sports technology.

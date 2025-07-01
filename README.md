Player Tracking System with YOLOv11 and DeepSORT
Overview
This project implements a player tracking system for sports videos, utilizing two synchronized camera perspectives: a broadcast view and a tactical (tacticam) view. The system leverages a custom-trained YOLOv11 model for object detection and an improved DeepSORT algorithm for multi-object tracking, with appearance-based matching to map player identities across the two camera views.
Features

Object Detection: Uses a custom-trained YOLOv11 model (best.pt) to detect players in video frames.
Multi-Object Tracking: Employs DeepSORT with enhanced feature extraction (OSNet or color histogram fallback) to track players across frames.
Cross-Camera Mapping: Matches player identities between broadcast and tacticam videos based on appearance features using cosine similarity and the Hungarian algorithm.
Frame Synchronization: Accounts for a time offset between the two camera views (tacticam starts 4 seconds before broadcast, handled with a 65-frame offset at the video's FPS).
Output: Generates annotated output videos (broadcast_out.mp4, tacticam_out.mp4) with bounding boxes and track IDs, plus a debug frame at frame 30 for each camera.

Files

main.py: Main script that processes input videos, performs detection and tracking, and maps player IDs across cameras.
utils.py: Utility functions for computing bounding box centers, averaging appearance features, team color classification, and matching player IDs across cameras.
deep_sort.py: Custom DeepSORT implementation with improved feature extraction (OSNet or color histogram) and robust error handling.
Input Videos:
broadcast.mp4: Broadcast camera video.
tacticam.mp4: Tactical camera video.


Output Files:
broadcast_out.mp4, tacticam_out.mp4: Annotated videos with bounding boxes and track IDs.
broadcast_debug_frame.jpg, tacticam_debug_frame.jpg: Debug frames saved at frame 30.


Model: best.pt: Custom-trained YOLOv11 model for player detection.

Requirements

Python 3.8+
Libraries:
opencv-python (cv2)
numpy
ultralytics (YOLOv11)
torch
torchvision
torchreid
scipy
scikit-learn


Hardware: GPU recommended for faster processing (DeepSORT uses CUDA if available).

How It Works

Video Processing:

Loads broadcast and tacticam videos.
Skips the first 65 frames of the tacticam video to synchronize with the broadcast video.
Processes each frame using YOLOv11 for player detection (classes 0, 1, 2; confidence > 0.5).


Tracking:

DeepSORT tracks detected players across frames, using either OSNet deep features or color histogram features (fallback).
Tracks are filtered to retain only those with at least 3 detections for reliability.
Bounding boxes and track IDs are drawn on the output videos.


Cross-Camera Mapping:

Extracts appearance features for each track.
Computes cosine similarity between tacticam and broadcast track features.
Uses the Hungarian algorithm to match player IDs, ensuring one-to-one mappings with a similarity threshold of 0.6.
Outputs the final mapping of tacticam IDs to broadcast IDs.


Output:

Saves annotated videos with bounding boxes and track IDs.
Saves a debug frame at frame 30 for each camera.
Prints the number of reliable tracks and the final player ID mappings.



Usage

Ensure input videos (broadcast.mp4, tacticam.mp4) and the YOLOv11 model (best.pt) are in the project directory.
Install dependencies:pip install opencv-python numpy ultralytics torch torchvision torchreid scipy scikit-learn


Run the main script:python main.py


Check the output:
Annotated videos: broadcast_out.mp4, tacticam_out.mp4
Debug frames: broadcast_debug_frame.jpg, tacticam_debug_frame.jpg
Console output: Number of tracks and player ID mappings.



Notes

The system assumes the YOLOv11 model is trained to detect players (classes 0, 1, 2).
The FRAME_OFFSET (65 frames) is specific to the provided videos; adjust if the offset differs.
If OSNet feature extraction fails, the system falls back to color histogram features.
The similarity threshold (0.6) for mapping can be adjusted in utils.py for stricter or looser matching.

Limitations

Requires synchronized videos with a known frame offset.
Performance depends on the quality of the YOLOv11 model and input video resolution.
Appearance-based matching may fail in cases of poor lighting or similar player appearances.

Future Improvements

Add team classification based on more robust color analysis.
Implement real-time processing optimizations.
Support dynamic frame offset detection for unsynchronized videos.

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort.deep_sort.deep_sort import DeepSort
from utils import get_center, match_by_appearance

# Load YOLOv11 model (custom-trained)
model = YOLO("best.pt")

# Input videos
video_paths = {
    "broadcast": "broadcast.mp4",
    "tacticam": "tacticam.mp4"
}

# Frame offset (broadcast starts when tacticam is at 4 seconds)
FRAME_OFFSET = 65

results = {}

for cam_name, path in video_paths.items():
    print(f"\nProcessing {cam_name} camera...")

    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(f"{cam_name}_out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    deepsort = DeepSort()
    all_tracks = {}
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip the first 120 frames for tacticam
        if cam_name == "tacticam" and frame_id < FRAME_OFFSET:
            frame_id += 1
            continue

        # YOLOv11 inference
        detections = model(frame)[0]
        boxes = []

        for det in detections.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) in [0, 1, 2] and conf > 0.5:
                boxes.append([x1, y1, x2 - x1, y2 - y1, conf])

        track_feature_pairs = deepsort.update_tracks(boxes, frame)

        for track, feat in track_feature_pairs:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            tlwh = track.to_tlwh()
            track_id = track.track_id
            x1, y1, w, h = tlwh
            x2, y2 = int(x1 + w), int(y1 + h)
            x1, y1 = int(x1), int(y1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            center = get_center([x1, y1, x2, y2])
            appearance = feat if feat is not None else np.zeros(512)

            if track_id not in all_tracks:
                all_tracks[track_id] = []
            all_tracks[track_id].append((frame_id, center[0], center[1], appearance))

        if frame_id == 30:
            cv2.imwrite(f"{cam_name}_debug_frame.jpg", frame)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    filtered_tracks = {
        tid: track for tid, track in all_tracks.items() if len(track) >= 3
    }
    results[cam_name] = filtered_tracks

    print(f"Finished processing {cam_name}, output saved to {cam_name}_out.mp4")
    print(f"Retained {len(filtered_tracks)} reliable tracks out of {len(all_tracks)} total.\n")

# Match player IDs using appearance features
print("\nMapping tacticam IDs to broadcast IDs based on appearance similarity...")
mapping = match_by_appearance(results["broadcast"], results["tacticam"])

print("\nFINAL PLAYER MAPPING:")
for tact_id, broad_id in mapping.items():
    print(f"Tacticam ID {tact_id} → Broadcast ID {broad_id}")

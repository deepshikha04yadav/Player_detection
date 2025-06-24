from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import extract_histogram, match_players
import os

# Load YOLOv11 model
model = YOLO("model/best.pt")

# Initialize DeepSORT
tracker_b = DeepSort(max_age=30)
tracker_t = DeepSort(max_age=30)

# Load videos
cap_b = cv2.VideoCapture("videos/broadcast.mp4")
cap_t = cv2.VideoCapture("videos/tacticam.mp4")

# Store features per tracked ID
features_b = {}
features_t = {}

def process_video(cap, tracker, label):
    id_features = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = map(int, r[:6])
            if cls == 0:  # class 0 = player
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "player"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x, y, w, h = map(int, track.to_ltrb())
            crop = frame[y:y+h, x:x+w]
            hist = extract_histogram(crop)
            if tid not in id_features:
                id_features[tid] = []
            id_features[tid].append(hist)

    return id_features

# Extract features from both videos
features_b = process_video(cap_b, tracker_b, "broadcast")
features_t = process_video(cap_t, tracker_t, "tacticam")

# Perform player mapping
player_mapping = match_players(features_t, features_b)

# Output mapping
print("Player Mapping (Tacticam → Broadcast):")
for t_id, b_id in player_mapping.items():
    print(f"T{t_id} → B{b_id}")


def visualize_and_save(video_path, output_path, tracker, model, id_translation=None, is_tacticam=False):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = map(int, r[:6])
            if cls == 0:  # player
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "player"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # Translate ID if tacticam and mapping is given
            display_id = tid
            if is_tacticam and id_translation and tid in id_translation:
                display_id = id_translation[tid]

            # Draw box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {display_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved output to {output_path}")

# Match players across videos
mapping = match_players(features_t, features_b)

# Print mapping
print("Player Mapping (Tacticam → Broadcast):")
for t_id, b_id in mapping.items():
    print(f"T{t_id} → B{b_id}")



# Re-initialize trackers
tracker_b = DeepSort(max_age=30)
tracker_t = DeepSort(max_age=30)

# Save broadcast with original IDs
visualize_and_save("videos/broadcast.mp4", "output/broadcast_output.mp4", tracker_b, model)

# Save tacticam with mapped IDs
visualize_and_save("videos/tacticam.mp4", "output/tacticam_output.mp4", tracker_t, model, id_translation=mapping, is_tacticam=True)

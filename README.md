# Player Detection and Identity Mapping Across Multi-Angle Videos

This project detects players and the ball from two video feeds (`broadcast.mp4` and `tacticam.mp4`) using a YOLOv11 model. It then matches the players across both videos to ensure consistent player IDs, enabling multi-camera tracking and analysis.

---

## 🎯 Objective

- Detect all players and the ball in both video feeds.
- Assign consistent `player_id`s across different camera angles.
- Output visualized results or mapped IDs for further analysis.

---

## 🧠 How It Works

1. **Object Detection**: YOLOv11 model detects players and ball in each frame.
2. **Feature Extraction**: Visual and spatial features are extracted.
3. **Player Mapping**: Players in `tacticam` are matched to `broadcast` feed using similarity metrics.
4. **Output**: Result includes consistent IDs overlayed on both videos.

---

## 🗂️ Project Structure
```
Player_detection/
├── model/
│   └── best.pt
├── output/
│   ├── broadcast_output.mp4 
│   └── tacticam_output.mp4
├── sort/
├── videos/
│   ├── broadcast.mp4
│   └── tacticam.mp4
├── main.py
└── utils.py
```


## 🔧 Setup Instructions

### 1. Clone the Repository and Install Dependencies

git clone https://github.com/deepshikha04yadav/Player_detection.git
cd your-project-directory

    pip install -r requirements.txt

If requirements.txt doesn't exist, run:

    pip install ultralytics opencv-python deep_sort_realtime numpy scipy


2. Download the Object Detection Model
Download the fine-tuned YOLOv11 model from this link:

🔗 https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view

Save it as:
```
model/best.pt
```

## 🚀 How It Works
### Step 1: Detection
The YOLOv11 model detects players (class 0) in each frame of both videos.

### Step 2: Tracking
Deep SORT tracks players in each video and assigns them unique temporary IDs.

### Step 3: Feature Extraction
For each detected player, we extract appearance features using color histograms.

### Step 4: Matching
We use cosine similarity between appearance features to map players from the tacticam view to the corresponding identities in the broadcast view.

### Step 5: Visualization
Two output videos are generated:
  * broadcast_output.mp4: shows original tracked IDs from broadcast.
  * tacticam_output.mp4: uses the mapping to show consistent player IDs as in broadcast.

## 🖥️ Running the Project
```
python main.py
```
#### The script will:

* Process both videos
* Match players
* Print the mapping (e.g., Tacticam ID 3 → Broadcast ID 7)
* Save two annotated videos in the output/ folder


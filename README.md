# 🏀 Basketball Analytics – Computer Vision Pipeline

A computer vision system that analyses basketball game footage and automatically detects **player tracking**, **team assignment**, **ball possession**, **passes & interceptions**, **shot events (MADE/MISSED)**, **jersey number recognition**, and **player speed & distance metrics** — all rendered as an annotated output video.

---

## 📁 Project Structure

```
BASKETBALL/
├── main.py                          # Entry point — runs the full pipeline
│
├── detection/                       # Player & ball detection + tracking
│   ├── player_detector.py           # YOLO + ByteTrack for player tracking
│   └── ball_detector.py             # YOLO-based ball detection & interpolation
│
├── possession/                      # Ball possession analysis
│   └── possession_detector.py       # Proximity + containment-based possession logic
│
├── classifier/                      # Team classification
│   └── team_classifier.py           # FashionCLIP-based jersey color classifier
│
├── event_detector/                  # Game event detection
│   └── event_detector.py            # Pass and interception detection
│
├── action_classifier/               # Shot event classification
│   └── action_classifier.py         # RF-DETR Roboflow model (MADE/MISSED via ShotEventTracker)
│
├── annotation/                      # Jersey number OCR
│   └── number_reader.py             # EasyOCR + IoS matching + temporal validation
│
├── motion_analyzer/                 # Speed & distance computation
│   └── motion_analyzer.py           # Pixel-to-meter conversion, km/h calculation
│
├── renderers/                       # Drawing / overlay modules
│   ├── player_renderer.py           # Ellipses, jersey number labels
│   ├── ball_renderer.py             # Ball triangle pointer
│   ├── control_renderer.py          # Team ball control % overlay
│   ├── events_renderer.py           # Pass & interception count overlay
│   ├── action_renderer.py           # MADE / MISSED banner
│   ├── motion_renderer.py           # Speed & distance text overlay
│   └── draw_helpers.py              # Shared drawing primitives
│
├── helpers/                         # Shared utilities
│   ├── io.py                        # Video read/write
│   ├── cache.py                     # Pickle stub read/write
│   └── geometry.py                  # BBox geometry helpers
│
├── settings/                        # Config constants
│   ├── action_config.py             # Roboflow model ID, class IDs, timing thresholds
│   └── annotation_config.py         # OCR class IDs, interval, IoS threshold
│
├── models/                          # Model weight files (see Drive link below)
│   ├── player.pt                    # YOLOv8 fine-tuned player detector
│   └── ball_detector_model.pt       # YOLOv8 fine-tuned ball detector
│
├── stubs/                           # Cached inference results (pickle files)
├── input_videos/                    # Raw input video clips
├── output_videos/                   # Processed output videos & graphs
├── training/                        # YOLO training notebooks
│   ├── bb_yolo.ipynb                # Player training pipeline
│   └── basketball_ball_training.ipynb # Ball detection training pipeline
│
├── .env                             # API keys (not committed)
├── requirements.txt
└── yolov8x.pt                       # Base YOLOv8x weights
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create & Activate Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate — Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activate — Windows CMD
venv\Scripts\activate.bat

# Activate — macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project root:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
```

> Get your free API key at [roboflow.com](https://roboflow.com)

---

## 📦 Download Models, Videos & Stubs

All model weights, input videos, cached stubs, and sample output videos are available on Google Drive:

> **📁 [Download from Google Drive](https://drive.google.com/drive/folders/1twaYMblu8oMnUmLMTplyIO6YZjJNkKzN?usp=sharing)**

After downloading, place the files as follows:

| Downloaded Folder | Place At |
|---|---|
| `models/` | `BASKETBALL/models/` |
| `input_videos/` | `BASKETBALL/input_videos/` |
| `stubs/` | `BASKETBALL/stubs/` |
| `output_videos/` | `BASKETBALL/output_videos/` (sample outputs) |

---

## ▶️ Running the Pipeline

```bash
# Run on a specific video
python main.py "input_videos/boston-celtics-new-york-knicks-game-1-q1-07.41-07.34.mp4"

# Run on the default test video (video_2.mp4)
python main.py
```

### Output files generated in `output_videos/`:

| File | Description |
|---|---|
| `output_<name>.avi` | Fully annotated output video |
| `output_<name>_shot_events.json` | JSON log of all MADE/MISSED events with timestamps |
| `output_<name>_ball_control_pie.png` | Team ball possession pie chart |
| `output_<name>_player_speed_profile.png` | Top 5 player speed-over-time graph |

---

## ⚙️ Pipeline Overview

```
Input Video
    │
    ▼
PlayerDetector (YOLOv8 + ByteTrack) ──► Tracks all players across frames
BallDetector   (YOLOv8)             ──► Tracks ball, removes outliers, interpolates
    │
    ▼
TeamClassifier (FashionCLIP)        ──► Assigns each player to Team 1 or Team 2
    │
    ▼
PossessionDetector                  ──► Determines which player has the ball per frame
    │
    ├──► GameEventDetector           ──► Detects passes & interceptions
    │
    ├──► ActionClassifier (RF-DETR)  ──► Classifies shot events (MADE / MISSED)
    │       └──► NumberAnnotator     ──► Reads jersey numbers via EasyOCR
    │
    └──► MotionAnalyzer              ──► Computes player speed (km/h) & distance (m)
    │
    ▼
Renderers (PlayerRenderer, BallRenderer, ControlRenderer, EventsRenderer,
           ActionRenderer, MotionRenderer)
    │
    ▼
Annotated Output Video + Analytics Graphs
```

---

## 🧠 Models Used

| Model | Purpose |
|---|---|
| YOLOv8 (fine-tuned) | Player detection |
| YOLOv8 (fine-tuned) | Ball detection |
| FashionCLIP (`patrickjohncyh/fashion-clip`) | Team jersey classification |
| RF-DETR via Roboflow (`basketball-player-detection-3-ycjdo/4`) | Shot event & jersey number detection |
| EasyOCR | Jersey digit recognition from crops |

---

## 📊 Analytics Outputs

- **Ball Possession %** — per-team ball control across the clip  
- **Pass & Interception Counts** — live overlay per team  
- **Shot Events** — MADE/MISSED banner with timestamp log  
- **Jersey Numbers** — OCR-confirmed numbers overlaid on player ellipses  
- **Speed & Distance** — real-time km/h and total meters per player  

---

## 📋 Requirements

```
opencv-python
numpy
pandas
Pillow
supervision
ultralytics
transformers
python-dotenv
inference
git+https://github.com/roboflow/sports.git@feat/basketball
easyocr
matplotlib
```

---

## 📝 Notes

- The `stubs/` folder caches heavy inference results (player tracks, team assignments) as `.pkl` files so the pipeline skips re-running models on subsequent runs of the same video.
- The first run on a new video will take significantly longer than subsequent runs.
- EasyOCR runs on CPU — no GPU required for jersey number recognition.
- The Roboflow model requires a valid `ROBOFLOW_API_KEY` in your `.env` file.

---

## 📄 License

This project is intended for academic and research use only.

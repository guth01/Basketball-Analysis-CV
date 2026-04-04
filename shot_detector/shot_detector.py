"""
shot_detector/shot_detector.py — Shot event detection (made/miss classification).

Loads the Roboflow model via the `inference` SDK and runs it per-frame to supply
class-level signals to `ShotEventTracker` from the `sports` library.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference import get_model
from sports.basketball import ShotEventTracker

from configs.shot_config import (
    ROBOFLOW_MODEL_ID,
    BALL_IN_BASKET_CLASS_ID,
    JUMP_SHOT_CLASS_ID,
    LAYUP_DUNK_CLASS_ID,
    SHOT_RESET_TIME_SECONDS,
    SHOT_MIN_BETWEEN_STARTS_SECONDS,
    SHOT_COOLDOWN_AFTER_MADE_SECONDS,
)

# Load .env so ROBOFLOW_API_KEY is available
load_dotenv()


class ShotDetector:
    """
    Detects shot events (made / missed) using class-level signals from
    the RF-DETR Roboflow model: player-jump-shot, player-layup-dunk, ball-in-basket.

    Usage:
        detector = ShotDetector(fps=30)
        for frame_index, frame in enumerate(video_frames):
            detections = detector.run_inference(frame)
            events = detector.update(frame_index, detections)
        detector.save_events(Path("output/shot_events.json"))
    """

    def __init__(self, fps: int = 30):
        self.fps = fps

        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ROBOFLOW_API_KEY is not set. Add it to your .env file."
            )

        print(f"[ShotDetector] Loading Roboflow model: {ROBOFLOW_MODEL_ID}")
        self.model = get_model(model_id=ROBOFLOW_MODEL_ID, api_key=api_key)

        self.tracker = ShotEventTracker(
            reset_time_frames=int(fps * SHOT_RESET_TIME_SECONDS),
            minimum_frames_between_starts=int(fps * SHOT_MIN_BETWEEN_STARTS_SECONDS),
            cooldown_frames_after_made=int(fps * SHOT_COOLDOWN_AFTER_MADE_SECONDS),
        )
        self.events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_inference(self, frame: np.ndarray) -> sv.Detections:
        """
        Run the Roboflow model on a single BGR frame (NumPy array).
        Returns a `supervision.Detections` object.
        """
        result = self.model.infer(frame)[0]
        return sv.Detections.from_inference(result)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self, frame_index: int, detections: sv.Detections
    ) -> Optional[List]:
        """
        Call once per frame with the detections produced by `run_inference`.
        Returns a list of resolved shot events if any occurred, else None.
        """
        has_jump_shot = (
            len(detections[detections.class_id == JUMP_SHOT_CLASS_ID]) > 0
        )
        has_layup_dunk = (
            len(detections[detections.class_id == LAYUP_DUNK_CLASS_ID]) > 0
        )
        has_ball_in_basket = (
            len(detections[detections.class_id == BALL_IN_BASKET_CLASS_ID]) > 0
        )

        events = self.tracker.update(
            frame_index=frame_index,
            has_jump_shot=has_jump_shot,
            has_layup_dunk=has_layup_dunk,
            has_ball_in_basket=has_ball_in_basket,
        )

        if events:
            for event in events:
                record = {
                    "frame_index": frame_index,
                    "time_seconds": round(frame_index / self.fps, 2),
                    "event": str(event),
                }
                self.events.append(record)
                print(f"  [ShotDetector] Frame {frame_index}: {event}")
            return events
        return None

    # ------------------------------------------------------------------
    # Batch helper
    # ------------------------------------------------------------------

    def process_video_frames(
        self, video_frames: List[np.ndarray]
    ) -> Dict[int, List]:
        """
        Convenience method: runs inference + update on every frame.

        Returns:
            shot_events_by_frame: mapping of frame_index -> list of events
                                  (only frames that produced events are present)
        """
        shot_events_by_frame: Dict[int, List] = {}
        total = len(video_frames)
        print(f"[ShotDetector] Processing {total} frames …")
        for frame_index, frame in enumerate(video_frames):
            if frame_index % 100 == 0:
                print(f"  [ShotDetector] Frame {frame_index}/{total}")
            detections = self.run_inference(frame)
            events = self.update(frame_index, detections)
            if events:
                shot_events_by_frame[frame_index] = events
        print(
            f"[ShotDetector] Done. Detected {len(self.events)} shot event(s)."
        )
        return shot_events_by_frame

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_events(self, path: Path) -> None:
        """Save all shot events to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.events, f, indent=2)
        print(f"[ShotDetector] Saved {len(self.events)} events → {path}")

    def reset(self) -> None:
        """Reset state for a new video."""
        self.events = []
        self.tracker = ShotEventTracker(
            reset_time_frames=int(self.fps * SHOT_RESET_TIME_SECONDS),
            minimum_frames_between_starts=int(
                self.fps * SHOT_MIN_BETWEEN_STARTS_SECONDS
            ),
            cooldown_frames_after_made=int(
                self.fps * SHOT_COOLDOWN_AFTER_MADE_SECONDS
            ),
        )

"""
ocr/jersey_recognizer.py — Jersey number recognition using EasyOCR.

Lightweight CPU-based OCR that reads jersey numbers from RF-DETR 'number'
class detections and validates them over time using ConsecutiveValueTracker.
"""

from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
import supervision as sv

from configs.ocr_config import (
    NUMBER_CLASS_ID,
    PLAYER_CLASS_ID,
    NUMBER_OCR_INTERVAL,
    NUMBER_CONSECUTIVE_THRESHOLD,
    NUMBER_IOS_THRESHOLD,
)


def compute_ios_matrix(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """
    Compute Intersection-over-Smaller (IoS) between two sets of bounding boxes.

    Args:
        boxes_a: (N, 4) array of [x1, y1, x2, y2] boxes.
        boxes_b: (M, 4) array of [x1, y1, x2, y2] boxes.

    Returns:
        (N, M) IoS matrix.
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    # Compute intersection
    x1 = np.maximum(boxes_a[:, 0].reshape(N, 1), boxes_b[:, 0].reshape(1, M))
    y1 = np.maximum(boxes_a[:, 1].reshape(N, 1), boxes_b[:, 1].reshape(1, M))
    x2 = np.minimum(boxes_a[:, 2].reshape(N, 1), boxes_b[:, 2].reshape(1, M))
    y2 = np.minimum(boxes_a[:, 3].reshape(N, 1), boxes_b[:, 3].reshape(1, M))

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # IoS = intersection / min(area_a, area_b)
    smaller = np.minimum(area_a.reshape(N, 1), area_b.reshape(1, M))
    smaller = np.maximum(smaller, 1e-6)  # avoid division by zero

    return intersection / smaller


def coords_above_threshold(
    matrix: np.ndarray, threshold: float, sort_desc: bool = True
) -> List[Tuple[int, int]]:
    """
    Return all (row_index, col_index) where value > threshold.
    Optionally sort by value descending.
    """
    rows, cols = np.where(matrix > threshold)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    if sort_desc:
        pairs.sort(key=lambda rc: matrix[rc[0], rc[1]], reverse=True)
    return pairs


class JerseyNumberRecognizer:
    """
    Detects jersey numbers in frames and matches them to tracked players.

    Uses:
      - RF-DETR 'number' class detections (from shared ShotDetector inference)
      - EasyOCR for digit recognition (CPU-only, lightweight)
      - IoS (Intersection over Smaller) matching to link numbers to players
      - ConsecutiveValueTracker for temporal validation
    """

    def __init__(self):
        """Initialize with EasyOCR reader and ConsecutiveValueTracker."""
        import easyocr
        from sports import ConsecutiveValueTracker

        print("[JerseyOCR] Loading EasyOCR reader (digits only, CPU)...")
        self.reader = easyocr.Reader(
            ['en'],
            gpu=False,
            verbose=False,
        )
        self.validator = ConsecutiveValueTracker(
            n_consecutive=NUMBER_CONSECUTIVE_THRESHOLD
        )
        print("[JerseyOCR] EasyOCR ready.")

    def _recognize_crops(
        self, frame: np.ndarray, number_detections: sv.Detections
    ) -> List[str]:
        """
        Run EasyOCR on each number crop to read the jersey digit(s).

        Preprocessing pipeline:
          1. Convert to grayscale
          2. Upscale 2.5x for better OCR accuracy
          3. CLAHE for contrast normalization across jersey colors
          4. Gaussian blur to reduce noise
        """
        frame_h, frame_w = frame.shape[:2]

        # Pad boxes slightly for context, then clip to frame bounds
        padded_boxes = sv.pad_boxes(xyxy=number_detections.xyxy, px=10, py=10)
        padded_boxes = np.clip(
            padded_boxes,
            [0, 0, 0, 0],
            [frame_w, frame_h, frame_w, frame_h],
        )

        numbers = []
        for xyxy in padded_boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                numbers.append("")
                continue

            try:
                # 1. Grayscale
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # 2. Upscale
                gray = cv2.resize(
                    gray, None, fx=2.5, fy=2.5,
                    interpolation=cv2.INTER_CUBIC
                )
                # 3. CLAHE — normalizes contrast across different jersey colors
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                # 4. Blur to reduce upscaling artifacts
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

                results = self.reader.readtext(
                    gray,
                    detail=0,
                    allowlist='0123456789',
                    paragraph=True,
                )
                text = "".join(results).strip() if results else ""
                numbers.append(text)
            except Exception as e:
                print(f"  [JerseyOCR] EasyOCR failed on crop: {e}")
                numbers.append("")

        return numbers

    def _match_numbers_to_players(
        self,
        player_boxes: np.ndarray,
        number_boxes: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Match number detection boxes to player boxes using IoS.

        Args:
            player_boxes: (N, 4) array of player bounding boxes.
            number_boxes: (M, 4) array of number bounding boxes.

        Returns:
            List of (player_idx, number_idx) pairs above IoS threshold.
        """
        if len(player_boxes) == 0 or len(number_boxes) == 0:
            return []

        ios_matrix = compute_ios_matrix(player_boxes, number_boxes)
        return coords_above_threshold(ios_matrix, NUMBER_IOS_THRESHOLD)

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        player_tracks_for_frame: Dict[int, Dict],
        detections: sv.Detections,
    ):
        """
        Process a single frame for jersey number recognition.

        Only runs OCR every NUMBER_OCR_INTERVAL frames for performance.

        Args:
            frame: BGR video frame (numpy array).
            frame_index: Current frame number.
            player_tracks_for_frame: Dict of {track_id: {"bbox": [x1,y1,x2,y2]}}
                                     from the PlayerTracker for this frame.
            detections: Full sv.Detections from the RF-DETR model inference
                        (shared with ShotDetector).
        """
        if frame_index % NUMBER_OCR_INTERVAL != 0:
            return

        # Filter for 'number' class detections
        number_mask = detections.class_id == NUMBER_CLASS_ID
        number_detections = detections[number_mask]

        if len(number_detections) == 0:
            return

        if len(player_tracks_for_frame) == 0:
            return

        # Build player boxes array and track ID list from our track dict
        track_ids = list(player_tracks_for_frame.keys())
        player_boxes = np.array([
            player_tracks_for_frame[tid]["bbox"] for tid in track_ids
        ])

        # Recognize digits via EasyOCR
        numbers = self._recognize_crops(frame, number_detections)

        # Match number boxes to player boxes via IoS
        pairs = self._match_numbers_to_players(
            player_boxes, number_detections.xyxy
        )
        if not pairs:
            return

        player_indices, number_indices = zip(*pairs)

        matched_tracker_ids = [track_ids[i] for i in player_indices]
        matched_numbers = [numbers[i] for i in number_indices]

        # Feed to the temporal validator
        self.validator.update(
            tracker_ids=matched_tracker_ids,
            values=matched_numbers,
        )

    def get_labels(self, tracker_ids: List[int]) -> Dict[int, str]:
        """
        Get validated jersey numbers for the given tracker IDs.

        Returns:
            Dict mapping track_id -> "#XX" for validated players.
            Only includes players with confirmed numbers.
        """
        if not tracker_ids:
            return {}

        validated = self.validator.get_validated(tracker_ids=tracker_ids)
        result = {}
        for tid, val in zip(tracker_ids, validated):
            if val:
                result[tid] = f"#{val}"
        return result

    def reset(self):
        """Reset validator state for a new video."""
        from sports import ConsecutiveValueTracker
        self.validator = ConsecutiveValueTracker(
            n_consecutive=NUMBER_CONSECUTIVE_THRESHOLD
        )

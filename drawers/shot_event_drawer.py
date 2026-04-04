"""
drawers/shot_event_drawer.py — Draws MADE / MISSED shot-event overlays on video frames.

The overlay displays for a configurable number of frames after the event is detected,
then fades away.  MADE events are shown in green; MISSED events in red.
"""

from typing import List, Dict, Any
import cv2
import numpy as np


# How many frames to keep the banner visible after an event fires
DISPLAY_DURATION_FRAMES = 90  # ~3 s at 30 fps


class ShotEventDrawer:
    """
    Draws a centred MADE / MISSED banner on each video frame whenever a shot
    event has been detected.

    Usage in main pipeline::

        drawer = ShotEventDrawer(fps=30)
        output_frames = drawer.draw(output_video_frames, shot_events_by_frame)
    """

    # Colours (BGR)
    MADE_COLOR = (0, 200, 60)      # vivid green
    MISSED_COLOR = (30, 30, 220)   # vivid red (BGR)
    TEXT_COLOR = (255, 255, 255)   # white text

    def __init__(self, fps: int = 30, display_duration_frames: int = DISPLAY_DURATION_FRAMES):
        self.fps = fps
        self.display_duration_frames = display_duration_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        video_frames: List[np.ndarray],
        shot_events_by_frame: Dict[int, List],
    ) -> List[np.ndarray]:
        """
        Overlay shot-event banners on every frame where an event is active.

        Args:
            video_frames:         List of BGR frames (already drawn by other drawers).
            shot_events_by_frame: Mapping of {frame_index: [event, …]} returned by
                                  ``ShotDetector.process_video_frames()``.

        Returns:
            A new list of frames with the banners composited in.
        """
        output_frames = []

        # Pre-compute a flat timeline: for each frame, what label is active?
        active_label_per_frame = self._build_label_timeline(
            total_frames=len(video_frames),
            shot_events_by_frame=shot_events_by_frame,
        )

        for frame_index, frame in enumerate(video_frames):
            label = active_label_per_frame.get(frame_index)
            if label is not None:
                frame = self._draw_banner(frame.copy(), label)
            output_frames.append(frame)

        return output_frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_label_timeline(
        self,
        total_frames: int,
        shot_events_by_frame: Dict[int, List],
    ) -> Dict[int, str]:
        """
        Returns a dict mapping frame_index -> label ("MADE" | "MISSED")
        for every frame that should show a banner.
        """
        timeline: Dict[int, str] = {}

        for event_frame, events in sorted(shot_events_by_frame.items()):
            label = self._classify_events(events)
            if label is None:
                continue
            end_frame = min(event_frame + self.display_duration_frames, total_frames)
            for f in range(event_frame, end_frame):
                # Earlier events take priority (don't overwrite)
                if f not in timeline:
                    timeline[f] = label

        return timeline

    @staticmethod
    def _classify_events(events: List) -> str | None:
        """
        Inspect a list of `ShotEvent` objects and return "MADE", "MISSED", or None.
        Converts to string and checks for keywords (robust to different enum formats).
        """
        for event in events:
            event_str = str(event).upper()
            if "MADE" in event_str:
                return "MADE"
            if "MISSED" in event_str or "MISS" in event_str:
                return "MISSED"
        return None

    def _draw_banner(self, frame: np.ndarray, label: str) -> np.ndarray:
        """
        Composite a centred pill-shaped banner onto *frame* (in-place copy).
        """
        h, w = frame.shape[:2]

        is_made = label == "MADE"
        bg_color = self.MADE_COLOR if is_made else self.MISSED_COLOR

        # --- Banner geometry ---
        banner_w = int(w * 0.30)
        banner_h = int(h * 0.09)
        cx, cy = w // 2, int(h * 0.12)

        x1 = cx - banner_w // 2
        y1 = cy - banner_h // 2
        x2 = cx + banner_w // 2
        y2 = cy + banner_h // 2
        radius = banner_h // 2

        # Semi-transparent background (blend onto a copy)
        overlay = frame.copy()
        # Draw rounded rectangle via filled rect + two circles on ends
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), bg_color, cv2.FILLED)
        cv2.circle(overlay, (x1 + radius, cy), radius, bg_color, cv2.FILLED)
        cv2.circle(overlay, (x2 - radius, cy), radius, bg_color, cv2.FILLED)
        alpha = 0.82
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # --- Emoji-style icon dot ---
        icon_radius = banner_h // 5
        icon_cx = x1 + radius + icon_radius + 4
        icon_cy = cy
        cv2.circle(frame, (icon_cx, icon_cy), icon_radius, self.TEXT_COLOR, cv2.FILLED)

        # --- Label text ---
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = banner_h / 55.0
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = cx - tw // 2 + icon_radius + 4
        text_y = cy + th // 2 - 2
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, self.TEXT_COLOR, thickness, cv2.LINE_AA)

        return frame

# configs/shot_config.py — Shot detection constants

# --- Roboflow model ---
ROBOFLOW_MODEL_ID = "basketball-player-detection-3-ycjdo/4"

# --- Class IDs from the RF-DETR model ---
BALL_IN_BASKET_CLASS_ID = 1
JUMP_SHOT_CLASS_ID = 5
LAYUP_DUNK_CLASS_ID = 6

# --- Timing constants (seconds) ---
SHOT_RESET_TIME_SECONDS = 1.7
SHOT_MIN_BETWEEN_STARTS_SECONDS = 0.5
SHOT_COOLDOWN_AFTER_MADE_SECONDS = 0.5

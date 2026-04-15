# configs/ocr_config.py — Jersey number OCR constants

# --- Class IDs from the RF-DETR model (alphabetical order) ---
# ball=0, ball-in-basket=1, number=2, player=3, player-in-possession=4,
# player-jump-shot=5, player-layup-dunk=6, player-shot-block=7, referee=8, rim=9
NUMBER_CLASS_ID = 2
PLAYER_CLASS_ID = 3

# --- OCR timing ---
# Run OCR every N frames (higher = faster pipeline, lower = more accurate)
NUMBER_OCR_INTERVAL = 5

# --- Validation ---
# How many consecutive agreeing OCR reads before a number is "confirmed"
NUMBER_CONSECUTIVE_THRESHOLD = 3

# --- Matching ---
# Intersection-over-Smaller threshold to link a number box to a player box
NUMBER_IOS_THRESHOLD = 0.3

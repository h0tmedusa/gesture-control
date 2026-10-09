"""
Configuration Settings for Gesture Control System
All tunable parameters, thresholds, and controls are centralized here.
"""

# ==========================================
# 1. Camera & Video Pipeline
# ==========================================
CAMERA_INDEX = 0               # 0 is typically the built-in laptop camera
FRAME_WIDTH = 640              # Capture resolution width
FRAME_HEIGHT = 480             # Capture resolution height
MIRROR_FEED = True             # Flip frame horizontally to mirror natural hand movement
MODEL_PATH = "models/hand_landmarker.task"

# ==========================================
# 2. Hand Tracking Confidence
# ==========================================
DETECTION_CONFIDENCE = 0.45    # Lower value maintains tracking during quick motion
TRACKING_CONFIDENCE = 0.45

# ==========================================
# 3. Gesture Thresholds & Hysteresis
# ==========================================
PINCH_IN_RATIO = 0.28          # Thumb-to-Index distance ratio required to ENGAGE pinch
PINCH_OUT_RATIO = 0.38         # Thumb-to-Index distance ratio required to RELEASE pinch (hysteresis)
FINGER_STRAIGHTNESS_MIN = 0.82 # Minimum bone straightness ratio (0.0 to 1.0) to register as fully stretched

# ==========================================
# 4. Pinch & Slide Motion Physics
# ==========================================
DEADZONE_RATIO = 0.015         # Minimum movement required (% of hand scale) to prevent idle tremor jitter
BASE_SPEED_FACTOR = 3.8        # Multiplier for slow, steady scrolling
SMOOTHING_ALPHA = 0.75         # Exponential Moving Average responsiveness (0.0=sluggish, 1.0=raw)
FLICK_ACCELERATION = 12.0      # Non-linear multiplier scaling up high-speed flicks
FLICK_MIN_VELOCITY = 15.0      # Minimum velocity (px/frame) to trigger momentum coasting on release
FRICTION_DECAY = 0.88          # Deceleration factor during flick coasting (0.80 = abrupt, 0.95 = long glide)
GHOST_FRAME_TOLERANCE = 2      # Frames of lost tracking to tolerate before dropping active pinch clutch

# ==========================================
# 5. Continuous Hands-Free Point Scrolling
# ==========================================
POINT_SCROLL_SPEED = 15        # Scroll ticks dispatched per frame during hands-free point scroll
POINT_HOLD_DELAY_SECONDS = 0.25 # Time in seconds to hold Point Up/Down pose before scrolling begins

# ==========================================
# 6. Discrete Media Actions (Play / Pause)
# ==========================================
MEDIA_COOLDOWN_SECONDS = 1.2   # Cooldown duration to prevent rapid repeat triggers of Play/Pause
SCROLL_MULTIPLIER = 1.2        # Master OS scroll multiplier for PyAutoGUI
PRIMARY_MEDIA_KEY = "playpause"# Native multimedia key ('playpause', fallback is 'space')

# ==========================================
# 7. Safety & Fail-safes
# ==========================================
ENABLE_FAILSAFE = True         # Slap cursor into screen corner to kill script
TOGGLE_KEY = "t"               # Keyboard shortcut to pause/resume gesture recognition
QUIT_KEY = "q"                 # Keyboard shortcut to exit application


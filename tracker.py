"""
Phase 2: Hand Landmark Tracking Module
Encapsulates MediaPipe Tasks HandLandmarker into a reusable class.
Provides pixel coordinates, normalized coordinates, hand scale reference,
and visual skeleton rendering.
"""

import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 21 Hand Landmark Connections for drawing skeleton
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17)
]


class HandTracker:
    def __init__(self, model_path: str = "models/hand_landmarker.task", max_hands: int = 1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.latest_result = None

    def process_frame(self, frame_bgr, timestamp_ms: int):
        """
        Runs hand tracking on a BGR OpenCV frame.
        Requires monotonically increasing timestamp_ms.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.latest_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return self.latest_result

    def get_landmarks(self, frame_shape):
        """
        Extracts landmark coordinates for the primary detected hand.
        Returns:
            landmarks_px: dict mapping landmark_id (0-20) -> (x_px, y_px)
            landmarks_norm: dict mapping landmark_id (0-20) -> (x_norm, y_norm, z_norm)
            hand_scale: Euclidean distance between Wrist (0) and Middle MCP (9) in pixels.
        Returns (None, None, 0.0) if no hand is detected.
        """
        if not self.latest_result or not self.latest_result.hand_landmarks:
            return None, None, 0.0

        h, w = frame_shape[:2]
        primary_hand = self.latest_result.hand_landmarks[0]

        landmarks_px = {}
        landmarks_norm = {}

        for idx, lm in enumerate(primary_hand):
            px_x = int(lm.x * w)
            px_y = int(lm.y * h)
            landmarks_px[idx] = (px_x, px_y)
            landmarks_norm[idx] = (lm.x, lm.y, lm.z)

        # Scale reference: Distance between Wrist (0) and Middle Finger Base (9)
        # Used for distance-independent gesture thresholds
        wrist = landmarks_px[0]
        middle_mcp = landmarks_px[9]
        hand_scale = math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])
        hand_scale = max(hand_scale, 1.0)  # Avoid division by zero

        return landmarks_px, landmarks_norm, hand_scale

    def draw_skeleton(self, frame, landmarks_px):
        """Draws clean skeleton connections and joint circles."""
        if not landmarks_px:
            return

        # Draw bone connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = landmarks_px[start_idx]
            pt2 = landmarks_px[end_idx]
            cv2.line(frame, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)

        # Draw joints
        for idx, pt in landmarks_px.items():
            # Highlight index tip (8) and thumb tip (4) differently for slide control
            if idx in (4, 8):
                cv2.circle(frame, pt, 6, (0, 165, 255), -1, cv2.LINE_AA)  # Orange for pinch tips
                cv2.circle(frame, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, (0, 255, 128), -1, cv2.LINE_AA)

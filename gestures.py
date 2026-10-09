"""
Phase 3: Gesture Classifier Module
Detects:
1. Pinch Clutch State (Thumb tip + Index tip touching/close) normalized by hand scale.
2. Discrete Poses:
   - "PLAY_PAUSE" (Peace Sign / Victory: Index and Middle fingers up, Ring & Pinky down).
   - "FIST" (All fingers folded).
   - "OPEN_PALM" (All fingers extended).
   - "NEUTRAL"
"""

import math


class GestureClassifier:
    def __init__(
        self,
        pinch_in_ratio: float = 0.28,
        pinch_out_ratio: float = 0.38,
        pinch_threshold_ratio: float = None,
    ):
        """
        pinch_in_ratio: threshold to ENTER pinch state.
        pinch_out_ratio: threshold to EXIT pinch state (hysteresis).
        pinch_threshold_ratio: legacy backward-compatible alias.
        """
        if pinch_threshold_ratio is not None:
            pinch_in_ratio = pinch_threshold_ratio
            pinch_out_ratio = pinch_threshold_ratio + 0.10

        self.pinch_in_ratio = pinch_in_ratio
        self.pinch_out_ratio = pinch_out_ratio
        self.pinch_threshold_ratio = pinch_in_ratio  # backwards compatibility attribute
        self.is_currently_pinched = False

    def get_finger_states(self, landmarks_px):
        """
        Determines which of the 5 fingers are extended/up.
        Returns dict: {'thumb': bool, 'index': bool, 'middle': bool, 'ring': bool, 'pinky': bool}
        """
        if not landmarks_px:
            return None

        # Landmarks:
        # Thumb: tip=4, ip=3, mcp=2
        # Index: tip=8, pip=6, mcp=5
        # Middle: tip=12, pip=10, mcp=9
        # Ring: tip=16, pip=14, mcp=13
        # Pinky: tip=20, pip=18, mcp=17

        # For fingers 8, 12, 16, 20: finger is "up" if tip Y is higher (smaller Y in image coords) than PIP joint Y
        index_up = landmarks_px[8][1] < landmarks_px[6][1]
        index_down = landmarks_px[8][1] > landmarks_px[6][1]
        middle_up = landmarks_px[12][1] < landmarks_px[10][1]
        ring_up = landmarks_px[16][1] < landmarks_px[14][1]
        pinky_up = landmarks_px[20][1] < landmarks_px[18][1]

        # For thumb: compare distance from thumb tip to pinky MCP (17) vs thumb IP (3) to pinky MCP
        # If tip is further away from the palm base (17) than joint 3, it is extended
        thumb_tip = landmarks_px[4]
        thumb_ip = landmarks_px[3]
        pinky_mcp = landmarks_px[17]
        dist_tip_to_palm = math.hypot(thumb_tip[0] - pinky_mcp[0], thumb_tip[1] - pinky_mcp[1])
        dist_ip_to_palm = math.hypot(thumb_ip[0] - pinky_mcp[0], thumb_ip[1] - pinky_mcp[1])
        thumb_up = dist_tip_to_palm > dist_ip_to_palm

        return {
            "thumb": thumb_up,
            "index": index_up,
            "index_down": index_down,
            "middle": middle_up,
            "ring": ring_up,
            "pinky": pinky_up,
        }

    def check_pinch(self, landmarks_px, hand_scale):
        """
        Returns (is_pinched, normalized_distance, pinch_center_point).
        Uses hysteresis: easier to stay pinched than to initiate.
        """
        if not landmarks_px or hand_scale <= 0:
            self.is_currently_pinched = False
            return False, 0.0, None

        thumb_tip = landmarks_px[4]
        index_tip = landmarks_px[8]

        raw_dist = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
        norm_dist = raw_dist / hand_scale

        if self.is_currently_pinched:
            # Must exceed the larger threshold to release
            if norm_dist > self.pinch_out_ratio:
                self.is_currently_pinched = False
        else:
            # Must be closer than smaller threshold to engage
            if norm_dist < self.pinch_in_ratio:
                self.is_currently_pinched = True

        pinch_center = (
            int((thumb_tip[0] + index_tip[0]) / 2),
            int((thumb_tip[1] + index_tip[1]) / 2),
        )

        return self.is_currently_pinched, norm_dist, pinch_center

    def classify_pose(self, finger_states, is_pinched):
        """
        Classifies static hand pose:
        - PINCH_DRAG: Thumb and index pinched together (drag/flick mode).
        - PLAY_PAUSE: Index and Middle fingers up, Ring & Pinky down (Peace sign).
        - POINT_UP: Only Index finger up, Middle/Ring/Pinky down (hands-free scroll up).
        - POINT_DOWN: Only Index finger down, Middle/Ring/Pinky folded (hands-free scroll down).
        - OPEN_PALM: All fingers extended.
        - FIST: All fingers closed.
        - NEUTRAL: Default resting state.
        """
        if not finger_states:
            return "NO_HAND"

        if is_pinched:
            return "PINCH_DRAG"

        # Peace Sign (Victory): Index & Middle extended, Ring & Pinky folded
        if (
            finger_states["index"]
            and finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
        ):
            return "PLAY_PAUSE"

        # Point UP: Only Index up, Middle, Ring & Pinky curled
        if (
            finger_states["index"]
            and not finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
        ):
            return "POINT_UP"

        # Point DOWN (Rock-On / Horns): Index and Pinky extended, Middle and Ring curled
        if (
            finger_states["index"]
            and finger_states["pinky"]
            and not finger_states["middle"]
            and not finger_states["ring"]
        ):
            return "POINT_DOWN"

        # All fingers open
        if (
            finger_states["thumb"]
            and finger_states["index"]
            and finger_states["middle"]
            and finger_states["ring"]
            and finger_states["pinky"]
        ):
            return "OPEN_PALM"

        # All fingers closed
        if (
            not finger_states["index"]
            and not finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
        ):
            return "FIST"

        return "NEUTRAL"

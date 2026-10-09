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
        finger_straightness_min: float = 0.82,
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
        self.finger_straightness_min = finger_straightness_min
        self.is_currently_pinched = False

    def get_finger_states(self, landmarks_px):
        """
        Determines which fingers are completely stretched/extended vs curled.
        Uses full extension ratio (tip-to-mcp distance vs bone segments sum)
        and joint ordering so partial curls do NOT trigger.
        """
        if not landmarks_px:
            return None

        # Helper to test full extension of a finger:
        # mcp -> pip -> dip -> tip
        # When completely stretched:
        # 1. tip Y is well above DIP Y and PIP Y (in upright hand)
        # 2. Euclidean distance between tip and mcp must be > 80% of total finger bone length
        def is_finger_fully_stretched(mcp_idx, pip_idx, dip_idx, tip_idx):
            mcp = landmarks_px[mcp_idx]
            pip = landmarks_px[pip_idx]
            dip = landmarks_px[dip_idx]
            tip = landmarks_px[tip_idx]

            # 1. Monotonic joint ordering (tip is above dip, dip is above pip)
            if not (tip[1] < dip[1] < pip[1]):
                return False

            # 2. Segment length ratio check (straightness)
            seg1 = math.hypot(pip[0] - mcp[0], pip[1] - mcp[1])
            seg2 = math.hypot(dip[0] - pip[0], dip[1] - pip[1])
            seg3 = math.hypot(tip[0] - dip[0], tip[1] - dip[1])
            total_bones = seg1 + seg2 + seg3

            span = math.hypot(tip[0] - mcp[0], tip[1] - mcp[1])
            straightness = span / max(total_bones, 1.0)
            return straightness > self.finger_straightness_min

        index_fully_stretched = is_finger_fully_stretched(5, 6, 7, 8)
        middle_fully_stretched = is_finger_fully_stretched(9, 10, 11, 12)
        ring_fully_stretched = is_finger_fully_stretched(13, 14, 15, 16)
        pinky_fully_stretched = is_finger_fully_stretched(17, 18, 19, 20)

        # Basic "up" check for fingers being curled/folded (for detecting that other fingers are curled)
        # A finger is considered folded if its tip is below its PIP
        index_curled = landmarks_px[8][1] > landmarks_px[6][1]
        middle_curled = landmarks_px[12][1] > landmarks_px[10][1]
        ring_curled = landmarks_px[16][1] > landmarks_px[14][1]
        pinky_curled = landmarks_px[20][1] > landmarks_px[18][1]

        # Thumb extension check
        thumb_tip = landmarks_px[4]
        thumb_ip = landmarks_px[3]
        pinky_mcp = landmarks_px[17]
        dist_tip_to_palm = math.hypot(thumb_tip[0] - pinky_mcp[0], thumb_tip[1] - pinky_mcp[1])
        dist_ip_to_palm = math.hypot(thumb_ip[0] - pinky_mcp[0], thumb_ip[1] - pinky_mcp[1])
        thumb_up = dist_tip_to_palm > dist_ip_to_palm

        return {
            "thumb": thumb_up,
            "index": index_fully_stretched,
            "middle": middle_fully_stretched,
            "ring": ring_fully_stretched,
            "pinky": pinky_fully_stretched,
            "index_curled": index_curled,
            "middle_curled": middle_curled,
            "ring_curled": ring_curled,
            "pinky_curled": pinky_curled,
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
        - FIST: All four main fingers closed (Neutral/idle state, overrides pinch).
        - PINCH_DRAG: Thumb and index pinched together (drag/flick mode).
        - PLAY_PAUSE: Index and Middle fingers up, Ring & Pinky down (Peace sign).
        - POINT_UP: Only Index finger up, Middle/Ring/Pinky down (hands-free scroll up).
        - POINT_DOWN: Index + Pinky up (Horns/Rock-on pose, hands-free scroll down).
        - OPEN_PALM: All fingers extended.
        - NEUTRAL: Default resting state.
        """
        if not finger_states:
            return "NO_HAND"

        # Check FIST first: if index, middle, ring, and pinky are all folded, it's a FIST
        if (
            not finger_states["index"]
            and not finger_states["middle"]
            and not finger_states["ring"]
            and not finger_states["pinky"]
        ):
            self.is_currently_pinched = False
            return "FIST"

        if is_pinched:
            return "PINCH_DRAG"

        # Peace Sign (Play/Pause): Index and Middle FULLY stretched, Ring and Pinky clearly CURLED
        if (
            finger_states["index"]
            and finger_states["middle"]
            and finger_states["ring_curled"]
            and finger_states["pinky_curled"]
        ):
            return "PLAY_PAUSE"

        # Point UP: Only Index FULLY stretched, Middle, Ring & Pinky clearly CURLED
        if (
            finger_states["index"]
            and finger_states["middle_curled"]
            and finger_states["ring_curled"]
            and finger_states["pinky_curled"]
        ):
            return "POINT_UP"

        # Point DOWN (Rock-On): Index and Pinky FULLY stretched, Middle and Ring clearly CURLED
        if (
            finger_states["index"]
            and finger_states["pinky"]
            and finger_states["middle_curled"]
            and finger_states["ring_curled"]
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

        return "NEUTRAL"

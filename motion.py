"""
Phase 4: Slide Tracker & Motion Physics Engine
Translates vertical hand motion during pinch clutch into smooth,
deadzone-filtered, scale-invariant velocity and scroll steps.
"""

import math
import time


class SlideTracker:
    def __init__(
        self,
        deadzone_ratio: float = 0.04,     # Minimum movement required (% of hand scale) to prevent tremor jitter
        speed_factor: float = 2.5,        # Multiplier to tune feel
        smoothing_alpha: float = 0.6,     # EMA smoothing: 0.0 (very sluggish) to 1.0 (raw unbuffered)
    ):
        self.deadzone_ratio = deadzone_ratio
        self.speed_factor = speed_factor
        self.smoothing_alpha = smoothing_alpha

        # Internal state
        self.is_tracking = False
        self.prev_y = None
        self.smoothed_delta_y = 0.0
        self.accumulated_scroll = 0.0

    def reset(self):
        """Resets tracking state when clutch is disengaged or hand leaves frame."""
        self.is_tracking = False
        self.prev_y = None
        self.smoothed_delta_y = 0.0
        self.accumulated_scroll = 0.0

    def update(self, is_pinched: bool, current_pos: tuple, hand_scale: float):
        """
        Updates motion state.
        Args:
            is_pinched: True if fingers are pinched.
            current_pos: (x, y) coordinates of pinch midpoint in pixels.
            hand_scale: Pixel distance across hand reference.
        Returns:
            scroll_delta: Integer or float units ready for scroll execution.
                          Positive = Scroll Up, Negative = Scroll Down.
            velocity_display: Current speed in pixels/frame for visualization.
        """
        if not is_pinched or current_pos is None or hand_scale <= 0:
            self.reset()
            return 0, 0.0

        current_y = current_pos[1]

        # First frame of pinch clutch engagement (touch-down)
        if not self.is_tracking or self.prev_y is None:
            self.is_tracking = True
            self.prev_y = current_y
            self.smoothed_delta_y = 0.0
            return 0, 0.0

        # Calculate raw displacement in pixels
        raw_dy = current_y - self.prev_y
        self.prev_y = current_y

        # Deadzone filter: ignore micro-tremors based on hand size
        deadzone_px = self.deadzone_ratio * hand_scale
        if abs(raw_dy) < deadzone_px:
            filtered_dy = 0.0
        else:
            # Subtract deadzone so motion doesn't "snap" abruptly
            sign = 1.0 if raw_dy > 0 else -1.0
            filtered_dy = (abs(raw_dy) - deadzone_px) * sign

        # Exponential Moving Average (EMA) smoothing for fluid motion
        self.smoothed_delta_y = (
            self.smoothing_alpha * filtered_dy
            + (1.0 - self.smoothing_alpha) * self.smoothed_delta_y
        )

        if abs(self.smoothed_delta_y) < 0.1:
            return 0, 0.0

        # Scale-invariant normalization:
        # Normalize by hand size so physical speed is identical at any camera distance
        norm_dy = self.smoothed_delta_y / hand_scale

        # Touchscreen convention:
        # Invert direction: moving hand UP (negative dy) scrolls page DOWN or UP depending on preference.
        # Natural touchscreen: dragging UP pushes content UP (scroll down).
        # We produce: Moving hand UP (dy < 0) -> +scroll (scroll content up)
        #            Moving hand DOWN (dy > 0) -> -scroll (scroll content down)
        scroll_units = -norm_dy * self.speed_factor * 100.0

        return scroll_units, -self.smoothed_delta_y

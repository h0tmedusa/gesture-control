"""
Slide Tracker & Motion Physics Engine (with Flick Persistence & Inertia Decay)
Translates vertical hand motion during pinch clutch into smooth,
deadzone-filtered, scale-invariant velocity, and handles flick release glide.
"""

import math
import time
from collections import deque


class SlideTracker:
    def __init__(
        self,
        deadzone_ratio: float = 0.015,    # Much tighter deadzone (was 0.035): picks up subtle, slow finger crawls
        speed_factor: float = 3.8,        # Increased base speed (was 2.8): more scroll ticks for gentle movement
        smoothing_alpha: float = 0.75,    # Higher responsiveness (was 0.65): reduces lag on small initial drifts
        flick_acceleration: float = 12.0, # Exponential sensitivity for fast flicks
        flick_min_velocity: float = 15.0, # Minimum velocity threshold to trigger glide after release
        friction_decay: float = 0.88,     # Coasting decay factor
        ghost_frame_tolerance: int = 2,   # Tolerate up to 2 lost frames before resetting clutch
    ):
        self.deadzone_ratio = deadzone_ratio
        self.speed_factor = speed_factor
        self.smoothing_alpha = smoothing_alpha
        self.flick_acceleration = flick_acceleration
        self.flick_min_velocity = flick_min_velocity
        self.friction_decay = friction_decay
        self.ghost_frame_tolerance = ghost_frame_tolerance

        # Internal state
        self.is_tracking = False
        self.prev_y = None
        self.smoothed_delta_y = 0.0
        self.recent_velocities = deque(maxlen=4)
        self.ghost_frames_left = 0

        # Inertia coasting state
        self.is_coasting = False
        self.coast_velocity = 0.0

    def reset(self):
        """Full reset of motion state."""
        self.is_tracking = False
        self.prev_y = None
        self.smoothed_delta_y = 0.0
        self.recent_velocities.clear()
        self.ghost_frames_left = 0
        self.is_coasting = False
        self.coast_velocity = 0.0

    def update(self, is_pinched: bool, current_pos: tuple, hand_scale: float):
        """
        Updates motion state with dropout recovery and flick glide.
        Returns:
            scroll_delta: Units ready for OS dispatch.
            velocity_display: Current speed in px/frame for HUD.
            is_coasting: True if scrolling via flick inertia glide.
        """
        # --- 1. ACTIVE PINCH TOUCH-DOWN / DRAG ---
        if is_pinched and current_pos is not None and hand_scale > 0:
            # If we were previously coasting, new pinch acts as instant brake
            self.is_coasting = False
            self.coast_velocity = 0.0
            self.ghost_frames_left = self.ghost_frame_tolerance

            current_y = current_pos[1]

            # First frame of engagement
            if not self.is_tracking or self.prev_y is None:
                self.is_tracking = True
                self.prev_y = current_y
                self.smoothed_delta_y = 0.0
                self.recent_velocities.clear()
                return 0.0, 0.0, False

            # Calculate raw displacement
            raw_dy = current_y - self.prev_y
            self.prev_y = current_y

            # Deadzone filter
            deadzone_px = self.deadzone_ratio * hand_scale
            if abs(raw_dy) < deadzone_px:
                filtered_dy = 0.0
            else:
                sign = 1.0 if raw_dy > 0 else -1.0
                filtered_dy = (abs(raw_dy) - deadzone_px) * sign

            # EMA smoothing
            self.smoothed_delta_y = (
                self.smoothing_alpha * filtered_dy
                + (1.0 - self.smoothing_alpha) * self.smoothed_delta_y
            )

            # Record recent velocity for flick release calculation
            self.recent_velocities.append(self.smoothed_delta_y)

            if abs(self.smoothed_delta_y) < 0.1:
                return 0.0, 0.0, False

            # Normalized displacement & acceleration curve
            norm_dy = self.smoothed_delta_y / hand_scale
            abs_norm = abs(norm_dy)
            accel = 1.0 + (abs_norm * self.flick_acceleration)
            scroll_units = -norm_dy * accel * self.speed_factor * 100.0

            return scroll_units, -self.smoothed_delta_y, False

        # --- 2. GHOST FRAME TOLERANCE (HAND BLUR / 1-2 FRAME DROPOUT) ---
        if self.is_tracking and self.ghost_frames_left > 0:
            self.ghost_frames_left -= 1
            # Replay recent velocity during ghost frame so motion doesn't stall
            if self.recent_velocities:
                avg_dy = sum(self.recent_velocities) / len(self.recent_velocities)
                norm_dy = avg_dy / max(hand_scale, 100.0)
                scroll_units = -norm_dy * self.speed_factor * 100.0
                return scroll_units, -avg_dy, False
            return 0.0, 0.0, False

        # --- 3. FLICK RELEASE TRIGGER (UNPINCH WITH MOMENTUM) ---
        if self.is_tracking and not is_pinched:
            # Calculate release velocity from the recent frames
            if self.recent_velocities:
                avg_release_vel = sum(self.recent_velocities) / len(self.recent_velocities)
                if abs(avg_release_vel) >= self.flick_min_velocity:
                    # Trigger inertia coasting
                    self.is_coasting = True
                    # Convert to scroll units with momentum
                    norm_vel = avg_release_vel / max(hand_scale, 100.0)
                    accel = 1.0 + (abs(norm_vel) * self.flick_acceleration)
                    self.coast_velocity = -norm_vel * accel * self.speed_factor * 90.0

            # Disengage active tracking
            self.is_tracking = False
            self.prev_y = None
            self.smoothed_delta_y = 0.0
            self.recent_velocities.clear()

        # --- 4. INERTIA COASTING DECAY ---
        if self.is_coasting:
            current_coast = self.coast_velocity
            self.coast_velocity *= self.friction_decay

            # Stop coasting once momentum decays below threshold
            if abs(self.coast_velocity) < 1.5:
                self.is_coasting = False
                self.coast_velocity = 0.0

            return current_coast, 0.0, True

        return 0.0, 0.0, False

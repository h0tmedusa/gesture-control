"""
Phase 5: OS Action Dispatcher Module
Dispatches operating system level actions safely using PyAutoGUI:
- Continuous variable-speed scrolling.
- Debounced discrete media control (Play/Pause via Media key or Space).
- Master pause/safety toggle.
"""

import time
import pyautogui

# Enable PyAutoGUI fail-safe: slam mouse to any screen corner to abort
pyautogui.FAILSAFE = True
# Set pause to 0 to prevent artificial lag in fast-dispatch loops
pyautogui.PAUSE = 0.0


class OSController:
    def __init__(
        self,
        media_cooldown_seconds: float = 1.2,
        scroll_multiplier: float = 1.0,
        point_scroll_speed: int = 15,            # Constant scroll ticks per frame for hands-free point scroll
        point_hold_duration_seconds: float = 0.25, # Must hold pointing pose for 250ms before scrolling starts
    ):
        self.media_cooldown_seconds = media_cooldown_seconds
        self.scroll_multiplier = scroll_multiplier
        self.point_scroll_speed = point_scroll_speed
        self.point_hold_duration_seconds = point_hold_duration_seconds

        self.last_media_trigger_time = 0.0
        self.is_enabled = True

        # State tracking for hands-free point scroll
        self.current_point_pose = None
        self.point_pose_start_time = 0.0

    def toggle_active(self):
        """Toggles the controller ON / OFF."""
        self.is_enabled = not self.is_enabled
        return self.is_enabled

    def dispatch_scroll(self, scroll_units: float):
        """
        Dispatches continuous scroll ticks.
        scroll_units: positive = scroll up, negative = scroll down.
        """
        if not self.is_enabled or abs(scroll_units) < 0.5:
            return 0

        # Windows PyAutoGUI scroll units: integers. Typically 10-100 per tick.
        ticks = int(scroll_units * self.scroll_multiplier)
        if ticks != 0:
            try:
                pyautogui.scroll(ticks)
            except Exception as e:
                print(f"[ERROR] Failed to dispatch scroll: {e}")
        return ticks

    def handle_point_scroll(self, pose: str):
        """
        Handles hands-free continuous scrolling for static pointing poses.
        Requires holding the pose for point_hold_duration_seconds before engaging.
        Returns:
            scrolling_active: bool
            direction: str ("UP", "DOWN", or None)
        """
        if not self.is_enabled:
            self.current_point_pose = None
            return False, None

        if pose not in ("POINT_UP", "POINT_DOWN"):
            self.current_point_pose = None
            self.point_pose_start_time = 0.0
            return False, None

        now = time.time()
        # First frame of this pointing pose
        if self.current_point_pose != pose:
            self.current_point_pose = pose
            self.point_pose_start_time = now
            return False, None

        # Check if held long enough to avoid accidental passing triggers
        elapsed = now - self.point_pose_start_time
        if elapsed >= self.point_hold_duration_seconds:
            direction = "UP" if pose == "POINT_UP" else "DOWN"
            ticks = self.point_scroll_speed if direction == "UP" else -self.point_scroll_speed
            try:
                pyautogui.scroll(ticks)
            except Exception as e:
                print(f"[ERROR] Failed to dispatch point scroll: {e}")
            return True, direction

        return False, None

    def trigger_play_pause(self):
        """
        Dispatches Play/Pause toggle with cooldown debouncing.
        Returns True if triggered, False if on cooldown.
        """
        if not self.is_enabled:
            return False

        current_time = time.time()
        time_since_last = current_time - self.last_media_trigger_time

        if time_since_last >= self.media_cooldown_seconds:
            self.last_media_trigger_time = current_time
            try:
                # 'playpause' simulates the hardware multimedia play/pause key
                pyautogui.press("playpause")
                print("[ACTION] Dispatched: PLAY / PAUSE")
                return True
            except Exception as e:
                # Fallback to 'space' if hardware media key fails
                try:
                    pyautogui.press("space")
                    print("[ACTION] Dispatched (fallback): SPACE")
                    return True
                except Exception as inner_e:
                    print(f"[ERROR] Failed to dispatch Play/Pause: {inner_e}")
                    return False
        return False

    def get_cooldown_progress(self):
        """Returns 0.0 (ready) to 1.0 (cooling down)."""
        elapsed = time.time() - self.last_media_trigger_time
        if elapsed >= self.media_cooldown_seconds:
            return 0.0
        return 1.0 - (elapsed / self.media_cooldown_seconds)
